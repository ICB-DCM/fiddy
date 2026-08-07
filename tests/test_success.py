import numpy as np
from scipy.optimize import rosen

from fiddy import MethodId, get_derivative
from fiddy.success import Consistency


def test_consistency_rejects_rounding_noise_dominated_step_sizes():
    """Regression test for a `Consistency` robustness bug.

    A step size can become small enough that forward/backward/central all
    sample points within the target function's floating-point noise floor.
    They then become correlated (affected by the same rounding/cancellation
    error), and can spuriously agree with each other ("self-consistent")
    while being biased away from the true derivative. `Consistency` must not
    blend such a step size into the final value while reporting
    `success=True`.
    """
    true_slope = 872.68
    noise_floor = 2e-7

    def f(point):
        x0 = point[0]
        value = -1023.447 + true_slope * (x0 - 1e-4)
        value += noise_floor * np.sin(1e8 * x0)
        return np.array(value)

    point = np.array([9.579126317171899e-05])
    step_sizes = [5e-1, 2e-1, 1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5]

    derivative = get_derivative(
        function=f,
        point=point,
        sizes=step_sizes,
        direction_ids=["x0"],
        method_ids=[MethodId.CENTRAL, MethodId.FORWARD, MethodId.BACKWARD],
        success_checker=Consistency(rtol=0.1, atol=1e-5),
        relative_sizes=True,
    )

    success = bool(derivative.df["success"].values[0])
    value = float(np.squeeze(derivative.series.values[0]))

    # Reporting `success=True` is only acceptable if the value is actually
    # accurate; silently returning a significantly biased value (as the
    # unpatched algorithm does: ~868.5, a ~0.5% error) is the bug.
    if success:
        assert np.isclose(value, true_slope, rtol=1e-2)


def test_consistency_averages_all_trustworthy_step_sizes():
    """A wide, but genuinely well-behaved, range of step sizes should not
    trigger spurious outlier rejection."""

    def f(point):
        return np.array([rosen(point)])

    point = np.array([1.3, 0.7])
    step_sizes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    derivative = get_derivative(
        function=f,
        point=point,
        sizes=step_sizes,
        direction_ids=["x0"],
        directions=[np.array([1.0, 0.0])],
        method_ids=[MethodId.CENTRAL, MethodId.FORWARD, MethodId.BACKWARD],
        success_checker=Consistency(rtol=1e-2, atol=1e-8),
    )

    assert bool(derivative.df["success"].values[0])
    value = float(np.squeeze(derivative.series.values[0]))
    # d/dx0 rosen([1.3, 0.7]) = -2*100*(0.7 - 1.3**2)*1.3 * ... expected via finite differences,
    # so just check against a fine central-difference estimate directly.
    h = 1e-6
    expected = (
        rosen(point + np.array([h, 0.0])) - rosen(point - np.array([h, 0.0]))
    ) / (2 * h)
    assert np.isclose(value, expected, rtol=1e-3)


class TestRejectOutliers:
    """Unit tests for `Consistency._reject_outliers`, the order-independent
    iterative outlier-rejection pass over step sizes' per-size means (added
    to reject step sizes that are spuriously self-consistent, see
    `test_consistency_rejects_rounding_noise_dominated_step_sizes` above)."""

    def test_below_min_trend_samples_keeps_everything(self):
        # An outlier (500.0) is present, but there are fewer candidates than
        # `min_trend_samples` -- too little data to estimate a spread, so no
        # trimming is attempted at all.
        checker = Consistency(min_trend_samples=5)
        means = [10.0, 10.01, 500.0]
        assert checker._reject_outliers(means) == means

    def test_no_outliers_keeps_all(self):
        checker = Consistency()
        means = [10.0, 10.01, 9.99, 10.02]
        assert checker._reject_outliers(means) == means

    def test_removes_single_outlier(self):
        checker = Consistency()
        means = [10.0, 10.01, 9.99, 10.02, 500.0]
        trusted = checker._reject_outliers(means)
        assert trusted == [10.0, 10.01, 9.99, 10.02]

    def test_removes_multiple_outliers_iteratively(self):
        # Two outliers on opposite sides of the trustworthy cluster; both
        # must be dropped, one per iteration, worst-first.
        checker = Consistency()
        means = [10.0, 10.01, 9.99, 10.02, 500.0, -500.0]
        trusted = checker._reject_outliers(means)
        assert trusted == [10.0, 10.01, 9.99, 10.02]

    def test_order_independent(self):
        # Dropping is based on value, not position: shuffling the input
        # must not change which candidates survive.
        checker = Consistency()
        means = [500.0, 10.0, 10.01, 9.99, 10.02]
        trusted = checker._reject_outliers(means)
        assert sorted(trusted) == [9.99, 10.0, 10.01, 10.02]

    def test_respects_trend_n_sigma(self):
        means = [10.0, 10.01, 9.99, 10.02, 500.0]
        lenient_checker = Consistency(trend_n_sigma=1e6)
        assert lenient_checker._reject_outliers(means) == means

        strict_checker = Consistency(trend_n_sigma=5.0)
        assert strict_checker._reject_outliers(means) == [
            10.0,
            10.01,
            9.99,
            10.02,
        ]

    def test_vector_valued_drops_whole_candidate_on_any_element_outlier(self):
        # A candidate that's fine in one output element but a severe
        # outlier in another must still be dropped entirely (not just
        # masked in the bad element) -- "badness" is reduced across all
        # output dimensions before picking the worst candidate.
        checker = Consistency()
        means = [
            np.array([10.0, 5.0]),
            np.array([10.01, 5.01]),
            np.array([9.99, 500.0]),  # fine in element 0, an outlier in 1
        ]
        trusted = checker._reject_outliers(means)
        assert len(trusted) == 2
        assert all(
            np.array_equal(t, means[i])
            for t, i in zip(trusted, [0, 1], strict=True)
        )

    def test_nan_candidate_is_never_flagged_as_worst(self):
        # Known, documented limitation: `nanargmax` ignores NaNs, so a
        # candidate whose mean is entirely NaN can never be selected as
        # "the worst" and is left in the trusted set untouched (harmless in
        # practice: it doesn't shift `np.nanmean` of the final value, and
        # `Consistency.method`'s final blanket `isclose` check against a
        # non-NaN blended value still reports `success=False` overall).
        checker = Consistency()
        means = [10.0, 10.01, np.nan]
        trusted = checker._reject_outliers(means)
        assert len(trusted) == 3
        assert np.isnan(trusted[-1])
