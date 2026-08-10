import numpy as np
import pytest
from scipy.optimize import rosen

from fiddy import MethodId, get_derivative
from fiddy.directional_derivative import ComputerResult
from fiddy.success import RobustConsistency


class FakeDirectionalDerivative:
    """Minimal stand-in exposing only what `Consistency.method` calls."""

    def __init__(self, computer_results, analysis_results=None):
        self._computer_results = computer_results
        self._analysis_results = analysis_results or []

    def get_computer_results(self):
        return self._computer_results

    def get_analysis_results(self):
        return self._analysis_results


def test_robust_consistency_rejects_rounding_noise_dominated_step_sizes():
    """Regression test for a `RobustConsistency` (nee `Consistency`)
    robustness bug.

    A step size can become small enough that forward/backward/central all
    sample points within the target function's floating-point noise floor.
    They then become correlated (affected by the same rounding/cancellation
    error), and can spuriously agree with each other ("self-consistent")
    while being biased away from the true derivative. `RobustConsistency`
    must not blend such a step size into the final value while reporting
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

    with pytest.warns(UserWarning, match="rejected as inconsistent"):
        derivative = get_derivative(
            function=f,
            point=point,
            sizes=step_sizes,
            direction_ids=["x0"],
            method_ids=[MethodId.CENTRAL, MethodId.FORWARD, MethodId.BACKWARD],
            success_checker=RobustConsistency(rtol=0.1, atol=1e-5),
            relative_sizes=True,
        )

    success = bool(derivative.df["success"].values[0])
    value = float(np.squeeze(derivative.series.values[0]))

    # Reporting `success=True` is only acceptable if the value is actually
    # accurate; silently returning a significantly biased value (as the
    # unpatched algorithm does: ~868.5, a ~0.5% error) is the bug.
    if success:
        assert np.isclose(value, true_slope, rtol=1e-2)


def test_robust_consistency_averages_all_trustworthy_step_sizes():
    """A wide, but genuinely well-behaved, range of step sizes should not
    trigger spurious outlier rejection (and thus no rejection warning)."""

    def f(point):
        return np.array([rosen(point)])

    point = np.array([1.3, 0.7])
    # Chosen to have comparable precision across the whole range (see
    # `test_robust_consistency_narrows_to_the_most_precise_step_sizes`
    # below for what happens once the range gets wide enough that the
    # smallest steps are far more precise than the largest).
    step_sizes = [1e-2, 1e-3, 1e-4]

    derivative = get_derivative(
        function=f,
        point=point,
        sizes=step_sizes,
        direction_ids=["x0"],
        directions=[np.array([1.0, 0.0])],
        method_ids=[MethodId.CENTRAL, MethodId.FORWARD, MethodId.BACKWARD],
        success_checker=RobustConsistency(rtol=1e-2, atol=1e-8),
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


def test_robust_consistency_narrows_to_the_most_precise_step_sizes():
    """Rejection isn't only about *biased* step sizes (the motivating bug):
    a genuinely wide, noise-free step-size range can legitimately narrow
    down to just the handful of smallest, most precise steps, even though
    the larger, excluded ones weren't wrong -- just comparatively imprecise
    (ordinary, shrinking-with-h truncation error) next to a cluster that
    happens to already be near machine precision. The blended value must
    stay accurate either way.
    """

    def f(point):
        return np.array([rosen(point)])

    point = np.array([1.3, 0.7])
    step_sizes = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    with pytest.warns(UserWarning, match="rejected as inconsistent"):
        derivative = get_derivative(
            function=f,
            point=point,
            sizes=step_sizes,
            direction_ids=["x0"],
            directions=[np.array([1.0, 0.0])],
            method_ids=[MethodId.CENTRAL, MethodId.FORWARD, MethodId.BACKWARD],
            success_checker=RobustConsistency(rtol=1e-2, atol=1e-8),
        )

    assert bool(derivative.df["success"].values[0])
    value = float(np.squeeze(derivative.series.values[0]))
    h = 1e-6
    expected = (
        rosen(point + np.array([h, 0.0])) - rosen(point - np.array([h, 0.0]))
    ) / (2 * h)
    assert np.isclose(value, expected, rtol=1e-6)


def test_robust_consistency_warns_when_rejecting_step_sizes():
    """`RobustConsistency` should tell the user when it rejects a step size
    that looked self-consistent on its own -- otherwise a legitimate-seeming
    result could silently vanish from the blend without a trace."""
    results = [
        ComputerResult(
            method_id="central", value=10.0, metadata={"size_absolute": 1.0}
        ),
        ComputerResult(
            method_id="central", value=10.01, metadata={"size_absolute": 0.5}
        ),
        ComputerResult(
            method_id="central", value=9.99, metadata={"size_absolute": 0.1}
        ),
        ComputerResult(
            method_id="central",
            value=500.0,
            metadata={"size_absolute": 0.01},
        ),
    ]

    checker = RobustConsistency()
    with pytest.warns(UserWarning, match="1 step size"):
        success, value = checker.method(FakeDirectionalDerivative(results))

    assert success
    assert np.isclose(value, np.mean([10.0, 10.01, 9.99]))


class TestRejectOutliers:
    """Unit tests for `RobustConsistency._reject_outliers`, the
    order-independent iterative outlier-rejection pass over step sizes'
    per-size means (added to reject step sizes that are spuriously
    self-consistent, see
    `test_robust_consistency_rejects_rounding_noise_dominated_step_sizes`
    above)."""

    def test_below_min_trend_samples_keeps_everything(self):
        # An outlier (500.0) is present, but there are fewer candidates than
        # `min_trend_samples` -- too little data to estimate a spread, so no
        # trimming is attempted at all.
        checker = RobustConsistency(min_trend_samples=5)
        means = [10.0, 10.01, 500.0]
        assert checker._reject_outliers(means) == means

    def test_no_outliers_keeps_all(self):
        checker = RobustConsistency()
        means = [10.0, 10.01, 9.99, 10.02]
        assert checker._reject_outliers(means) == means

    def test_removes_single_outlier(self):
        checker = RobustConsistency()
        means = [10.0, 10.01, 9.99, 10.02, 500.0]
        trusted = checker._reject_outliers(means)
        assert trusted == [10.0, 10.01, 9.99, 10.02]

    def test_removes_multiple_outliers_iteratively(self):
        # Two outliers on opposite sides of the trustworthy cluster; both
        # must be dropped, one per iteration, worst-first.
        checker = RobustConsistency()
        means = [10.0, 10.01, 9.99, 10.02, 500.0, -500.0]
        trusted = checker._reject_outliers(means)
        assert trusted == [10.0, 10.01, 9.99, 10.02]

    def test_order_independent(self):
        # Dropping is based on value, not position: shuffling the input
        # must not change which candidates survive.
        checker = RobustConsistency()
        means = [500.0, 10.0, 10.01, 9.99, 10.02]
        trusted = checker._reject_outliers(means)
        assert sorted(trusted) == [9.99, 10.0, 10.01, 10.02]

    def test_respects_trend_n_sigma(self):
        means = [10.0, 10.01, 9.99, 10.02, 500.0]
        lenient_checker = RobustConsistency(trend_n_sigma=1e6)
        assert lenient_checker._reject_outliers(means) == means

        strict_checker = RobustConsistency(trend_n_sigma=5.0)
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
        checker = RobustConsistency()
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
        # `RobustConsistency.method`'s final blanket `isclose` check against
        # a non-NaN blended value still reports `success=False` overall).
        checker = RobustConsistency()
        means = [10.0, 10.01, np.nan]
        trusted = checker._reject_outliers(means)
        assert len(trusted) == 3
        assert np.isnan(trusted[-1])
