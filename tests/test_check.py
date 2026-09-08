import math

import numpy as np
import pytest

from fiddy.check import check_gradient, check_jacobian
from fiddy.function import FunctionEvaluationError


def test_correct_gradient_passes():
    def f(x):
        return np.array([np.sin(x[0]) * np.cos(x[1])])

    point = np.array([0.6, -0.3])
    expected = [
        math.cos(0.6) * math.cos(-0.3),
        -math.sin(0.6) * math.sin(-0.3),
    ]

    result = check_gradient(f, point, expected)

    assert result.success
    result.assert_success()  # must not raise


def test_wrong_gradient_fails():
    def f(x):
        return np.array([np.sin(x[0])])

    point = np.array([0.7])
    wrong_expected = [math.cos(0.7) + 1.0]  # deliberately wrong

    result = check_gradient(f, point, wrong_expected)

    assert not result.success
    with pytest.raises(AssertionError, match="FAILED"):
        result.assert_success()


def test_kink_is_inconclusive_not_failed():
    """A direction fiddy cannot reliably check (a kink) must not silently
    count as a check failure just because the FD estimate happens to
    disagree with the supplied gradient -- mismatches near kinks are
    expected, not bugs (see e.g. CS231n's gradient-checking notes)."""

    def f(x):
        return np.array([abs(x[0])])

    point = np.array([0.0])
    # abs(x) is not differentiable at 0; supply some arbitrary "expected".
    expected = [0.5]

    result = check_gradient(f, point, expected)

    assert result.direction_results[0].outcome == "inconclusive"
    assert result.success  # inconclusive directions don't fail the check


def test_tolerance_is_auto_derived_per_direction():
    """Each direction's tolerance is `safety_factor * error_estimate`, floored at the
    engine's shared noise-derived `tol` -- so per-direction error
    estimates (which do vary with each direction's own scale) drive the
    tolerance whenever they exceed that shared floor, rather than one
    fixed constant being used everywhere."""

    def f(x):
        return np.array([np.sin(x[0]) + np.cos(x[1]) * 100])

    point = np.array([0.4, 0.8])
    expected = [math.cos(0.4), -100 * math.sin(0.8)]

    result = check_gradient(f, point, expected)

    assert result.success
    error_estimates = [
        r.estimate.error_estimate for r in result.direction_results
    ]
    assert error_estimates[0] != error_estimates[1]


def test_rtol_floor_passes_a_large_magnitude_direction_with_tiny_relative_error():
    """A direction whose derivative barely varies with fiddy's own probe
    (here: an exactly linear function, so fiddy's re-evaluation noise is
    measured as near-zero) must not demand near-exact agreement with an
    independently-supplied `expected` that differs only by a tiny
    *relative* amount on a large-magnitude value -- exactly the real
    failure mode found checking AMICI models (a pure output-scaling
    parameter whose FD estimate and analytic sensitivity agreed to
    ~1e-10 relative error, but failed the old, purely noise-derived
    tolerance by ~900x). Passing `rtol=0.0` reproduces the old
    (pre-fix) behavior and must still fail, confirming this scenario
    genuinely needs the new floor, not something else already covering
    it."""

    def f(x):
        return np.array([1e8 * x[0]])

    point = np.array([5.0])
    true_derivative = 1e8
    expected = [true_derivative * (1 + 1e-10)]

    result = check_gradient(f, point, expected)
    assert result.success
    result.assert_success()  # must not raise

    result_without_rtol = check_gradient(f, point, expected, rtol=0.0)
    assert not result_without_rtol.success


def test_rtol_floor_does_not_mask_a_real_relative_error():
    """A negative control: the same large-magnitude scenario as above,
    but `expected` differs by a relative amount (1e-4) clearly larger
    than the default `rtol` (1e-8) -- representative of a genuine
    small-percentage sensitivity bug (e.g. an under-tightened solver
    tolerance). Must still fail; guards against `rtol` ever being
    loosened far enough to swallow this class of real bug."""

    def f(x):
        return np.array([1e8 * x[0]])

    point = np.array([5.0])
    true_derivative = 1e8
    wrong_expected = [true_derivative * (1 + 1e-4)]

    result = check_gradient(f, point, wrong_expected)

    assert not result.success


def test_rtol_floor_does_not_mask_a_wrong_near_zero_derivative():
    """The new relative floor must stay a no-op when the true derivative
    is near zero (common for `check_jacobian`'s independent output
    components) -- `rtol * max(|value|, |expected|)` only grows large
    enough to matter when *both* sides are large-magnitude, so a clearly
    wrong small-magnitude `expected` must still fail."""

    def f(x):
        return np.array([x[0]])

    point = np.array([1.0, 2.0])
    direction = np.array([0.0, 1.0])  # derivative wrt x[1] is exactly 0
    wrong_expected = [0.05]

    result = check_gradient(f, point, wrong_expected, directions=[direction])

    assert not result.success


def test_mismatched_expected_length_raises():
    def f(x):
        return np.array([x[0] + x[1]])

    point = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="entries"):
        check_gradient(f, point, expected=[1.0])


def test_random_directions_mode_checks_full_gradient_projection():
    def f(x):
        return np.array([x[0] ** 2 + 3 * x[1] - x[2]])

    point = np.array([2.0, 1.0, 0.5])
    full_gradient = [2 * point[0], 3.0, -1.0]

    result = check_gradient(
        f, point, expected=full_gradient, random_directions=5, rng=0
    )

    assert len(result.direction_results) == 5
    assert result.success


def test_random_directions_and_directions_are_mutually_exclusive():
    def f(x):
        return np.array([x[0]])

    with pytest.raises(ValueError, match="only one of"):
        check_gradient(
            f,
            np.array([1.0]),
            expected=[1.0],
            directions=[np.array([1.0])],
            random_directions=3,
        )


def test_report_mentions_failed_direction_id():
    def f(x):
        return np.array([x[0] ** 2 + x[1] ** 2])

    point = np.array([1.0, 1.0])
    expected = [2.0, 100.0]  # second entry deliberately wrong

    result = check_gradient(f, point, expected)

    assert not result.success
    with pytest.raises(AssertionError) as error:
        result.assert_success()
    message = str(error.value)
    assert "1" in message  # the failing direction's index


def test_direction_labels_appear_alongside_not_instead_of_the_index():
    """`direction_labels` is purely cosmetic -- the bare `direction_index`
    must still be present (as the `.df`/report's own index), with the
    label added as an extra column/annotation, never replacing it."""

    def f(x):
        return np.array([x[0] ** 2 + x[1] ** 2])

    point = np.array([1.0, 1.0])
    expected = [2.0, 100.0]  # second entry deliberately wrong

    result = check_gradient(f, point, expected, direction_labels=["k1", "k2"])

    assert list(result.df.index) == [0, 1]
    assert list(result.df["direction_label"]) == ["k1", "k2"]

    with pytest.raises(AssertionError) as error:
        result.assert_success()
    message = str(error.value)
    assert "1" in message  # the index is still there
    assert "k2" in message  # and so is the label


def test_no_direction_labels_means_no_label_column():
    """The `direction_label` column should not clutter the report for
    callers who never supply labels."""

    def f(x):
        return np.array([x[0]])

    result = check_gradient(f, np.array([1.0]), expected=[1.0])

    assert "direction_label" not in result.df.columns


def test_direction_labels_length_mismatch_raises():
    def f(x):
        return np.array([x[0] ** 2 + x[1] ** 2])

    point = np.array([1.0, 1.0])
    expected = [2.0, 2.0]

    with pytest.raises(ValueError, match="direction_labels"):
        check_gradient(f, point, expected, direction_labels=["only_one"])


def test_direction_labels_incompatible_with_random_directions():
    def f(x):
        return np.array([x[0]])

    with pytest.raises(ValueError, match="random_directions"):
        check_gradient(
            f,
            np.array([1.0]),
            expected=[1.0],
            random_directions=3,
            direction_labels=["k1"],
        )


def test_check_gradient_no_longer_crashes_on_raw_dict_function():
    """Regression test: previously, a raw (unwrapped) dict-returning
    function crashed with a TypeError, since `fiddy.output`'s bundling
    only happened if the caller pre-wrapped the function in
    `fiddy.Function` themselves. `check_gradient` now auto-wraps
    internally instead -- the derivative checked is of the *first* named
    output only (documented, known behavior for this single-output entry
    point; use `check_jacobian` for every output at once)."""

    def f(x):
        return {"a": np.array([x[0] ** 2]), "b": np.array([x[0] ** 3])}

    point = np.array([2.0])
    expected = [2 * 2.0]  # d/dx of "a" (first output) only

    result = check_gradient(f, point, expected)

    assert result.success


def test_check_jacobian_passes_with_correct_expected_dict():
    def f(x):
        return {
            "a": np.array([x[0] ** 2 + x[1]]),
            "b": np.array([np.sin(x[0]) * x[1]]),
        }

    point = np.array([1.3, 0.6])
    expected = {
        "a": np.array([[2 * point[0], 1.0]]),
        "b": np.array([[math.cos(point[0]) * point[1], math.sin(point[0])]]),
    }

    result = check_jacobian(f, point, expected)

    assert result.success
    result.assert_success()  # must not raise


def test_check_jacobian_passes_with_correct_expected_array():
    def f(x):
        return np.array([x[0] ** 2 + x[1], np.sin(x[0]) * x[1]])

    point = np.array([1.3, 0.6])
    expected = np.array(
        [
            [2 * point[0], 1.0],
            [math.cos(point[0]) * point[1], math.sin(point[0])],
        ]
    )

    result = check_jacobian(f, point, expected)

    assert result.success


def test_check_jacobian_fails_with_wrong_expected():
    def f(x):
        return {"a": np.array([x[0] ** 2])}

    point = np.array([1.3])
    wrong_expected = {"a": np.array([[999.0]])}

    result = check_jacobian(f, point, wrong_expected)

    assert not result.success
    with pytest.raises(AssertionError):
        result.assert_success()


def test_check_jacobian_inherits_the_rtol_floor():
    """`check_jacobian` shares `_check_direction` with `check_gradient`,
    so it must inherit the same relative-tolerance floor for a
    large-magnitude output with a tiny relative error against `expected`
    (see `test_rtol_floor_passes_a_large_magnitude_direction_with_tiny_relative_error`)."""

    def f(x):
        return {"a": np.array([1e8 * x[0]])}

    point = np.array([5.0])
    true_derivative = 1e8
    expected = {"a": np.array([[true_derivative * (1 + 1e-10)]])}

    result = check_jacobian(f, point, expected)

    assert result.success
    result.assert_success()  # must not raise


def test_check_jacobian_output_lookup_by_name():
    def f(x):
        return {"a": np.array([x[0] ** 2]), "b": np.array([x[0] ** 3])}

    point = np.array([2.0])
    expected = {
        "a": np.array([[2 * point[0]]]),
        "b": np.array([[3 * point[0] ** 2]]),
    }

    result = check_jacobian(f, point, expected)

    assert result.output("a").success
    assert result.output("b").success


def test_check_jacobian_missing_output_in_expected_raises():
    def f(x):
        return {"a": np.array([x[0] ** 2]), "b": np.array([x[0] ** 3])}

    point = np.array([2.0])
    incomplete_expected = {"a": np.array([[1.0]])}  # missing "b"

    with pytest.raises(ValueError, match="missing"):
        check_jacobian(f, point, incomplete_expected)


def test_check_jacobian_wrong_shape_expected_raises():
    def f(x):
        return np.array([x[0] ** 2, x[0] ** 3])

    point = np.array([2.0])
    wrong_shape_expected = np.array([1.0, 2.0, 3.0])  # 3 entries, not 2

    with pytest.raises(ValueError, match="shape"):
        check_jacobian(f, point, wrong_shape_expected)


def test_check_jacobian_output_labels_appear_alongside_not_instead_of_the_index():
    """`output_labels` is purely cosmetic -- the bare output index (as
    used in `assert_success`'s "--- output N ---" headers) must still be
    present, with the label added, never replacing it. `direction_labels`
    is applied identically to every output's own directions."""

    def f(x):
        return {"a": np.array([x[0] ** 2]), "b": np.array([x[0] ** 3])}

    point = np.array([2.0])
    expected = {
        "a": np.array([[999.0]]),  # deliberately wrong
        "b": np.array([[3 * point[0] ** 2]]),
    }

    result = check_jacobian(
        f,
        point,
        expected,
        direction_labels=["k1"],
        output_labels=["a[0]", "b[0]"],
    )

    assert not result.success
    with pytest.raises(AssertionError) as error:
        result.assert_success()
    message = str(error.value)
    assert "output 0" in message  # the index is still there
    assert "a[0]" in message  # and so is the label
    assert "k1" in message  # direction_labels forwarded too


def test_check_jacobian_output_labels_length_mismatch_raises():
    def f(x):
        return {"a": np.array([x[0] ** 2]), "b": np.array([x[0] ** 3])}

    point = np.array([2.0])
    expected = {
        "a": np.array([[2 * point[0]]]),
        "b": np.array([[3 * point[0] ** 2]]),
    }

    with pytest.raises(ValueError, match="output_labels"):
        check_jacobian(f, point, expected, output_labels=["only_one"])


def test_bounds_prevent_a_domain_violation_that_would_otherwise_fail():
    """Regression test: a finite-difference step evaluated outside a
    function's known valid domain is not a hypothetical concern -- e.g.
    `sqrt` is undefined for negative input. Close to the domain's edge,
    an unbounded probe/step can (and here does) push the evaluated point
    negative; supplying `bounds` must keep every evaluation inside it."""

    def f(x):
        return np.array([np.sqrt(x[0])])

    point = np.array([0.01])
    expected = [1 / (2 * math.sqrt(point[0]))]

    with pytest.raises(FunctionEvaluationError):
        check_gradient(f, point, expected)

    result = check_gradient(
        f, point, expected, bounds=(np.array([0.0]), np.array([np.inf]))
    )
    assert result.success
