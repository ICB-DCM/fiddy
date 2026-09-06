import math

import numpy as np
import pytest
from helpers import deterministic_noise

from fiddy.estimate import (
    estimate_directional_derivative,
    estimate_gradient,
    estimate_jacobian,
)


def test_smooth_function_converges_accurately():
    def f(x):
        return np.array([np.sin(x[0]) * np.exp(0.1 * x[0])])

    x0 = np.array([1.3])
    direction = np.array([1.0])
    true_derivative = math.cos(1.3) * math.exp(0.13) + math.sin(
        1.3
    ) * 0.1 * math.exp(0.13)

    result = estimate_directional_derivative(f, x0, direction)

    assert result.status == "converged"
    assert abs(result.value - true_derivative) < 1e-6


def test_far_ladder_catches_a_kink_the_main_ladder_never_crosses():
    """Reproduces the mechanism behind a confirmed real bug ("ladder too
    coarse near events"): a parameter-space kink (continuous value,
    discontinuous derivative -- the same character as a real AMICI/SBML
    event-triggered discontinuity) sitting closer to the evaluation point
    than every main-ladder rung ever reaches. Every main rung straddles
    the kink and reports a self-consistent but wrong "average of both
    slopes" value; only the far ladder (anchored near machine epsilon,
    far below the kink's distance from the evaluation point) ever
    resolves the correct, single-branch derivative.

    Confirmed via 13 distinct real AMICI/SBML models (case 00026's
    reproducer: a step-size ladder from 54% to 0.4% relative reported a
    "converged" value 5.2x off the true one, whose crossing perturbation
    was ~0.1% relative -- below every main rung)."""
    c = 1.0
    slope_before, slope_after = 2.0, 5.0
    offset = 1e-5
    x0 = c - offset

    def f(x):
        xv = x[0]
        if xv < c:
            return np.array([slope_before * xv])
        const = c * (slope_before - slope_after)
        return np.array([slope_after * xv + const])

    result = estimate_directional_derivative(
        f, np.array([x0]), np.array([1.0]), noise_floor=1e-6
    )

    # The main ladder alone is confidently wrong -- self-consistently
    # converging to roughly the average of both branches' slopes, not
    # either one.
    assert abs(result.value - (slope_before + slope_after) / 2) < 1e-2
    # The far ladder, entirely below the kink's distance from `x0`,
    # resolves the correct (before-branch) derivative.
    assert result.far_extrapolation is not None
    assert abs(result.far_extrapolation.value - slope_before) < 1e-6
    # The disagreement between the two is flagged, not silently reported
    # as a confident, wrong "converged" result.
    assert result.cross_regime.suspected
    assert result.status == "discontinuity_suspected"


def test_noisy_function_converges_within_tolerance():
    noise_amplitude = 1e-8

    def f(x):
        return np.array(
            [np.sin(x[0]) + deterministic_noise(x[0], noise_amplitude)]
        )

    x0 = np.array([0.7])
    direction = np.array([1.0])
    true_derivative = math.cos(0.7)

    result = estimate_directional_derivative(f, x0, direction)

    assert result.status == "converged"
    assert abs(result.value - true_derivative) < 1e-4


def test_reproducible_across_repeated_calls():
    def f(x):
        return np.array([np.cos(x[0]) * x[0] ** 2])

    x0 = np.array([1.1])
    direction = np.array([1.0])
    first = estimate_directional_derivative(f, x0, direction)
    second = estimate_directional_derivative(f, x0, direction)

    assert first.value == second.value
    assert first.error_estimate == second.error_estimate
    assert first.status == second.status


def test_near_noise_floor_derivative_never_falsely_converges():
    """A direction whose true derivative is comparable in magnitude to
    the function's own noise floor must not be reported "converged"
    unless the reported value is actually accurate. This is the hard
    regime a single `argmin` over a possibly non-monotonic error sequence
    can get badly wrong (confidently, not just imprecisely) -- see
    `fiddy.extrapolation`'s module docstring for the real case that
    motivated the two-chain corroboration fix this regression-tests."""

    noise_amplitude = 1e-8

    def f(x):
        # Derivative at x0 is ~1e-8 * cos(0.4), i.e. comparable to the
        # noise amplitude itself -- the hard regime.
        return np.array(
            [1e-8 * np.sin(x[0]) + deterministic_noise(x[0], noise_amplitude)]
        )

    x0 = np.array([0.4])
    direction = np.array([1.0])
    true_derivative = 1e-8 * math.cos(0.4)

    result = estimate_directional_derivative(f, x0, direction)

    if result.status == "converged":
        relative_error = abs(result.value - true_derivative) / max(
            abs(true_derivative), 1e-300
        )
        assert relative_error < 0.05, (
            "Reported 'converged' but the value is significantly wrong "
            f"(relative error {relative_error:.2%}) -- exactly the "
            "false-convergence failure mode the corroboration fix is "
            "meant to prevent."
        )


def test_kink_is_flagged_not_silently_wrong():
    """`abs(x)` at its corner: central differences alone are exactly 0 for
    every step size (a self-consistent but wrong answer). The
    discontinuity check must catch this."""

    def f(x):
        return np.array([abs(x[0])])

    x0 = np.array([0.0])
    direction = np.array([1.0])

    result = estimate_directional_derivative(f, x0, direction)

    assert result.status == "discontinuity_suspected"
    assert result.discontinuity.suspected


def test_nondet_tol_avoids_false_kink_flag():
    """A function with legitimate, known value nondeterminism at the same
    point (e.g. simulate with vs. without sensitivities) should not be
    misclassified as a kink just because that jitter looks inconsistent."""

    amplitude = 1e-6

    def f(x):
        return np.array([np.sin(x[0]) + deterministic_noise(x[0], amplitude)])

    x0 = np.array([0.7])
    direction = np.array([1.0])

    result = estimate_directional_derivative(
        f, x0, direction, nondet_tol=amplitude
    )

    assert result.status != "discontinuity_suspected"


def test_estimate_gradient_defaults_to_standard_basis():
    def f(x):
        return np.array([x[0] ** 2 + 3 * x[1]])

    point = np.array([2.0, 1.0])
    results = estimate_gradient(f, point)

    assert len(results) == 2
    assert results[0].status == "converged"
    assert abs(results[0].value - 4.0) < 1e-4  # d/dx0 = 2*x0
    assert results[1].status == "converged"
    assert abs(results[1].value - 3.0) < 1e-4  # d/dx1 = 3


def test_estimate_gradient_matches_per_direction_calls():
    def f(x):
        return np.array([np.sin(x[0]) * np.cos(x[1])])

    point = np.array([0.6, -0.3])
    directions = list(np.eye(2))

    gradient_results = estimate_gradient(f, point, directions=directions)
    individual_results = [
        estimate_directional_derivative(f, point, d) for d in directions
    ]

    for g, i in zip(gradient_results, individual_results, strict=True):
        assert g.value == i.value
        assert g.status == i.status


def test_estimate_gradient_shares_base_point_evaluation():
    call_count = 0

    def f(x):
        nonlocal call_count
        call_count += 1
        return np.array([np.sum(np.sin(x))])

    point = np.array([0.3, 0.5, -0.2])
    directions = list(np.eye(3))
    n_rungs = 8
    n_rungs_far = 4

    estimate_gradient(f, point, directions=directions, n_rungs=n_rungs)

    # One shared noise-floor probe (15 points) + one shared f(x0) + each
    # direction's own 2*n_rungs main-ladder plus 2*n_rungs_far far-ladder
    # perturbed evaluations -- not a separate f(x0) or noise probe per
    # direction.
    expected = 15 + 1 + len(directions) * 2 * (n_rungs + n_rungs_far)
    assert call_count == expected


def test_estimate_directional_derivative_no_longer_crashes_on_raw_dict_function():
    """Regression test: previously, a raw (unwrapped) dict-returning
    function crashed with a TypeError, since `fiddy.output`'s bundling
    only happened if the caller pre-wrapped the function in
    `fiddy.Function` themselves. Entry points now auto-wrap internally
    instead -- the derivative computed is of the *first* named output
    only (documented, known behavior for this single-output entry point;
    use `estimate_jacobian` for every output at once)."""

    def f(x):
        return {"a": np.array([x[0] ** 2]), "b": np.array([x[0] ** 3])}

    point = np.array([2.0])
    direction = np.array([1.0])

    result = estimate_directional_derivative(f, point, direction)

    assert result.status == "converged"
    assert abs(result.value - 2 * 2.0) < 1e-6  # d/dx of "a" (first output)


def test_estimate_gradient_no_longer_crashes_on_raw_dict_function():
    def f(x):
        return {
            "a": np.array([x[0] ** 2 + 3 * x[1]]),
            "b": np.array([x[0] ** 3]),
        }

    point = np.array([2.0, 1.0])
    results = estimate_gradient(f, point)

    assert results[0].status == "converged"
    assert abs(results[0].value - 4.0) < 1e-4  # d(a)/dx0 = 2*x0
    assert results[1].status == "converged"
    assert abs(results[1].value - 3.0) < 1e-4  # d(a)/dx1 = 3


def test_estimate_jacobian_recovers_analytic_derivatives():
    def f(x):
        return {
            "a": np.array([x[0] ** 2 + x[1]]),
            "b": np.array([np.sin(x[0]) * x[1]]),
        }

    point = np.array([1.3, 0.6])
    jacobian = estimate_jacobian(f, point)

    assert jacobian.n_outputs == 2
    assert jacobian.n_directions == 2
    assert jacobian.statuses == [
        ["converged", "converged"],
        ["converged", "converged"],
    ]

    # d(a)/dx0 = 2*x0, d(a)/dx1 = 1
    assert abs(jacobian.estimates[0][0].value - 2 * point[0]) < 1e-6
    assert abs(jacobian.estimates[0][1].value - 1.0) < 1e-6
    # d(b)/dx0 = cos(x0)*x1, d(b)/dx1 = sin(x0)
    assert (
        abs(jacobian.estimates[1][0].value - math.cos(point[0]) * point[1])
        < 1e-6
    )
    assert abs(jacobian.estimates[1][1].value - math.sin(point[0])) < 1e-6


def test_estimate_jacobian_shares_one_batch_regardless_of_output_count():
    """Checking N outputs costs no more function evaluations than
    checking 1 -- the entire point of a shared-batch multi-output
    estimator."""
    call_count = 0

    def f(x):
        nonlocal call_count
        call_count += 1
        return {
            "a": np.array([np.sum(np.sin(x))]),
            "b": np.array([np.sum(np.cos(x))]),
            "c": np.array([np.sum(x**2)]),
        }

    point = np.array([0.3, 0.5, -0.2])
    n_rungs = 8
    n_rungs_far = 4

    estimate_jacobian(f, point, n_rungs=n_rungs)

    # One shared noise-floor probe (15 points) + one direct f(x0) call
    # (also populates `.schema`) + each direction's own 2*n_rungs main-
    # ladder plus 2*n_rungs_far far-ladder perturbed evaluations --
    # independent of how many outputs "f" bundles.
    expected = 15 + 1 + len(point) * 2 * (n_rungs + n_rungs_far)
    assert call_count == expected


def test_jacobian_estimate_output_lookup_by_name():
    def f(x):
        return {"a": np.array([x[0] ** 2]), "b": np.array([x[0] ** 3])}

    point = np.array([2.0])
    jacobian = estimate_jacobian(f, point)

    assert jacobian.output("a") == jacobian.estimates[0]
    assert jacobian.output("b") == jacobian.estimates[1]
    assert jacobian.output(0) == jacobian.estimates[0]

    with pytest.raises(KeyError, match="Unknown output"):
        jacobian.output("c")


def test_jacobian_estimate_name_lookup_requires_structured_output():
    def f(x):
        return np.array([x[0] ** 2])

    point = np.array([2.0])
    jacobian = estimate_jacobian(f, point)

    with pytest.raises(KeyError, match="did not return a named"):
        jacobian.output("a")

    assert jacobian.output(0) == jacobian.estimates[0]


def test_estimate_jacobian_flags_discontinuity_per_output_independently():
    """A genuine kink in one bundled output must not be masked, or
    falsely triggered, by another smooth output sharing the same ladder
    of evaluations."""

    def f(x):
        return {
            "smooth": np.array([np.sin(x[0])]),
            "kinked": np.array([abs(x[0])]),
        }

    point = np.array([0.0])
    jacobian = estimate_jacobian(f, point)

    assert jacobian.output("smooth")[0].status == "converged"
    assert jacobian.output("kinked")[0].status == "discontinuity_suspected"


def test_estimate_jacobian_noise_floor_differs_per_output():
    noise_amplitude = 1e-8

    def f(x):
        return {
            "noisy": np.array(
                [np.sin(x[0]) + deterministic_noise(x[0], noise_amplitude)]
            ),
            "smooth": np.array([np.cos(x[0])]),
        }

    point = np.array([0.7])
    jacobian = estimate_jacobian(f, point)

    noisy_sigma = jacobian.output("noisy")[0].noise.sigma
    smooth_sigma = jacobian.output("smooth")[0].noise.sigma
    assert noisy_sigma > 10 * smooth_sigma
    # Both should still resolve correctly despite the shared ladder being
    # driven by the noisier component.
    assert jacobian.output("noisy")[0].status == "converged"
    assert jacobian.output("smooth")[0].status == "converged"
    assert abs(jacobian.output("smooth")[0].value - (-math.sin(0.7))) < 1e-6


def _bound_crushed_shared_probe_setup():
    """A function/point/bounds combination that reliably (deterministically,
    not by numerical luck) crushes the *shared*, all-ones noise-floor
    probe: one dummy component (`x[2]`) sits exactly at its own declared
    upper bound, giving it zero room in the direction the shared probe
    would perturb it -- `fiddy.step_size.clamp_step_to_bounds` then
    clamps the *whole* shared step to `0`, so every probe point evaluates
    identically and the plateau-detection heuristic reports a spuriously
    confident-looking `sigma=0.0` (see `fiddy.noise.noise_floor_is_confident`).
    `x[0]` is a large-magnitude dummy component (drives the shared probe's
    step size via the dot product with the all-ones direction, but does
    not otherwise affect `f`); only `x[1]`'s own direction is checked --
    it is not near any bound and has a perfectly ordinary, resolvable
    noisy derivative, but is *cross-contaminated* by `x[2]`'s tightness
    under the shared probe. Regression case for a real failure found on
    `Boehm_JProteomeRes2014`/`Weber_BMC2015` (a parameter close to its
    own bound crushing every other direction's noise floor too).

    :return: `(f, point, bounds, directions)`, ready to pass to
        `estimate_gradient`.
    """

    def f(x):
        return np.array([x[1] ** 2 + deterministic_noise(x[1], 1e-6)])

    point = np.array([1e4, 1.0, 1.0])
    bounds = (
        np.array([-1e5, -10.0, -10.0]),
        np.array([1e5, 10.0, 1.0]),  # x[2] exactly at its own upper bound
    )
    directions = [np.array([0.0, 1.0, 0.0])]
    return f, point, bounds, directions


def test_shared_noise_floor_strategy_can_be_crushed_by_an_unrelated_bound():
    """Regression test for the failure mode `noise_floor_strategy` exists
    to fix: with the (still-available, opt-in) `"shared"` strategy, a
    single unrelated component sitting at its own bound crushes the
    noise floor for *every* direction, including this well-behaved one --
    turning a resolvable derivative into a falsely `"noise_dominated"`
    result."""
    f, point, bounds, directions = _bound_crushed_shared_probe_setup()
    result = estimate_gradient(
        f,
        point,
        directions=directions,
        bounds=bounds,
        noise_floor_strategy="shared",
    )[0]
    assert result.status == "noise_dominated"


def test_auto_noise_floor_strategy_escalates_when_shared_is_crushed():
    """`"auto"` (the default) must recover from exactly the failure
    demonstrated in
    `test_shared_noise_floor_strategy_can_be_crushed_by_an_unrelated_bound`,
    matching `"per_direction"`'s own (correct) result."""
    f, point, bounds, directions = _bound_crushed_shared_probe_setup()
    auto_result = estimate_gradient(
        f, point, directions=directions, bounds=bounds
    )[0]
    per_direction_result = estimate_gradient(
        f,
        point,
        directions=directions,
        bounds=bounds,
        noise_floor_strategy="per_direction",
    )[0]
    assert auto_result.status == "converged"
    assert abs(auto_result.value - 2.0) < 1e-3
    assert auto_result.value == per_direction_result.value


def test_noise_floor_strategy_auto_matches_shared_when_not_crushed():
    """`"auto"` must cost/behave identically to `"shared"` whenever the
    shared probe is already confident -- no silent behavior change for
    the common case this whole engine was already validated against."""

    def f(x):
        return np.array([math.sin(x[0])])

    point = np.array([0.6])
    shared_result = estimate_directional_derivative(
        f, point, np.array([1.0]), noise_floor_strategy="shared"
    )
    auto_result = estimate_directional_derivative(
        f, point, np.array([1.0]), noise_floor_strategy="auto"
    )
    assert auto_result.value == shared_result.value
    assert auto_result.error_estimate == shared_result.error_estimate
    assert auto_result.status == shared_result.status == "converged"


def test_invalid_noise_floor_strategy_raises():
    def f(x):
        return np.array([x[0]])

    with pytest.raises(ValueError, match="noise_floor_strategy"):
        estimate_directional_derivative(
            f, np.array([1.0]), np.array([1.0]), noise_floor_strategy="bogus"
        )


def test_point_violating_bounds_raises():
    def f(x):
        return np.array([x[0]])

    with pytest.raises(ValueError, match="bounds"):
        estimate_directional_derivative(
            f,
            np.array([1.0]),
            np.array([1.0]),
            bounds=(np.array([0.0]), np.array([0.5])),
        )


def test_point_within_bounds_is_not_rejected():
    def f(x):
        return np.array([x[0] ** 2])

    result = estimate_directional_derivative(
        f,
        np.array([0.5]),
        np.array([1.0]),
        bounds=(np.array([0.0]), np.array([1.0])),
    )
    assert result.status == "converged"
    assert abs(result.value - 1.0) < 1e-6
