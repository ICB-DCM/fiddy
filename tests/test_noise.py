import numpy as np
import pytest
from helpers import deterministic_noise

from fiddy.noise import (
    NoiseFloor,
    _estimate_ladder_noise,
    default_probe_direction,
    estimate_model_noise_floor,
    estimate_noise_floor,
    noise_floor_is_confident,
)


def test_smooth_function_has_low_confidence_or_tiny_floor():
    """A perfectly smooth function has no real noise; the estimator should
    either report low confidence (no plateau found within resolvable
    orders) or a floor near machine precision -- never a large, confident
    "noise floor" that would force an unnecessarily large step size."""

    def f(x):
        return np.array([np.sin(x[0])])

    result = estimate_noise_floor(f, np.array([0.7]), np.array([1.0]))
    assert (not result.confident) or result.sigma < 1e-6


def test_noisy_function_recovers_injected_noise_level():
    noise_amplitude = 1e-8

    def f(x):
        return np.array(
            [np.sin(x[0]) + deterministic_noise(x[0], noise_amplitude)]
        )

    result = estimate_noise_floor(f, np.array([0.7]), np.array([1.0]))

    assert result.confident
    # True standard deviation of Uniform(-amplitude, amplitude).
    true_sigma = noise_amplitude / np.sqrt(3)
    assert 0.1 * true_sigma < result.sigma < 10 * true_sigma


def test_estimate_is_reproducible():
    def f(x):
        return np.array([np.cos(x[0]) * x[0] ** 2])

    point = np.array([1.3])
    direction = np.array([1.0])
    first = estimate_noise_floor(f, point, direction)
    second = estimate_noise_floor(f, point, direction)
    assert first.sigma == second.sigma
    assert first.level == second.level


def test_multi_output_returns_one_sigma_per_component_from_one_probe_batch():
    """A function returning several components should get one noise-floor
    estimate per component, from the same probe batch used for a
    single-output function -- no extra evaluations."""
    noise_amplitude = 1e-8
    call_count = 0

    def f(x):
        nonlocal call_count
        call_count += 1
        # Component 0: noisy. Component 1: perfectly smooth (no noise).
        return np.array(
            [
                np.sin(x[0]) + deterministic_noise(x[0], noise_amplitude),
                np.cos(x[0]),
            ]
        )

    result = estimate_noise_floor(f, np.array([0.7]), np.array([1.0]))

    assert call_count == 15  # one shared probe batch, not one per component
    assert isinstance(result.sigma, np.ndarray)
    assert result.sigma.shape == (2,)
    true_sigma = noise_amplitude / np.sqrt(3)
    assert 0.1 * true_sigma < result.sigma[0] < 10 * true_sigma
    # The smooth component should not inherit the noisy one's floor.
    assert result.sigma[1] < result.sigma[0]


def test_default_probe_direction_is_unit_all_ones():
    point = np.array([1.0, 2.0, -3.0, 0.5])
    direction = default_probe_direction(point)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    # All components equal (it's a normalized all-ones vector).
    assert np.allclose(direction, direction[0])


def test_estimate_model_noise_floor_uses_one_shared_probe():
    call_count = 0

    def f(point):
        nonlocal call_count
        call_count += 1
        return np.array([np.sum(np.sin(point))])

    point = np.array([0.5, 1.5, -0.5])
    result = estimate_model_noise_floor(f, point)

    assert isinstance(result.sigma, float)
    # One probing batch only (n_points default), not one per dimension.
    assert call_count == 15


def test_bounds_can_crush_the_probe_step_to_zero():
    """A point sitting exactly at its own declared bound gives the probe
    zero room to perturb it -- `clamp_step_to_bounds` clamps the whole
    step to `0`, so every probe point evaluates identically and the
    plateau-detection heuristic reports a degenerate `sigma=0.0`. This is
    the mechanism `noise_floor_is_confident` exists to flag (see
    `fiddy.estimate`'s `noise_floor_strategy="auto"` escalation, added
    after this was found to silently corrupt otherwise-unrelated
    directions' noise floors on real bounded models)."""
    point = np.array([1.0])
    direction = np.array([1.0])
    bounds = (np.array([0.0]), np.array([1.0]))  # point already at upper bound

    result = estimate_noise_floor(
        lambda x: np.array([x[0] ** 2]), point, direction, bounds=bounds
    )

    assert result.sigma == 0.0
    assert result.confident is False
    assert not noise_floor_is_confident(result)


def test_noise_floor_is_confident_true_case():
    assert noise_floor_is_confident(
        NoiseFloor(sigma=1e-9, level=5, confident=True, sigmas=[])
    )


def test_noise_floor_is_confident_false_for_zero_sigma():
    # A degenerate NoiseFloor can report confident=True with sigma=0.0
    # (every probe point evaluating identically looks like a perfect,
    # confident plateau to the plateau-detection heuristic) -- this must
    # still be treated as unusable.
    assert not noise_floor_is_confident(
        NoiseFloor(sigma=0.0, level=1, confident=True, sigmas=[])
    )


def test_noise_floor_is_confident_false_for_unconfident():
    assert not noise_floor_is_confident(
        NoiseFloor(sigma=1e-3, level=None, confident=False, sigmas=[])
    )


def test_noise_floor_is_confident_multi_output_requires_all_confident():
    assert not noise_floor_is_confident(
        NoiseFloor(
            sigma=np.array([1e-9, 1e-9]),
            level=np.array([5, -1]),
            confident=np.array([True, False]),
            sigmas=[],
        )
    )


def _ladder_evaluations(f, x0, ladder):
    f_0 = np.array([f(x0)])
    f_plus = np.array([[f(x0 + h)] for h in ladder])
    f_minus = np.array([[f(x0 - h)] for h in ladder])
    return f_0, f_plus, f_minus


@pytest.mark.parametrize("step_ratio", [2.0, 3.0])
def test_ladder_noise_cancels_curvature_and_a_kink_and_jump_at_the_point(
    step_ratio,
):
    def f(x):
        return np.exp(x) + 0.7 * abs(x - 0.3) + 2.0 * (x > 0.3)

    ladder = 1e-2 / step_ratio ** np.arange(8)
    sigma = _estimate_ladder_noise(
        ladder, *_ladder_evaluations(f, 0.3, ladder)
    )

    # Only rounding errors remain.
    assert sigma[0] < 1e-13


@pytest.mark.parametrize("step_ratio", [2.0, 3.0])
def test_ladder_noise_estimates_independent_noise(step_ratio):
    noise_sigma = 1e-6
    ladder = 1e-2 / step_ratio ** np.arange(8)
    estimates = []
    for seed in range(200):
        rng = np.random.default_rng(seed)

        def f(x, rng=rng):
            return np.sin(x) + rng.normal(0, noise_sigma)

        estimates.append(
            _estimate_ladder_noise(
                ladder, *_ladder_evaluations(f, 0.3, ladder)
            )
        )

    assert 0.8 < np.median(estimates) / noise_sigma < 1.25


def test_ladder_noise_is_per_output_component():
    rng = np.random.default_rng(0)
    noise_sigmas = np.array([1e-6, 1e-9])
    ladder = 1e-2 / 2.0 ** np.arange(8)

    def f(x):
        return np.sin(x) + rng.normal(0, noise_sigmas)

    f_0 = f(0.3)
    f_plus = np.array([f(0.3 + h) for h in ladder])
    f_minus = np.array([f(0.3 - h) for h in ladder])
    sigma = _estimate_ladder_noise(ladder, f_0, f_plus, f_minus)

    assert sigma.shape == (2,)
    assert np.all(noise_sigmas / 3 < sigma)
    assert np.all(sigma < 3 * noise_sigmas)


@pytest.mark.parametrize(
    "ladder",
    [
        1e-2 / 2.0 ** np.arange(5),
        np.array([1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5]),
        np.zeros(8),
    ],
    ids=["too-few-rungs", "not-geometric", "at-bound"],
)
def test_ladder_noise_is_nan_without_a_usable_ladder(ladder):
    def f(x):
        return np.sin(x) + deterministic_noise(x, 1e-6)

    sigma = _estimate_ladder_noise(
        ladder, *_ladder_evaluations(f, 0.3, ladder)
    )
    assert np.isnan(sigma[0])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_ladder_noise_is_nan_for_non_finite_values(value):
    ladder = 1e-2 / 2.0 ** np.arange(8)
    f_0, f_plus, f_minus = _ladder_evaluations(np.sin, 0.3, ladder)
    f_plus[-1] = value

    assert np.isnan(_estimate_ladder_noise(ladder, f_0, f_plus, f_minus)[0])
