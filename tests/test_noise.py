import numpy as np
import pytest
from helpers import deterministic_noise

from fiddy.noise import (
    NoiseFloor,
    default_probe_direction,
    estimate_ladder_noise,
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
    noise floor collapses to the rounding floor ``eps * |f|``. This is
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

    assert result.sigma == np.finfo(float).eps
    assert result.confident is False
    assert not noise_floor_is_confident(result)


def test_noise_floor_is_at_least_the_rounding_error():
    """Noise of ~1 ulp, below eps * |f|, is reported as eps * |f|."""
    rng = np.random.default_rng(0)
    f0 = 1e3
    jitter = rng.integers(-1, 2, size=15) * np.spacing(f0)
    probe_values = f0 + jitter
    values = iter(probe_values)
    direction = np.array([1.0])

    result = estimate_noise_floor(
        lambda x: np.array([next(values)]), np.array([0.0]), direction
    )

    assert result.confident
    assert result.sigma == np.finfo(float).eps * np.median(probe_values)
    assert result.sigmas[result.level - 1] < result.sigma


def test_rounding_floor_is_per_output_component():
    noise_amplitude = 1e-6

    def f(x):
        return np.array(
            [1e3 + deterministic_noise(x[0], noise_amplitude), 0.0, 3.0]
        )

    result = estimate_noise_floor(f, np.array([0.7]), np.array([1.0]))

    # Noise well above the rounding floor is unaffected.
    true_sigma = noise_amplitude / np.sqrt(3)
    assert 0.1 * true_sigma < result.sigma[0] < 10 * true_sigma
    # An exactly-zero component gets no floor.
    assert result.sigma[1] == 0.0
    # A constant component gets its own floor.
    assert result.sigma[2] == np.finfo(float).eps * 3.0


def test_rounding_floor_ignores_non_finite_probe_values():
    values = iter([np.nan] + [1e3 + i * 1e-3 for i in range(14)])

    result = estimate_noise_floor(
        lambda x: np.array([next(values)]), np.array([0.0]), np.array([1.0])
    )

    assert np.isfinite(result.sigma)


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
def test_ladder_noise_ignores_curvature_kinks_and_jumps_at_the_point(
    step_ratio,
):
    def f(x):
        # smooth + kink at 0.3 + jump at 0.3
        return np.exp(x) + 0.7 * abs(x - 0.3) + 2.0 * (x > 0.3)

    ladder = 1e-2 / step_ratio ** np.arange(8)
    sigma = estimate_ladder_noise(ladder, *_ladder_evaluations(f, 0.3, ladder))

    # Only rounding errors remain.
    assert sigma[0] < 1e-13


def test_ladder_noise_estimates_independent_noise():
    noise_sigma = 1e-6
    ladder = 1e-2 / 2.0 ** np.arange(8)
    estimates = []
    for seed in range(200):
        rng = np.random.default_rng(seed)

        def f(x, rng=rng):
            return np.sin(x) + rng.normal(0, noise_sigma)

        estimates.append(
            estimate_ladder_noise(ladder, *_ladder_evaluations(f, 0.3, ladder))
        )

    assert 0.8 < np.median(estimates) / noise_sigma < 1.25


@pytest.mark.parametrize(
    "ladder",
    [
        np.array([1e-2, 5e-3, 2.5e-3]),
        np.array([1e-2, 5e-3, 2.5e-3, 1e-3]),
        np.zeros(8),
    ],
    ids=["too-few-rungs", "not-geometric", "at-bound"],
)
def test_ladder_noise_is_zero_without_a_usable_ladder(ladder):
    def f(x):
        return np.sin(x) + deterministic_noise(x, 1e-6)

    sigma = estimate_ladder_noise(ladder, *_ladder_evaluations(f, 0.3, ladder))
    assert sigma[0] == 0.0


def test_ladder_noise_is_zero_for_non_finite_values():
    ladder = 1e-2 / 2.0 ** np.arange(8)
    f_0, f_plus, f_minus = _ladder_evaluations(np.sin, 0.3, ladder)
    f_plus[-1] = np.nan

    assert estimate_ladder_noise(ladder, f_0, f_plus, f_minus)[0] == 0.0
