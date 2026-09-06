import numpy as np
from helpers import deterministic_noise

from fiddy.noise import (
    NoiseFloor,
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
