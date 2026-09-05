import numpy as np

from fiddy.step_size import build_step_ladder


def test_ladder_shape_and_ordering():
    point = np.array([1.0])
    direction = np.array([1.0])
    ladder = build_step_ladder(point, direction, noise_floor=1e-9, n_rungs=8)

    assert ladder.shape == (8,)
    # Strictly decreasing (each rung smaller than the last).
    assert np.all(np.diff(ladder) < 0)
    assert np.all(ladder > 0)


def test_ladder_scales_with_step_ratio():
    point = np.array([1.0])
    direction = np.array([1.0])
    ladder = build_step_ladder(
        point, direction, noise_floor=1e-9, n_rungs=4, step_ratio=2.0
    )
    ratios = ladder[:-1] / ladder[1:]
    assert np.allclose(ratios, 2.0)


def test_ladder_uses_absolute_floor_near_origin():
    """Near the origin, the point-magnitude scale should fall back to an
    absolute step (max(|x . d|, 1.0)), not collapse to zero."""
    point = np.array([1e-12])
    direction = np.array([1.0])
    ladder = build_step_ladder(point, direction, noise_floor=1e-9)
    assert ladder[0] > 0
    assert np.isclose(ladder[0], 1.0 * 1e-9 ** (1 / 3))


def test_smaller_noise_floor_gives_smaller_initial_step():
    point = np.array([1.0])
    direction = np.array([1.0])
    ladder_low_noise = build_step_ladder(point, direction, noise_floor=1e-14)
    ladder_high_noise = build_step_ladder(point, direction, noise_floor=1e-6)
    assert ladder_low_noise[0] < ladder_high_noise[0]


def test_huge_noise_floor_is_clamped_to_a_sane_relative_step():
    """Regression test (found via broad real-model validation against
    `Oliveira_NatCommun2021`): a genuinely huge measured noise floor
    (there, ~1.7e11 -- larger than the function's own value at that
    point, from a stiff model's shared all-ones noise probe mistaking
    real nonlinear sensitivity for noise) must not blow up the largest
    rung to thousands of times the point's own scale -- that evaluates
    the function at nonsensical perturbed points, which was observed to
    make every direction's estimate `NaN`."""
    point = np.array([1.0])
    direction = np.array([1.0])
    huge_noise_floor = 1.7e11  # same order of magnitude as the real case

    ladder = build_step_ladder(point, direction, noise_floor=huge_noise_floor)

    # Without the clamp this would be ~5546 (huge_noise_floor ** (1/3)).
    assert ladder[0] <= 1.0  # max_relative_step (default 1.0) * scale (1.0)


def test_max_relative_step_bounds_the_largest_rung():
    point = np.array([2.0])
    direction = np.array([1.0])
    ladder = build_step_ladder(
        point, direction, noise_floor=1e6, max_relative_step=0.1
    )
    assert ladder[0] <= 0.1 * 2.0


def test_zero_noise_floor_does_not_collapse_the_ladder_to_zero():
    """Regression test: a bundled output component that is exactly
    constant along `direction` (e.g. a fixed noise parameter that doesn't
    depend on the perturbed parameters at all) has a true noise floor of
    exactly 0.0, not merely small. Unfloored, `h0 ~ noise_floor ** (1/3)`
    is then also exactly 0.0, collapsing the whole ladder to zeros --
    which turns the central-difference formula
    `(f_plus - f_minus) / (2 * ladder)` downstream into a 0/0 divide."""
    point = np.array([1.0])
    direction = np.array([1.0])
    ladder = build_step_ladder(point, direction, noise_floor=0.0)
    assert np.all(ladder > 0)
