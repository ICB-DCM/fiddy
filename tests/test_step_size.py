import numpy as np

from fiddy.step_size import build_step_ladder, clamp_step_to_bounds


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


def test_clamp_step_to_bounds_is_a_noop_without_bounds():
    point = np.array([1.0])
    direction = np.array([1.0])
    assert clamp_step_to_bounds(point, direction, h=10.0, bounds=None) == 10.0


def test_clamp_step_to_bounds_shrinks_for_a_tighter_bound():
    point = np.array([1.0])
    direction = np.array([1.0])
    bounds = (np.array([0.0]), np.array([1.2]))

    h = clamp_step_to_bounds(point, direction, h=10.0, bounds=bounds)

    # Forward: point + h <= 1.2 -> h <= 0.2. Backward: point - h >= 0.0 ->
    # h <= 1.0. The forward (tighter) constraint wins.
    assert np.isclose(h, 0.2)


def test_clamp_step_to_bounds_is_a_noop_for_a_looser_bound():
    point = np.array([1.0])
    direction = np.array([1.0])
    bounds = (np.array([-100.0]), np.array([100.0]))

    h = clamp_step_to_bounds(point, direction, h=10.0, bounds=bounds)

    assert h == 10.0


def test_clamp_step_to_bounds_uses_the_tightest_component():
    point = np.array([1.0, 5.0])
    direction = np.array([1.0, 1.0])
    # Component 0 allows up to 5.0 either way; component 1 is nearly
    # against its own upper bound, allowing only 0.1.
    bounds = (np.array([-10.0, -10.0]), np.array([6.0, 5.1]))

    h = clamp_step_to_bounds(point, direction, h=10.0, bounds=bounds)

    assert np.isclose(h, 0.1)


def test_clamp_step_to_bounds_ignores_zero_direction_components():
    point = np.array([1.0, 5.0])
    direction = np.array([1.0, 0.0])
    # Component 1 is not perturbed at all (direction is 0 there), so its
    # own tight bound must not constrain the step.
    bounds = (np.array([-10.0, 5.0]), np.array([6.0, 5.0]))

    h = clamp_step_to_bounds(point, direction, h=10.0, bounds=bounds)

    # Only component 0 constrains: forward <= 5.0, backward <= 11.0.
    assert np.isclose(h, 5.0)


def test_clamp_step_to_bounds_floors_at_zero_when_already_at_the_bound():
    point = np.array([1.0])
    direction = np.array([1.0])
    bounds = (np.array([0.0]), np.array([1.0]))

    h = clamp_step_to_bounds(point, direction, h=10.0, bounds=bounds)

    assert h == 0.0


def test_eps_anchored_far_ladder_is_far_smaller_than_a_typical_main_ladder():
    """Documents the intended use of `build_step_ladder` for
    `fiddy.estimate`'s "far" ladder (see
    `fiddy.discontinuity.check_cross_regime_disagreement`): calling it a
    second time with `noise_floor=numpy.finfo(float).eps` -- the
    theoretical minimum possible noise floor (pure rounding error only) --
    requires no signature or behavior change to this function itself, and
    reliably produces a ladder far below a typical noise-floor-derived
    main ladder's own finest rung."""
    point = np.array([1.0])
    direction = np.array([1.0])
    main_ladder = build_step_ladder(point, direction, noise_floor=1e-9)
    far_ladder = build_step_ladder(
        point,
        direction,
        noise_floor=np.finfo(float).eps,
        n_rungs=4,
        step_ratio=10.0,
    )

    assert far_ladder.shape == (4,)
    assert far_ladder[0] < main_ladder[-1]


def test_build_step_ladder_respects_bounds():
    """Regression test for the unscaled-parameter-domain failure mode
    (e.g. a PEtab model's log10-scale parameter requiring a strictly
    positive linear value): without `bounds`, a large enough noise floor
    can size a step that pushes the evaluated point outside a known
    valid domain; with `bounds` supplied, every rung of the ladder must
    stay within it."""
    point = np.array([1.0])
    direction = np.array([1.0])
    bounds = (np.array([0.5]), np.array([1.5]))

    ladder = build_step_ladder(
        point, direction, noise_floor=1e6, bounds=bounds
    )

    assert np.all(point[0] + ladder <= 1.5)
    assert np.all(point[0] - ladder >= 0.5)
