import numpy as np
from helpers import deterministic_noise

from fiddy.discontinuity import (
    check_cross_regime_disagreement,
    check_discontinuity,
)


def _gap_curve(f, x0, ladder):
    f0 = f(x0)
    return np.array([(f(x0 + h) - 2 * f0 + f(x0 - h)) / h for h in ladder])


def test_smooth_function_is_not_flagged():
    def f(x):
        return np.sin(x)

    x0 = 0.7
    ladder = 1e-2 / 2.0 ** np.arange(8)
    gap_values = _gap_curve(f, x0, ladder)

    result = check_discontinuity(
        ladder, gap_values, best_index=3, noise_sigma=1e-12
    )

    assert not result.suspected


def test_kink_is_flagged():
    """Central differences of `abs` at its corner are exactly 0 for every
    step size (silently wrong); the gap check must catch this instead."""

    def f(x):
        return abs(x)

    x0 = 0.0
    ladder = 1e-2 / 2.0 ** np.arange(8)
    gap_values = _gap_curve(f, x0, ladder)

    result = check_discontinuity(
        ladder, gap_values, best_index=3, noise_sigma=1e-12
    )

    assert result.suspected


def test_multi_output_flags_only_the_kinked_column():
    """A 2D `gap_values` array (n_rungs, n_outputs) should flag each
    output column independently -- a genuine kink in one output must not
    be masked, or falsely triggered, by another smooth output sharing the
    same ladder."""

    def smooth(x):
        return np.sin(x)

    def kinked(x):
        return abs(x)

    x0 = 0.0
    ladder = 1e-2 / 2.0 ** np.arange(8)
    gap_smooth = _gap_curve(smooth, x0 + 0.7, ladder)
    gap_kinked = _gap_curve(kinked, x0, ladder)
    gap_values = np.stack([gap_smooth, gap_kinked], axis=1)

    result = check_discontinuity(
        ladder, gap_values, best_index=np.array([3, 3]), noise_sigma=1e-12
    )

    assert result.suspected.shape == (2,)
    assert not result.suspected[0]
    assert result.suspected[1]


def test_nondet_tol_prevents_false_positive_from_known_nondeterminism():
    """A function whose value legitimately jitters by up to `nondet_tol`
    between calls at the same point (e.g. a simulator run with vs. without
    sensitivities) must not be flagged as a kink just because that jitter
    looks like noise/inconsistency in the gap curve."""

    amplitude = 1e-6

    def f(x):
        return np.sin(x) + deterministic_noise(x, amplitude)

    x0 = 0.7
    ladder = 1e-2 / 2.0 ** np.arange(8)
    gap_values = _gap_curve(f, x0, ladder)

    without_hint = check_discontinuity(
        ladder, gap_values, best_index=3, noise_sigma=1e-12
    )
    with_hint = check_discontinuity(
        ladder,
        gap_values,
        best_index=3,
        noise_sigma=1e-12,
        nondet_tol=amplitude,
    )

    assert not with_hint.suspected
    assert with_hint.noise_budget > without_hint.noise_budget


def test_agreeing_regimes_are_not_flagged():
    """Two independently-obtained values that agree well within the far
    ladder's own reported error estimate must not be flagged."""
    result = check_cross_regime_disagreement(
        main_value=1.2018,
        far_value=1.2017,
        far_error=0.001,
        noise_sigma=1e-9,
    )
    assert not result.suspected


def test_disagreeing_regimes_are_flagged():
    """Reproduces the real, confirmed "ladder too coarse near an event"
    bug (case 00026 of the AMICI SBML semantic test suite): the main
    ladder confidently converges to a value from the wrong branch of a
    hidden discontinuity, while a far, independently-anchored ladder
    (entirely on the correct branch) converges to the true value -- a
    disagreement far too large for the far ladder's own small error
    estimate to explain."""
    result = check_cross_regime_disagreement(
        main_value=1.2017546904865188,
        far_value=0.230595,
        far_error=1e-6,
        noise_sigma=1e-9,
    )
    assert result.suspected


def test_main_error_alone_does_not_widen_the_budget():
    """Regression test: `main_error` must NOT be part of the budget, even
    though it looks like a natural candidate (mirroring how
    `extrapolate_central_differences` folds its own two-chain
    disagreement against `max(chain_a_error, chain_b_error, ...)`).
    Confirmed on a real, still-failing AMICI/SBML case that this is
    actively wrong here: the main ladder's own error estimate, when it is
    itself sitting on the wrong side of a hidden discontinuity, is
    *partially informative about the same underlying problem, but
    underestimates it* -- observed at roughly 1/9-1/10 of the true
    disagreement there, i.e. large enough that including it (even with
    this function's own smaller `safety_factor=3.0`) would have masked
    that real, confirmed bug."""
    main_error_mirroring_the_real_case = 0.4595  # ~ disagreement / 9.4
    result = check_cross_regime_disagreement(
        main_value=-1.4898387087224596,
        far_value=-5.804725386626403,
        far_error=9.267026063852768e-06,
        noise_sigma=0.010826586716221222,
    )
    assert result.suspected
    assert result.noise_budget < main_error_mirroring_the_real_case


def test_noise_dominated_far_ladder_widens_the_budget_instead_of_misfiring():
    """A far ladder that is itself noise-dominated (expected whenever a
    function's real noise floor sits far above machine epsilon, the far
    ladder's anchor) must not be mistaken for a genuine cross-regime
    disagreement just because its own reported value differs a lot from
    the main ladder's -- its own large `far_error` should widen the
    budget enough to absorb that."""
    result = check_cross_regime_disagreement(
        main_value=2.08,
        far_value=2.68,
        far_error=54.5,
        noise_sigma=1e-9,
    )
    assert not result.suspected


def test_multi_output_flags_only_the_disagreeing_column():
    """Vectorized over an output-component axis, mirroring
    `test_multi_output_flags_only_the_kinked_column` for
    `check_discontinuity`."""
    result = check_cross_regime_disagreement(
        main_value=np.array([1.2018, 5.0]),
        far_value=np.array([0.230595, 5.0001]),
        far_error=np.array([1e-6, 0.001]),
        noise_sigma=np.array([1e-9, 1e-9]),
    )
    assert result.suspected.shape == (2,)
    assert result.suspected[0]
    assert not result.suspected[1]
