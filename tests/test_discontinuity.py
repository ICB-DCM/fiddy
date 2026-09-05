import numpy as np
from helpers import deterministic_noise

from fiddy.discontinuity import check_discontinuity


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
