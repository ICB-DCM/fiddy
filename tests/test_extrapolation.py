import math

import numpy as np
import pytest
from helpers import deterministic_noise

from fiddy.extrapolation import (
    _best_diagonal_estimate,
    extrapolate_central_differences,
    neville_extrapolate,
)


def test_neville_extrapolate_recovers_exact_polynomial():
    # y = 3 + 2*x (linear in x = h**2 here), extrapolated to x=0 should
    # recover the constant term exactly, for any x's.
    xs = np.array([4.0, 1.0, 0.25])
    ys = 3.0 + 2.0 * xs
    diagonal = neville_extrapolate(xs, ys)
    assert np.isclose(diagonal[-1], 3.0)


def test_extrapolate_smooth_central_differences_is_accurate():
    def f(t):
        return math.sin(t)

    x0 = 1.3
    ladder = x0 * 1e-2 / 2.0 ** np.arange(8)
    central_values = np.array(
        [(f(x0 + h) - f(x0 - h)) / (2 * h) for h in ladder]
    )
    result = extrapolate_central_differences(ladder, central_values)

    assert abs(result.value - math.cos(x0)) < 1e-10
    assert result.error_estimate < 1e-8
    assert result.disagreement < 1e-8


def test_extrapolate_requires_at_least_four_rungs():
    with pytest.raises(ValueError, match="at least 4"):
        extrapolate_central_differences(
            np.array([1.0, 0.5, 0.25]), np.array([1.0, 1.0, 1.0])
        )


def test_extrapolate_multi_output_matches_per_column_scalar_calls():
    """A 2D `central_values` array (n_rungs, n_outputs) should give
    exactly the same per-column results as calling
    `extrapolate_central_differences` once per column -- multi-output
    extrapolation is meant to be a batched version of the scalar case, not
    a different algorithm."""

    def f_a(t):
        return math.sin(t)

    def f_b(t):
        return t**3

    x0 = 1.3
    ladder = x0 * 1e-2 / 2.0 ** np.arange(8)
    central_a = np.array(
        [(f_a(x0 + h) - f_a(x0 - h)) / (2 * h) for h in ladder]
    )
    central_b = np.array(
        [(f_b(x0 + h) - f_b(x0 - h)) / (2 * h) for h in ladder]
    )

    scalar_a = extrapolate_central_differences(ladder, central_a)
    scalar_b = extrapolate_central_differences(ladder, central_b)
    multi = extrapolate_central_differences(
        ladder, np.stack([central_a, central_b], axis=1)
    )

    assert multi.value.shape == (2,)
    assert np.isclose(multi.value[0], scalar_a.value)
    assert np.isclose(multi.value[1], scalar_b.value)
    assert np.isclose(multi.error_estimate[0], scalar_a.error_estimate)
    assert np.isclose(multi.error_estimate[1], scalar_b.error_estimate)
    assert multi.diagonal.shape == (len(ladder), 2)
    assert multi.best_index.shape == (2,)


def test_chain_disagreement_flags_noise_dominated_case():
    """A case engineered so noise dominates the whole ladder (comparable
    in scale to the derivative itself): the two independently-
    corroborating chains should disagree noticeably, unlike the clean
    smooth case above -- this is the mechanism that fixes the false-
    convergence failure mode described in `fiddy.extrapolation`'s module
    docstring."""

    noise_amplitude = 1e-8

    def f(x):
        # A function whose derivative at x0 is itself tiny (comparable to
        # the noise amplitude), unlike the well-conditioned smooth case.
        return 1e-8 * math.sin(x) + deterministic_noise(x, noise_amplitude)

    x0 = 0.4
    ladder = 1e-2 / 2.0 ** np.arange(8)
    central_values = np.array(
        [(f(x0 + h) - f(x0 - h)) / (2 * h) for h in ladder]
    )
    result = extrapolate_central_differences(ladder, central_values)

    # The disagreement between chains should dominate (be at least as
    # large as) either chain's own internal error estimate here -- i.e.
    # corroboration is doing real work, not a no-op.
    assert result.disagreement > 0
    assert result.error_estimate >= result.disagreement


def test_early_stopping_avoids_a_coincidentally_agreeing_deep_entry_case_a():
    """Real ladder/central_values captured from a directional derivative
    whose true value is 0 (AMICI SBML semantic test suite case 00193).
    The old global-argmin selection picked the deepest (most noise-
    contaminated) table entry (true error ~5.9e-8) purely because it
    happened to numerically coincide with its neighbor, when a much
    shallower entry (true error ~1.5e-11, ~3.8 million times more
    accurate) was sitting earlier in the very same table. Ridders'-style
    early stopping must pick the shallow, accurate entry instead."""
    ladder = np.array(
        [
            3.570203215001967e-05,
            1.7851016075009836e-05,
            8.925508037504918e-06,
            4.462754018752459e-06,
            2.2313770093762294e-06,
            1.1156885046881147e-06,
            5.578442523440574e-07,
            2.789221261720287e-07,
        ]
    )
    central_values = np.array(
        [
            -4.975506245515696e-11,
            5.099893901653589e-10,
            1.4926518736547087e-10,
            1.3533376987802693e-08,
            -6.5676682440807185e-09,
            -3.045009822255606e-08,
            3.1047158972017943e-08,
            5.174526495336324e-08,
        ]
    )
    result = extrapolate_central_differences(ladder, central_values)
    assert abs(result.value) < 1e-9
    assert result.best_index < 6


def test_early_stopping_avoids_a_coincidentally_agreeing_deep_entry_case_b():
    """Same failure mode as case A above, on a different real direction
    whose true value is an ordinary-magnitude 0.855252628629093 (AMICI
    SBML semantic test suite case 00831) -- confirms the bug isn't
    specific to near-zero true values. The old selection picked the
    deepest entry (true error ~4.9e-8) over a shallower one (true error
    ~7e-10, ~70x more accurate)."""
    ladder = np.array(
        [
            2.7118905466986e-05,
            1.3559452733493e-05,
            6.7797263667465e-06,
            3.38986318337325e-06,
            1.694931591686625e-06,
            8.474657958433125e-07,
            4.2373289792165626e-07,
            2.1186644896082813e-07,
        ]
    )
    central_values = np.array(
        [
            0.8552526293278522,
            0.8552526279338765,
            0.8552526258459835,
            0.8552526297925107,
            0.8552526251745826,
            0.8552526412227017,
            0.8552525991700793,
            0.8552525847595234,
        ]
    )
    expectation = 0.855252628629093
    result = extrapolate_central_differences(ladder, central_values)
    assert abs(result.value - expectation) < 1e-8
    assert result.best_index < 6


def test_early_stopping_does_not_truncate_a_genuinely_converging_diagonal():
    """A synthetic diagonal that keeps improving all the way to the
    deepest order (no noise-driven false agreement anywhere) must still
    pick that deepest, most accurate entry -- early stopping should never
    engage for a well-behaved sequence."""
    # Errors shrink by 10x each order: 1e-1, 1e-2, ..., down to the last.
    true_value = 3.0
    diagonal = true_value + np.array(
        [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    )
    value, error, best_index = _best_diagonal_estimate(diagonal.reshape(-1, 1))
    assert best_index[0] == 7
    assert abs(value[0] - true_value) < 1e-7


def test_best_diagonal_estimate_early_stopping_boundary():
    """Pin the exact `safe` semantics: an error just under `safe *
    best_error` does not itself become the new best, but does *not* stop
    the scan -- a later, genuinely smaller error can still be found and
    used. An error just over that threshold stops the scan outright,
    discarding even a later, coincidentally tiny error."""
    # errors (successive diagonal diffs): [1.0, 1.9, 0.05]. errors[1]=1.9
    # is below safe(2.0)*best_error(1.0)=2.0, so scanning continues past
    # it (without it becoming best) and reaches the genuinely smaller
    # errors[2]=0.05 at index 3.
    diagonal_continues = np.array([0.0, 1.0, 2.9, 2.95]).reshape(-1, 1)
    value, error, best_index = _best_diagonal_estimate(
        diagonal_continues, safe=2.0
    )
    assert best_index[0] == 3

    # errors: [1.0, 2.1, 0.05]. errors[1]=2.1 >= safe(2.0)*1.0=2.0, so the
    # scan stops there -- the later errors[2]=0.05, though numerically
    # smaller than everything seen, is never considered; frozen at index 1.
    diagonal_stops = np.array([0.0, 1.0, 3.1, 3.15]).reshape(-1, 1)
    value, error, best_index = _best_diagonal_estimate(
        diagonal_stops, safe=2.0
    )
    assert best_index[0] == 1
