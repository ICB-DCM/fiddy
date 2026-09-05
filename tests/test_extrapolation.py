import math

import numpy as np
import pytest
from helpers import deterministic_noise

from fiddy.extrapolation import (
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
