"""Neville/Richardson extrapolation over a step-size ladder.

Central differences have an error series in even powers of h only
(``D(h) = D_true + a1*h**2 + a2*h**4 + ...``), so :func:`neville_extrapolate`
extrapolates in the variable ``u = h**2`` to ``u -> 0`` via the standard
Neville's-algorithm polynomial-interpolation recursion (``BurdenFaires2015``
in ``doc/references.bib``), following the DERIVEST/numdifftools idea
(``Derrico2006derivest``) of using disagreement between successive
extrapolation orders as a built-in error estimate.

**Why a single best-entry pick is not enough.** Picking the "best" table
entry via a single ``argmin`` of successive-difference sizes can be
confidently wrong: on a real ODE-based likelihood, one parameter whose
true derivative was comparable in magnitude to the solver's own noise
floor produced a table whose error sequence never actually formed a clean
V-shape, yet ``argmin`` still picked an entry that happened to agree with
its neighbor *by chance* -- reporting a value ~11% wrong with high
apparent confidence. :func:`extrapolate_central_differences` guards
against this by splitting the ladder into two independent, interleaved
sub-ladders, extrapolating each independently, and using their
*disagreement* -- not just either chain's own internal successive-
difference estimate -- as part of the reported error. Two independent
estimates from disjoint data agreeing is a much stronger convergence
signal than one estimate's internal self-consistency; disagreeing is a
strong signal that noise (not real higher-order structure) is what the
"convergence" was tracking.

**Corroboration widens the error estimate, but does not by itself fix
*which value* gets reported.** Both the full ladder and each chain
individually still picked their own "best" entry via the same
single-``argmin``-over-the-whole-diagonal search -- and confirmed on
real ODE-model directions whose true value was (near) zero, that search
remained vulnerable to the exact same false-agreement failure mode
described above, even with corroboration already in place: it picked the
deepest, most noise-contaminated table entry because two deep entries
happened to coincide, while several shallower entries in the very same
table were dramatically more accurate (up to ~3.8 million times, in one
confirmed case). :func:`_best_diagonal_estimate` now follows Ridders'
method's early-stopping rule instead (see its own docstring) -- fixing
the value-selection problem directly, rather than only widening the
error bar around a value that was still avoidably wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "neville_extrapolate",
    "ExtrapolationResult",
    "extrapolate_central_differences",
]


def neville_extrapolate(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Neville's algorithm, evaluated at target x=0.

    :param xs: The x-coordinates (e.g. ``h**2`` for each ladder rung).
    :param ys: The corresponding y-values. May carry trailing output-
        component axes (shape ``(n, ...)``), extrapolating every
        component independently from the same `xs`.
    :return: The diagonal ``Q[j, j]`` of the standard Neville tableau,
        i.e. the sequence of increasingly higher-order extrapolated
        estimates using the first ``j + 1`` points.
    """
    n = len(xs)
    q = np.zeros((n, n, *np.shape(ys)[1:]))
    q[:, 0] = ys
    with np.errstate(divide="ignore", invalid="ignore"):
        # `xs[i] - xs[i - j]` can be exactly 0 when the underlying ladder
        # collapsed to an all-zero step (a bounds-clamped direction with
        # zero room to step at all -- see
        # `fiddy.step_size.clamp_step_to_bounds`'s docstring). The
        # resulting NaN correctly propagates to an unresolved
        # ("noise_dominated") direction downstream, not a bug.
        for j in range(1, n):
            for i in range(j, n):
                q[i, j] = (
                    -xs[i - j] * q[i, j - 1] + xs[i] * q[i - 1, j - 1]
                ) / (xs[i] - xs[i - j])
    return np.array([q[j, j] for j in range(n)])


def _best_diagonal_estimate(
    diagonal: np.ndarray,
    safe: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the best trade-off point along a Neville-table diagonal.

    Going to the deepest (highest-order, smallest-step) entry is not
    always best: past some order, further extrapolation just amplifies
    noise (the classical truncation-vs-rounding trade-off -- see
    :mod:`fiddy.step_size`). Neville's algorithm is an *exact* polynomial
    interpolant: at the deepest order, an n-point polynomial fits those n
    points exactly, zero residual, whether the data is genuine smooth
    curvature or pure noise -- so a plain global ``argmin`` of successive-
    difference sizes across the *whole* diagonal can be fooled by two
    deep, heavily-noise-contaminated entries that happen to coincide by
    chance (a high-enough-degree polynomial can wiggle to match almost
    anything). Confirmed on real ODE-model directions whose true value was
    (near) zero: the global-argmin search picked the deepest table entry
    (true error up to ~3.8 million times worse than the best available
    order), because two deep entries coincidentally agreed while several
    shallower, far more accurate entries sat earlier in the same table.

    Instead, this follows Ridders' method's early-stopping rule
    (``Ridders1978`` in ``doc/references.bib``; the standard reference
    implementation is Numerical Recipes' ``dfridr``): scan the diagonal
    from shallow to deep, tracking the best (smallest) successive-
    difference error seen so far, and stop considering any further,
    deeper entries as soon as a new one is worse than that best-so-far by
    more than a factor of `safe` -- rather than searching the entire
    diagonal for a global minimum that a later, coincidentally-small
    difference could win by chance. This needs no restructuring of
    fiddy's batch-then-analyze evaluation (unlike Ridders' own adaptive,
    incrementally-built table): it is a pure post-hoc analysis of the
    already-computed diagonal.

    Used for the full ladder and for each corroborating chain
    individually -- the corroboration in
    :func:`extrapolate_central_differences` comes from comparing what
    this picks *independently* on disjoint data, not from trusting one
    call to this function alone (see the module docstring for why that
    matters).

    Vectorized over a trailing output-component axis: `diagonal` has
    shape ``(n_rungs, n_outputs)`` -- each output component picks its own
    best trade-off order independently, since different bundled outputs
    (e.g. a state trajectory vs. a scalar likelihood) may converge at
    different orders.

    :param diagonal: The Neville-table diagonal, shape
        ``(n_rungs, n_outputs)``.
    :param safe: Safety factor on the early-stopping rule -- matches
        Numerical Recipes' own default; verified insensitive to the exact
        value (tested 1.2-4.0 against real failing cases with identical
        results), since the transition from well-behaved to noise-
        dominated is a sharp, orders-of-magnitude jump, not a marginal
        one.
    :return: A tuple ``(value, error, index)``, each shape
        ``(n_outputs,)``. `index` is into the *diagonal*, which for the
        full ladder also indexes the original ladder/step-size array (the
        discontinuity check, :func:`fiddy.discontinuity.check_discontinuity`,
        reuses it to know which step size the chosen value came from, per
        output component).
    """
    n, n_outputs = diagonal.shape
    if n == 1:
        return (
            diagonal[0].copy(),
            np.full(n_outputs, np.inf),
            np.zeros(n_outputs, dtype=int),
        )
    errors = np.abs(np.diff(diagonal, axis=0))
    best_index = np.ones(n_outputs, dtype=int)
    best_error = errors[0].copy()
    stopped = np.zeros(n_outputs, dtype=bool)
    for i in range(2, n):
        errt = errors[i - 1]
        worse = (~stopped) & (errt >= safe * best_error)
        improve = (~stopped) & (~worse) & (errt < best_error)
        best_error = np.where(improve, errt, best_error)
        best_index = np.where(improve, i, best_index)
        stopped = stopped | worse
    value = np.take_along_axis(diagonal, best_index[None, :], axis=0)[0]
    return value, best_error, best_index


@dataclass
class ExtrapolationResult:
    value: float | np.ndarray
    """The extrapolated derivative estimate (from the full ladder). A
    plain float for a scalar-output function; shape ``(n_outputs,)`` when
    `central_values` was given as a 2D, multi-output array."""
    error_estimate: float | np.ndarray
    """``max`` of: the full-ladder's own best-trade-off error, each
    independent chain's own best-trade-off error, and the disagreement
    between the two chains' best estimates -- see module docstring for why
    the cross-chain disagreement term is the important fix here."""
    diagonal: np.ndarray
    """The full-ladder Neville-table diagonal (increasingly higher-order
    estimates), for diagnostics/plotting
    (:func:`fiddy.plotting.plot_extrapolation`). Shape ``(n_rungs,)``, or
    ``(n_rungs, n_outputs)`` for multi-output."""
    best_index: int | np.ndarray
    """Index into the original ladder/diagonal that `value` came from --
    reused by the discontinuity check to pick a vetted step size
    (:func:`fiddy.discontinuity.check_discontinuity`). One index per
    output component for multi-output, since different outputs may
    converge at different orders."""
    chain_a_value: float | np.ndarray
    chain_b_value: float | np.ndarray
    chain_a_error: float | np.ndarray
    chain_b_error: float | np.ndarray
    disagreement: float | np.ndarray
    """``abs(chain_a_value - chain_b_value)``."""


def extrapolate_central_differences(
    ladder: np.ndarray, central_values: np.ndarray
) -> ExtrapolationResult:
    """Extrapolate central-difference estimates across a step-size ladder.

    Requires at least 4 rungs, so each of the two corroborating chains
    (the even-indexed and odd-indexed rungs) has at least 2 points.

    :param ladder: Step sizes, decreasing, e.g. from
        :func:`fiddy.step_size.build_step_ladder`.
    :param central_values: The central-difference estimate at each step
        size in `ladder` -- shape ``(n_rungs,)`` for a scalar-output
        function, or ``(n_rungs, n_outputs)`` to extrapolate every output
        component's derivative at once from the same ladder (e.g. several
        bundled simulation outputs sharing one batch of perturbed-point
        evaluations). All :class:`ExtrapolationResult` fields that are
        per-output become shape ``(n_outputs,)`` arrays in that case
        instead of plain floats.
    :return: The extrapolation result.
    :raises ValueError: If `ladder` has fewer than 4 rungs.
    """
    ladder = np.asarray(ladder, dtype=float)
    central_values = np.asarray(central_values, dtype=float)
    if len(ladder) < 4:
        raise ValueError(
            "extrapolate_central_differences needs at least 4 ladder rungs "
            "(so each of the two independently-corroborating chains has "
            f"at least 2 points); got {len(ladder)}."
        )
    was_1d = central_values.ndim == 1
    values_2d = central_values.reshape(len(ladder), -1)

    diagonal = neville_extrapolate(ladder**2, values_2d)
    value, full_error, best_index = _best_diagonal_estimate(diagonal)

    chain_a_ladder, chain_a_values = ladder[0::2], values_2d[0::2]
    chain_b_ladder, chain_b_values = ladder[1::2], values_2d[1::2]

    diagonal_a = neville_extrapolate(chain_a_ladder**2, chain_a_values)
    diagonal_b = neville_extrapolate(chain_b_ladder**2, chain_b_values)

    chain_a_value, chain_a_error, _ = _best_diagonal_estimate(diagonal_a)
    chain_b_value, chain_b_error, _ = _best_diagonal_estimate(diagonal_b)
    disagreement = np.abs(chain_a_value - chain_b_value)

    error_estimate = np.maximum.reduce(
        [full_error, chain_a_error, chain_b_error, disagreement]
    )

    result = ExtrapolationResult(
        value=value,
        error_estimate=error_estimate,
        diagonal=diagonal,
        best_index=best_index,
        chain_a_value=chain_a_value,
        chain_b_value=chain_b_value,
        chain_a_error=chain_a_error,
        chain_b_error=chain_b_error,
        disagreement=disagreement,
    )
    if was_1d:
        return _squeeze_extrapolation_result(result)
    return result


def _squeeze_extrapolation_result(
    result: ExtrapolationResult,
) -> ExtrapolationResult:
    """Collapse a single-output (``n_outputs == 1``) result's array fields
    back to plain floats/1D arrays, so scalar-output callers get plain
    scalars rather than length-1 arrays.

    :param result: A result computed from 2D (multi-output-shaped) input.
    :return: The same result, with every per-output field collapsed to a
        plain scalar/1D array.
    """
    return ExtrapolationResult(
        value=float(result.value[0]),
        error_estimate=float(result.error_estimate[0]),
        diagonal=result.diagonal[:, 0],
        best_index=int(result.best_index[0]),
        chain_a_value=float(result.chain_a_value[0]),
        chain_b_value=float(result.chain_b_value[0]),
        chain_a_error=float(result.chain_a_error[0]),
        chain_b_error=float(result.chain_b_error[0]),
        disagreement=float(result.disagreement[0]),
    )
