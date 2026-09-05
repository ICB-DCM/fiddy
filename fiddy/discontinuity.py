"""Kink/discontinuity detection via a forward-backward-central cross-check.

Central differences alone cannot see a kink: central differences of
``abs(x)`` at its corner are exactly 0 for *every* step size, silently
masking the fact that the one-sided derivatives are +-1 -- and the
noise-floor/extrapolation machinery in :mod:`fiddy.noise`/
:mod:`fiddy.extrapolation` has no way to catch this either, since a
symmetric kink like that produces perfectly self-consistent (but wrong)
central differences at every scale.

Instead, this looks at the forward-minus-backward "gap"
``[f(x+h) - 2f(x) + f(x-h)] / h`` (proportional to the central second
difference) at two step sizes: for a smooth function this gap is ``O(h)``
(it estimates ``h * f''(x)``), so its value at a smaller step should match
its value at a larger step scaled by their ratio, up to noise; for a
genuine kink the gap stays ``O(1)`` regardless of h, so that scaling
badly overpredicts how much it should have shrunk.

**Comparing the wrong pair of step sizes causes false positives on real
curvature.** An earlier version of this check compared the ladder's
*coarsest* rung against the extrapolation-vetted rung, often several
rungs deeper (e.g. index 5-7 of an 8-rung ladder). ``gap(h) = h*f''(x) +
O(h**3)`` is only a first-order approximation -- comparing across a wide
h-ratio (up to ``step_ratio**best_index``, e.g. 128x for
``step_ratio=2``, ``best_index=7``) let the ignored ``O(h**3)`` term at
the *coarse* end dominate the predicted value, producing a residual that
looked like a kink but was really just curvature the linear-only
prediction model couldn't see. This was confirmed on a real ODE-based
likelihood (all 19 parameters of one model had a correct value, relative
error ~1e-8..1e-12, yet were flagged `discontinuity_suspected`). Fixed by
comparing `best_index` against its immediate coarser neighbor
(``best_index - 1``) instead of the ladder's widest possible span: the
same `step_ratio` (typically 2x) gap is just as revealing for a genuine
kink (its ``gap(h)`` does not shrink at *any* scale, however small the
comparison ratio), while keeping the linear approximation valid enough to
stop misfiring on real curvature.

**A second, independent false-positive source: an under-calibrated noise
budget.** Even after the fix above, some parameters' gap curves matched
the smooth O(h) prediction almost perfectly (well under 1% relative
deviation -- nowhere near a genuine kink's 50-100%+) yet still exceeded a
noise-floor-only budget. Root cause:
:func:`fiddy.noise.estimate_model_noise_floor` shares one noise estimate
from a single generic probe direction across every parameter direction
(a deliberate few-evaluations trade-off -- see its docstring), which can
under-estimate the real per-direction noise for some models. See
`curvature_rtol` below for the fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DiscontinuityCheck", "check_discontinuity"]


@dataclass
class DiscontinuityCheck:
    suspected: bool | np.ndarray
    ladder: np.ndarray
    """The step-size ladder the gap curve was evaluated on, for
    diagnostics/plotting (`fiddy.plotting.plot_discontinuity`)."""
    gap_values: np.ndarray
    """The gap ``[f(x+h) - 2f(x) + f(x-h)] / h`` at each rung of `ladder`."""
    best_index: int | np.ndarray
    """Index into `ladder`/`gap_values` used as the "smaller" step (h_b) in
    the cross-check -- a step size already vetted by extrapolation
    (`fiddy.extrapolation.ExtrapolationResult.best_index`), not the
    ladder's noisiest extreme."""
    reference_index: int | np.ndarray
    """Index into `ladder`/`gap_values` used as the "larger" step (h_a) --
    `best_index`'s immediate coarser neighbor (``max(best_index - 1,
    0)``), not the ladder's widest rung (see module docstring for why a
    wide comparison span produced false positives on real models)."""
    predicted_gap_b: float | np.ndarray
    """What `gap_values[best_index]` should be if the function were smooth,
    extrapolated from `gap_values[reference_index]`."""
    residual: float | np.ndarray
    """``gap_values[best_index] - predicted_gap_b``."""
    noise_budget: float | np.ndarray


def check_discontinuity(
    ladder: np.ndarray,
    gap_values: np.ndarray,
    best_index: int | np.ndarray,
    noise_sigma: float | np.ndarray,
    nondet_tol: float = 0.0,
    safety_factor: float = 10.0,
    curvature_rtol: float = 0.05,
) -> DiscontinuityCheck:
    """Cross-check a step-size ladder's gap curve for a genuine kink.

    :param ladder: Step sizes, decreasing, e.g. from
        :func:`fiddy.step_size.build_step_ladder`.
    :param gap_values: ``[f(x+h) - 2f(x) + f(x-h)] / h`` at each step size
        in `ladder` -- shape ``(n_rungs,)`` for a scalar-output function,
        or ``(n_rungs, n_outputs)`` to check every output component at
        once from the same ladder.
    :param best_index: Index of the "smaller" step size to compare
        against its coarser neighbor -- typically
        :attr:`fiddy.extrapolation.ExtrapolationResult.best_index`, a step
        already vetted by extrapolation rather than the ladder's smallest
        (noisiest) rung. Shape ``(n_outputs,)`` if `gap_values` is 2D --
        one vetted step per output component.
    :param noise_sigma: The function's empirically estimated noise floor,
        e.g. from :func:`fiddy.noise.estimate_model_noise_floor`. Shape
        ``(n_outputs,)`` if `gap_values` is 2D.
    :param nondet_tol: An expected magnitude of legitimate, non-noise
        value nondeterminism at the *same* point -- e.g. a simulator
        giving slightly different values when run with vs. without
        sensitivities enabled. Folded into the noise budget so this
        doesn't get mistaken for a discontinuity, the same way
        `torch.autograd.gradcheck`'s `nondet_tol` (``TorchGradcheck`` in
        ``doc/references.bib``) tolerates a non-deterministic analytic
        gradient.
    :param safety_factor: Multiplier on the noise-driven (absolute)
        budget below which a residual is considered explainable by noise
        rather than a kink.
    :param curvature_rtol: An additional budget of
        ``curvature_rtol * abs(predicted_gap_b)`` (i.e. relative to the
        gap's own magnitude), on top of the noise-driven absolute budget
        -- see the module docstring for the false positive this guards
        against. A genuine kink deviates by 50-100%+ (e.g. exactly 100%
        for ``abs(x)`` at its corner), so a small relative allowance here
        does not meaningfully weaken kink detection.
    :return: The discontinuity check result.
    """
    ladder = np.asarray(ladder, dtype=float)
    gap_values = np.asarray(gap_values, dtype=float)
    was_1d = gap_values.ndim == 1
    values_2d = gap_values.reshape(len(ladder), -1)
    n_outputs = values_2d.shape[1]

    best_index_arr = np.broadcast_to(
        np.asarray(best_index, dtype=int), (n_outputs,)
    )
    noise_sigma_arr = np.broadcast_to(
        np.asarray(noise_sigma, dtype=float), (n_outputs,)
    )

    # Compare against the immediate coarser neighbor, not the ladder's
    # widest rung -- see module docstring for why the wider comparison
    # produced false positives from ordinary curvature on real models.
    reference_index_arr = np.maximum(best_index_arr - 1, 0)
    h_a = ladder[reference_index_arr]
    h_b = ladder[best_index_arr]
    gap_a = np.take_along_axis(
        values_2d, reference_index_arr[None, :], axis=0
    )[0]
    gap_b = np.take_along_axis(values_2d, best_index_arr[None, :], axis=0)[0]
    predicted_gap_b = gap_a * (h_b / h_a)
    residual = gap_b - predicted_gap_b

    effective_noise = np.maximum(noise_sigma_arr, nondet_tol)
    noise_budget = safety_factor * np.maximum(
        effective_noise / h_b, np.finfo(float).eps
    ) + curvature_rtol * np.abs(predicted_gap_b)

    suspected = (np.abs(residual) > noise_budget) & (
        np.abs(gap_b) > noise_budget
    )

    result = DiscontinuityCheck(
        suspected=suspected,
        ladder=ladder,
        gap_values=gap_values,
        best_index=best_index_arr,
        reference_index=reference_index_arr,
        predicted_gap_b=predicted_gap_b,
        residual=residual,
        noise_budget=noise_budget,
    )
    if was_1d:
        return _squeeze_discontinuity_check(result)
    return result


def _squeeze_discontinuity_check(
    result: DiscontinuityCheck,
) -> DiscontinuityCheck:
    """Collapse a single-output (``n_outputs == 1``) result's array fields
    back to plain scalars, so scalar-output callers get plain scalars
    rather than length-1 arrays.

    :param result: A result computed from 2D (multi-output-shaped) input.
    :return: The same result, with every per-output field collapsed to a
        plain scalar.
    """
    return DiscontinuityCheck(
        suspected=bool(result.suspected[0]),
        ladder=result.ladder,
        gap_values=result.gap_values,
        best_index=int(result.best_index[0]),
        reference_index=int(result.reference_index[0]),
        predicted_gap_b=float(result.predicted_gap_b[0]),
        residual=float(result.residual[0]),
        noise_budget=float(result.noise_budget[0]),
    )
