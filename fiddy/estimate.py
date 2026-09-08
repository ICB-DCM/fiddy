"""Directional-derivative estimation: noise floor + ladder + extrapolation
+ discontinuity diagnostics, composed.

This composes :mod:`fiddy.noise`, :mod:`fiddy.step_size`,
:mod:`fiddy.extrapolation`, :mod:`fiddy.discontinuity`, and
:mod:`fiddy.executor` into one internal engine that, given only a
function/point/direction(s), needs no user-supplied step sizes and
reports a value, a trustworthy error estimate, and a classification --
this is what :mod:`fiddy.check` wraps in a minimal public API
(:func:`fiddy.check_gradient`).

Three entry points:

- :func:`estimate_directional_derivative` -- one direction, one (the
  first) output component.
- :func:`estimate_gradient` -- several directions (default: the standard
  basis, i.e. the full gradient) at once, sharing one noise-floor probe
  and dispatching *every* direction's ladder evaluations as a single
  combined batch through `executor` -- checking N parameters costs one
  batch dispatch, not N. Like :func:`estimate_directional_derivative`,
  only the first output component is estimated.
- :func:`estimate_jacobian` -- multi-output support: every output
  component of `function` (e.g. a model's bundled state/observable/
  likelihood sensitivities, not just its scalar objective), along several
  directions, from the *same* shared batch of evaluations used for a
  single output -- checking N outputs costs no more function evaluations
  than checking 1, since that is the entire reason a multi-output
  function bundles several outputs into one evaluation in the first
  place. Checking every output component together is a central use case
  this engine is meant to support well, not an afterthought.

A raw, unwrapped function (a plain callable, not a :class:`fiddy.Function`)
is automatically wrapped in :class:`fiddy.Function` by every entry point
above, so a dict-returning function does not need to be pre-wrapped by
the caller to avoid crashing -- it is simply treated as a bundled
multi-output function like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np

from .constants import Type
from .discontinuity import (
    CrossRegimeCheck,
    DiscontinuityCheck,
    check_cross_regime_disagreement,
    check_discontinuity,
)
from .executor import Executor, SequentialExecutor
from .extrapolation import ExtrapolationResult, extrapolate_central_differences
from .function import Function
from .noise import (
    NoiseFloor,
    _analyze_noise_table,
    _noise_floor_probe_points,
    estimate_model_noise_floor,
    noise_floor_is_confident,
)
from .output import OutputSchema
from .step_size import build_step_ladder

__all__ = [
    "DerivativeEstimate",
    "EstimateKwargs",
    "JacobianEstimate",
    "estimate_directional_derivative",
    "estimate_gradient",
    "estimate_jacobian",
]


class EstimateKwargs(TypedDict, total=False):
    """Options shared by :func:`estimate_gradient`/:func:`estimate_jacobian`
    -- see their docstrings (and :func:`estimate_directional_derivative`
    for the most detailed per-option rationale). Forwarded as-is by
    :func:`fiddy.check.check_gradient`/:func:`fiddy.check.check_jacobian`
    via ``**estimate_kwargs`` rather than being re-declared on those call
    sites too.
    """

    noise_floor: float | None
    nondet_tol: float
    n_rungs: int
    step_ratio: float
    n_rungs_far: int
    step_ratio_far: float
    bounds: Type.BOUNDS | None
    noise_floor_strategy: str
    executor: Executor | None


def _ensure_function(function: Type.FUNCTION) -> Function:
    """Wrap a raw callable in :class:`fiddy.Function` if it isn't already
    one.

    Every entry point in this module needs `function`'s output flattened
    (see :mod:`fiddy.output`) to support a dict-returning (multi-output)
    raw function; wrapping here means a caller need not pre-wrap the
    function themselves (a raw dict-returning function would otherwise
    crash), with no behavior change for already-working plain-array
    functions.

    :param function: The function to ensure is wrapped.
    :return: `function` itself if already a :class:`fiddy.Function`, or a
        new wrapper around it.
    """
    if isinstance(function, Function):
        return function
    return Function(function)


def _validate_point_in_bounds(
    point: np.ndarray, bounds: Type.BOUNDS | None
) -> None:
    """Raise a clear error if `point` itself already violates `bounds`,
    rather than silently letting bounds-aware clamping (see
    :func:`fiddy.step_size.clamp_step_to_bounds`) collapse every probe/
    ladder step down to zero and produce a confusing, uniformly
    "noise_dominated" result with no indication of the real cause.
    Mirrors `scipy.optimize.approx_derivative`'s own behavior for an
    infeasible `x0`. Found via a real regression: a caller's own jittered
    starting point pushed one parameter past its declared bound, silently
    crushing every direction's noise-floor estimate instead of raising.

    :param point: The point to validate.
    :param bounds: `(lower, upper)`, or `None` to skip validation.
    :raises ValueError: If any component of `point` is outside `bounds`.
    """
    if bounds is None:
        return
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    violated = (point < lower) | (point > upper)
    if np.any(violated):
        bad = np.where(violated)[0]
        raise ValueError(
            "`point` violates the supplied `bounds` at component index/"
            f"indices {list(bad)}: point={point[bad]}, "
            f"lower={lower[bad]}, upper={upper[bad]}. A finite-difference "
            "step cannot be safely clamped to a domain the starting "
            "point itself is already outside of."
        )


def _estimate_noise_floors_per_direction(
    function: Function,
    point: np.ndarray,
    directions: list[np.ndarray],
    executor: Executor,
    bounds: Type.BOUNDS | None,
) -> list[NoiseFloor]:
    """Estimate one independent, bounds-clamped noise floor per direction,
    from one *combined* batch dispatch across every direction's own probe
    points -- per fiddy's batch-then-analyze design, this is one
    `executor` call for all `len(directions)` directions' probes, not one
    per direction (see :func:`fiddy.noise._noise_floor_probe_points`).

    :param function: The blackbox function (already wrapped).
    :param point: The point to probe around.
    :param directions: The directions to probe along, one noise floor
        each.
    :param executor: How to dispatch the combined probe batch.
    :param bounds: Optional per-parameter valid domain.
    :return: One :class:`NoiseFloor` per direction, in the same order.
    """
    n_points = 15
    all_probe_points: list[np.ndarray] = []
    for d in directions:
        all_probe_points.extend(
            _noise_floor_probe_points(point, d, None, n_points, bounds)
        )
    values = (
        np.array([np.asarray(v) for v in executor(function, all_probe_points)])
        .reshape(len(all_probe_points), -1)
        .astype(float)
    )

    noises = []
    offset = 0
    for _ in directions:
        chunk = values[offset : offset + n_points]
        offset += n_points
        noises.append(_analyze_noise_table(chunk, 3.0))
    return noises


def _resolve_noise_floors(
    function: Function,
    point: np.ndarray,
    directions: list[np.ndarray],
    noise_floor_strategy: str,
    executor: Executor,
    bounds: Type.BOUNDS | None,
) -> list[NoiseFloor]:
    """Resolve one :class:`NoiseFloor` per direction, per
    `noise_floor_strategy` -- see :func:`estimate_gradient`'s
    `noise_floor_strategy` parameter for the full rationale.

    :param function: The blackbox function (already wrapped).
    :param point: The point to probe around.
    :param directions: The directions each result is needed for.
    :param noise_floor_strategy: `"shared"`, `"per_direction"`, or
        `"auto"`.
    :param executor: How to dispatch probe evaluations.
    :param bounds: Optional per-parameter valid domain.
    :return: One :class:`NoiseFloor` per direction, in the same order
        (the same shared object repeated, for `"shared"`/un-escalated
        `"auto"`).
    :raises ValueError: If `noise_floor_strategy` isn't one of the three
        values above.
    """
    if noise_floor_strategy not in ("shared", "per_direction", "auto"):
        raise ValueError(
            "`noise_floor_strategy` must be one of 'shared', "
            f"'per_direction', 'auto'; got {noise_floor_strategy!r}."
        )

    if noise_floor_strategy == "per_direction":
        return _estimate_noise_floors_per_direction(
            function, point, directions, executor, bounds
        )

    shared = estimate_model_noise_floor(
        function, point, executor=executor, bounds=bounds
    )
    if noise_floor_strategy == "shared" or noise_floor_is_confident(shared):
        return [shared] * len(directions)
    # "auto", and the shared probe came back unconfident/degenerate --
    # escalate to an independent probe per direction (see
    # `fiddy.noise.noise_floor_is_confident`'s own docstring for why this
    # can happen and why it's not safe to use as-is).
    return _estimate_noise_floors_per_direction(
        function, point, directions, executor, bounds
    )


@dataclass
class DerivativeEstimate:
    """The result of estimating one directional derivative (one output
    component, one direction)."""

    value: float
    """The extrapolated derivative estimate."""
    error_estimate: float
    """The estimate's own reported uncertainty."""
    status: str
    """One of "converged", "noise_dominated", "discontinuity_suspected"."""
    noise: NoiseFloor
    """The noise floor the estimate was computed with."""
    extrapolation: ExtrapolationResult
    """The full extrapolation result `value`/`error_estimate` came from."""
    discontinuity: DiscontinuityCheck
    """The kink/discontinuity cross-check result (adjacent-rung, within
    the main ladder)."""
    far_extrapolation: ExtrapolationResult | None = None
    """The independently-anchored "far" ladder's own extrapolation result
    (see :func:`fiddy.step_size.build_step_ladder`'s `noise_floor=
    numpy.finfo(float).eps` use in `_estimate_from_ladder`) -- `None` only
    if no far ladder was supplied (internal/testing use of
    `_estimate_from_ladder`; every public entry point always supplies
    one)."""
    far_discontinuity: DiscontinuityCheck | None = None
    """The far ladder's own adjacent-rung discontinuity check -- catches
    an ordinary in-range kink on the far side too, independently of the
    main ladder's own check."""
    cross_regime: CrossRegimeCheck | None = None
    """Whether the main and far ladders' independently-obtained values
    agree -- see :func:`fiddy.discontinuity.check_cross_regime_disagreement`.
    Catches the harder failure mode neither `discontinuity` nor
    `far_discontinuity` can see on their own: the *entire* main ladder
    sitting on the wrong side of a hidden discontinuity closer to the
    evaluation point than its finest rung, so no adjacent-rung comparison
    within either ladder ever disagrees."""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Additional diagnostics (``ladder``, ``central_values``,
    ``gap_values``, ``tol``, ``relative_error``, ``far_ladder``,
    ``far_central_values``), for plotting/debugging."""


def _default_tol(effective_sigma: float | np.ndarray) -> float | np.ndarray:
    """The default absolute-error tolerance for the "converged"
    classification, derived from a noise floor.

    The classical central-difference scaling gives an achievable error on
    the order of ``noise_floor**(2/3)`` (see :mod:`fiddy.step_size`); the
    constant in front of that scaling is not exactly 1 in practice (finite
    ladder, residual extrapolation error), hence the safety factor rather
    than treating the bare scaling as a hard tolerance. The two-chain
    corroboration fix (see :mod:`fiddy.extrapolation`) deliberately
    reports a more conservative (larger, more honest) error estimate than
    a single uncorroborated chain would -- each of the two independent
    half-ladders has fewer points, so even a well-resolved direction has
    some inherent cross-chain disagreement. Calibrated against a real
    ODE-based likelihood: several genuinely well-resolved parameters
    (relative error ~1e-8) needed up to ~42x this base scaling to be
    classified "converged" rather than "noise_dominated"; 50x leaves some
    margin. `fiddy.check.check_gradient`'s `safety_factor` parameter provides a
    proper per-direction, auto-derived tolerance on top of this; this is
    only the engine's own internal, preliminary classification. Elementwise
    over `effective_sigma`, so a per-output-component noise floor produces
    a per-component tolerance.

    :param effective_sigma: The noise level (scalar or per-output array)
        to derive a tolerance from.
    :return: The tolerance, same shape as `effective_sigma`.
    """
    return 50 * np.maximum(effective_sigma, np.finfo(float).eps) ** (2 / 3)


def _noise_floor_component(noise: NoiseFloor, j: int) -> NoiseFloor:
    """Slice a (possibly multi-output) `NoiseFloor` down to one output
    component's scalar view. A no-op for an already-scalar `NoiseFloor`.

    :param noise: The (possibly multi-output) noise floor.
    :param j: The output component's index.
    :return: The noise floor for output component `j` alone.
    """
    sigma = noise.sigma
    if not isinstance(sigma, np.ndarray) or sigma.ndim == 0:
        return noise
    confident_j = bool(np.atleast_1d(noise.confident)[j])
    level = noise.level
    if isinstance(level, np.ndarray):
        level_j = int(level[j])
        level_value = level_j if (confident_j and level_j >= 0) else None
    else:
        level_value = level
    sigmas_j = [
        None if s is None else float(np.atleast_1d(s)[j]) for s in noise.sigmas
    ]
    return NoiseFloor(
        sigma=float(sigma[j]),
        level=level_value,
        confident=confident_j,
        sigmas=sigmas_j,
    )


def _extrapolation_component(
    extrapolation: ExtrapolationResult, j: int
) -> ExtrapolationResult:
    """Slice a (multi-output) `ExtrapolationResult` down to one output
    component's scalar view -- used to build each output's own
    `DerivativeEstimate` after one shared, vectorized extrapolation call.

    :param extrapolation: The (multi-output) extrapolation result.
    :param j: The output component's index.
    :return: The extrapolation result for output component `j` alone.
    """
    diagonal = extrapolation.diagonal
    return ExtrapolationResult(
        value=float(np.atleast_1d(extrapolation.value)[j]),
        error_estimate=float(np.atleast_1d(extrapolation.error_estimate)[j]),
        diagonal=diagonal[:, j] if diagonal.ndim > 1 else diagonal,
        best_index=int(np.atleast_1d(extrapolation.best_index)[j]),
        chain_a_value=float(np.atleast_1d(extrapolation.chain_a_value)[j]),
        chain_b_value=float(np.atleast_1d(extrapolation.chain_b_value)[j]),
        chain_a_error=float(np.atleast_1d(extrapolation.chain_a_error)[j]),
        chain_b_error=float(np.atleast_1d(extrapolation.chain_b_error)[j]),
        disagreement=float(np.atleast_1d(extrapolation.disagreement)[j]),
    )


def _discontinuity_component(
    discontinuity: DiscontinuityCheck, j: int
) -> DiscontinuityCheck:
    """Slice a (multi-output) `DiscontinuityCheck` down to one output
    component's scalar view, mirroring `_extrapolation_component`.

    :param discontinuity: The (multi-output) discontinuity check result.
    :param j: The output component's index.
    :return: The discontinuity check result for output component `j` alone.
    """
    gap_values = discontinuity.gap_values
    return DiscontinuityCheck(
        suspected=bool(np.atleast_1d(discontinuity.suspected)[j]),
        ladder=discontinuity.ladder,
        gap_values=gap_values[:, j] if gap_values.ndim > 1 else gap_values,
        best_index=int(np.atleast_1d(discontinuity.best_index)[j]),
        reference_index=int(np.atleast_1d(discontinuity.reference_index)[j]),
        predicted_gap_b=float(np.atleast_1d(discontinuity.predicted_gap_b)[j]),
        residual=float(np.atleast_1d(discontinuity.residual)[j]),
        noise_budget=float(np.atleast_1d(discontinuity.noise_budget)[j]),
    )


def _cross_regime_component(
    cross_regime: CrossRegimeCheck, j: int
) -> CrossRegimeCheck:
    """Slice a (multi-output) `CrossRegimeCheck` down to one output
    component's scalar view, mirroring `_discontinuity_component`.

    :param cross_regime: The (multi-output) cross-regime check result.
    :param j: The output component's index.
    :return: The cross-regime check result for output component `j` alone.
    """
    return CrossRegimeCheck(
        suspected=bool(np.atleast_1d(cross_regime.suspected)[j]),
        disagreement=float(np.atleast_1d(cross_regime.disagreement)[j]),
        noise_budget=float(np.atleast_1d(cross_regime.noise_budget)[j]),
    )


def _estimate_from_ladder(
    ladder: np.ndarray,
    f_0: np.ndarray,
    f_plus: np.ndarray,
    f_minus: np.ndarray,
    noise: NoiseFloor,
    tol: float | np.ndarray,
    nondet_tol: float,
    discontinuity_noise_sigma: float | np.ndarray | None = None,
    far_ladder: np.ndarray | None = None,
    far_f_plus: np.ndarray | None = None,
    far_f_minus: np.ndarray | None = None,
) -> list[DerivativeEstimate]:
    """Shared analysis: given a ladder and its evaluations, produce one
    :class:`DerivativeEstimate` per output component. No function
    evaluations happen here -- :func:`estimate_directional_derivative`,
    :func:`estimate_gradient`, and :func:`estimate_jacobian` all decide
    and dispatch their whole batch of evaluations first, then call this
    purely on the results.

    :param ladder: The step-size ladder the evaluations were taken at.
    :param f_0: The function value at the unperturbed point, shape
        ``(n_outputs,)`` (``n_outputs == 1`` for the single-output entry
        points).
    :param f_plus: The ladder's forward perturbed-point evaluations,
        shape ``(n_rungs, n_outputs)``.
    :param f_minus: The ladder's backward perturbed-point evaluations,
        shape ``(n_rungs, n_outputs)``.
    :param noise: The noise floor the ladder was built from.
    :param tol: A scalar (applied to every output) or a per-output-
        component array of shape ``(n_outputs,)``.
    :param nondet_tol: Forwarded to :func:`fiddy.discontinuity.check_discontinuity`.
    :param discontinuity_noise_sigma: The noise level
        :func:`fiddy.discontinuity.check_discontinuity` should treat the
        shared `ladder` as calibrated to -- defaults to `noise.sigma`
        (correct for the single-output entry points, where the ladder
        *was* built from this component's own noise). `estimate_jacobian`
        passes the shared, ladder-*driving* sigma instead (the max across
        output components -- see its docstring): a component whose own
        noise floor is much smaller than that shared value was evaluated
        at step sizes much larger than its own noise alone would justify,
        so genuine truncation-error curvature at those larger steps must
        not be mistaken for a kink just because it dwarfs that
        component's own (much tighter) noise budget -- a real false
        positive found via real-model multi-output validation, not a
        hypothetical.
    :param far_ladder: An independently-anchored, much-smaller-scale
        "far" ladder (see :func:`fiddy.step_size.build_step_ladder`,
        called with ``noise_floor=numpy.finfo(float).eps``) -- catches a
        hidden discontinuity closer to the evaluation point than
        `ladder`'s own finest rung (see the `fiddy.discontinuity` module
        docstring). `None` skips this check entirely (only used
        internally/for tests exercising the base ladder logic in
        isolation -- every public entry point always supplies one).
    :param far_f_plus: The far ladder's forward perturbed-point
        evaluations, shape ``(len(far_ladder), n_outputs)``. Required if
        `far_ladder` is given.
    :param far_f_minus: The far ladder's backward perturbed-point
        evaluations. Required if `far_ladder` is given.
    :return: A list of length ``n_outputs`` (length 1 for the single-
        output entry points, which take element 0).
    """
    n_outputs = f_0.shape[0]
    tol_arr = np.broadcast_to(np.atleast_1d(tol), (n_outputs,)).astype(float)
    if discontinuity_noise_sigma is None:
        discontinuity_noise_sigma = noise.sigma
    discontinuity_noise_sigma_arr = np.broadcast_to(
        np.atleast_1d(discontinuity_noise_sigma), (n_outputs,)
    ).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # `ladder` can legitimately collapse to an all-zero step (e.g.
        # `clamp_step_to_bounds` finds zero room to move at all because
        # `point` sits exactly on its declared bound in this direction --
        # see that function's own docstring). The resulting NaN/inf is
        # the correct, honest outcome: `converged_arr`/`suspected_arr`
        # below classify it as "noise_dominated" (comparisons against NaN
        # are always False), not a bug to work around.
        central_values = (f_plus - f_minus) / (2 * ladder[:, None])
        gap_values = (f_plus - 2 * f_0[None, :] + f_minus) / ladder[:, None]

    extrapolation = extrapolate_central_differences(ladder, central_values)

    discontinuity = check_discontinuity(
        ladder,
        gap_values,
        extrapolation.best_index,
        noise_sigma=discontinuity_noise_sigma_arr,
        nondet_tol=nondet_tol,
    )

    value_arr = np.atleast_1d(extrapolation.value)
    error_estimate_arr = np.atleast_1d(extrapolation.error_estimate)

    far_extrapolation = None
    far_discontinuity = None
    cross_regime = None
    if far_ladder is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            far_central_values = (far_f_plus - far_f_minus) / (
                2 * far_ladder[:, None]
            )
            far_gap_values = (
                far_f_plus - 2 * f_0[None, :] + far_f_minus
            ) / far_ladder[:, None]

        far_extrapolation = extrapolate_central_differences(
            far_ladder, far_central_values
        )
        far_discontinuity = check_discontinuity(
            far_ladder,
            far_gap_values,
            far_extrapolation.best_index,
            noise_sigma=discontinuity_noise_sigma_arr,
            nondet_tol=nondet_tol,
        )
        cross_regime = check_cross_regime_disagreement(
            value_arr,
            np.atleast_1d(far_extrapolation.value),
            np.atleast_1d(far_extrapolation.error_estimate),
            noise_sigma=discontinuity_noise_sigma_arr,
            nondet_tol=nondet_tol,
        )
        # Unlike the two-chain disagreement in `extrapolate_central_
        # differences` (always folded into `error_estimate`, since both
        # chains share the same noise-floor calibration, so their
        # disagreement is always a meaningful comparison), the far
        # ladder's disagreement is *expected* to be large whenever a
        # function's real noise floor sits far above machine epsilon (the
        # far ladder's anchor) -- that is not itself evidence the main
        # ladder's estimate is wrong. Only fold it into the status
        # (`suspected_arr` below), the same way `discontinuity.suspected`
        # already is, not into `error_estimate`/the tolerance check.

    # Absolute check: has the extrapolation converged to within the
    # noise-derived tolerance at all?
    #
    # Relative check: is the error estimate small *compared to the value
    # itself*? An absolute tolerance alone is not enough -- if the true
    # derivative's magnitude happens to be near or below the noise-limited
    # achievable precision (see the `fiddy.extrapolation` module docstring
    # for a real case of this), `error_estimate` can be comfortably under
    # `tol` while still being several times *larger* than the reported
    # value, i.e. the value carries no real signal.
    # `np.finfo(float).eps` is a pure divide-by-zero guard here, not a
    # scale-setting choice -- it only matters when `value` itself is at
    # (near-)machine-precision, in which case any nonzero error estimate
    # correctly fails this check (an honest "can't resolve this" rather
    # than a confident answer -- this is the intended behavior for
    # near-zero-gradient directions, which are fundamentally
    # indistinguishable from noise by finite differences alone).
    relative_error_arr = error_estimate_arr / np.maximum(
        np.abs(value_arr), np.finfo(float).eps
    )
    default_rtol = 0.5
    converged_arr = (error_estimate_arr <= tol_arr) & (
        relative_error_arr <= default_rtol
    )
    # Deliberately does NOT fold `far_discontinuity.suspected` in: the far
    # ladder is often *itself* genuinely noise-dominated for a function
    # whose real noise floor sits far above machine epsilon (its anchor),
    # in which case its own adjacent-rung gap comparison will routinely
    # look like a kink (wildly oscillating central differences) with no
    # bearing on whether the *main* ladder's value is trustworthy -- the
    # informative cross-check is `cross_regime` (whether the two ladders'
    # independently-obtained *values* agree), not the far ladder's own
    # internal consistency.
    suspected_arr = np.atleast_1d(discontinuity.suspected)
    if far_ladder is not None:
        suspected_arr = suspected_arr | np.atleast_1d(cross_regime.suspected)

    results = []
    for j in range(n_outputs):
        if suspected_arr[j]:
            status = "discontinuity_suspected"
        elif converged_arr[j]:
            status = "converged"
        else:
            status = "noise_dominated"

        diagnostics = {
            "ladder": ladder,
            "central_values": central_values[:, j],
            "gap_values": gap_values[:, j],
            "tol": float(tol_arr[j]),
            "relative_error": float(relative_error_arr[j]),
        }
        if far_ladder is not None:
            diagnostics["far_ladder"] = far_ladder
            diagnostics["far_central_values"] = far_central_values[:, j]

        results.append(
            DerivativeEstimate(
                value=float(value_arr[j]),
                error_estimate=float(error_estimate_arr[j]),
                status=status,
                noise=_noise_floor_component(noise, j),
                extrapolation=_extrapolation_component(extrapolation, j),
                discontinuity=_discontinuity_component(discontinuity, j),
                far_extrapolation=(
                    _extrapolation_component(far_extrapolation, j)
                    if far_ladder is not None
                    else None
                ),
                far_discontinuity=(
                    _discontinuity_component(far_discontinuity, j)
                    if far_ladder is not None
                    else None
                ),
                cross_regime=(
                    _cross_regime_component(cross_regime, j)
                    if far_ladder is not None
                    else None
                ),
                diagnostics=diagnostics,
            )
        )
    return results


def estimate_directional_derivative(
    function: Type.FUNCTION,
    point: Type.POINT,
    direction: Type.DIRECTION,
    tol: float | None = None,
    noise_floor: float | None = None,
    nondet_tol: float = 0.0,
    n_rungs: int = 8,
    step_ratio: float = 2.0,
    n_rungs_far: int = 4,
    step_ratio_far: float = 10.0,
    bounds: Type.BOUNDS | None = None,
    noise_floor_strategy: str = "auto",
    executor: Executor | None = None,
) -> DerivativeEstimate:
    """Estimate a directional derivative with no user-supplied step size.

    Only `function`, `point`, and `direction` are required.

    :param function: The blackbox function.
    :param point: The point to estimate the derivative at.
    :param direction: The direction to estimate the derivative along.
    :param tol: Optional absolute-error tolerance hint; bypasses the
        default noise-derived tolerance.
    :param noise_floor: Optional noise-floor hint (e.g. an ODE solver's
        own abstol); bypasses empirical noise estimation.
    :param nondet_tol: An expected magnitude of legitimate, non-noise
        value nondeterminism at the *same* point -- e.g. a simulator
        giving slightly different values with vs. without sensitivities
        enabled. Folded into the effective noise floor used everywhere
        below, the same way `torch.autograd.gradcheck`'s `nondet_tol`
        (``TorchGradcheck`` in ``doc/references.bib``) tolerates a
        non-deterministic analytic gradient rather than treating it as an
        error. Defaults to 0 (no known nondeterminism).
    :param n_rungs: Forwarded to :func:`fiddy.step_size.build_step_ladder`.
    :param step_ratio: Forwarded to
        :func:`fiddy.step_size.build_step_ladder`.
    :param n_rungs_far: Number of rungs in a second, independently-
        anchored "far" ladder, always evaluated and cross-checked against
        the main ladder's value (see :mod:`fiddy.discontinuity`'s module
        docstring, ``check_cross_regime_disagreement``) -- catches a
        hidden parameter-space discontinuity (e.g. an SBML event trigger)
        whose crossing perturbation is smaller than the main ladder's own
        finest rung, which neither the main ladder's extrapolation nor
        its own adjacent-rung discontinuity check can see on their own
        (confirmed via 13 distinct real AMICI/SBML models). Anchored at
        ``scale * numpy.finfo(float).eps ** (1 / 3)`` -- the classical
        central-difference optimum evaluated at the theoretical minimum
        possible noise floor (pure rounding error only), not an arbitrary
        constant -- via the same, unmodified
        :func:`fiddy.step_size.build_step_ladder`. 4 is the minimum
        :func:`fiddy.extrapolation.extrapolate_central_differences`
        accepts (so the far ladder gets its own error estimate and
        internal discontinuity check "for free"). Always dispatched, at
        a real, known extra cost (`2 * n_rungs_far` more evaluations per
        direction) -- there is no cheap signal in the main ladder's own
        data that could safely skip this (the entire nature of this
        failure mode: the wrong branch is itself locally smooth, so
        nothing about the main ladder's data hints anything is wrong).
    :param step_ratio_far: Ratio between the far ladder's rungs. `10.0`
        (vs. the main ladder's `2.0`) trades table resolution for reach:
        the far ladder needs to get far below the main ladder's finest
        rung, not finely resolve an extrapolation order.
    :param bounds: Optional per-parameter valid domain, forwarded to both
        the noise-floor probe and :func:`fiddy.step_size.build_step_ladder`
        -- see :func:`fiddy.step_size.clamp_step_to_bounds`. `None` (the
        default) disables clamping entirely. `point` itself must already
        satisfy `bounds`; a violation raises `ValueError` rather than
        silently collapsing every step to zero.
    :param noise_floor_strategy: See :func:`estimate_gradient` -- with
        only one direction here, `"per_direction"`/escalated `"auto"`
        probes along `direction` itself rather than the shared,
        all-ones default.
    :param executor: How to dispatch the batch of
        ``2 * (n_rungs + n_rungs_far) + 1`` ladder evaluations
        (``f(x0)``, ``f(x0 +/- h)`` per main and far rung) -- e.g.
        :class:`fiddy.executor.JoblibExecutor` to run them in parallel.
        Defaults to :class:`fiddy.executor.SequentialExecutor`. Also
        forwarded to noise-floor estimation (its own, separate probe
        batch). Every point needed is decided upfront and dispatched as a
        single batch per phase, so switching executors changes wall-clock
        time only, never the result.
    :return: The directional derivative estimate.
    :raises ValueError: If `point` violates `bounds`, or
        `noise_floor_strategy` is invalid.
    """
    if executor is None:
        executor = SequentialExecutor()
    function = _ensure_function(function)
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    _validate_point_in_bounds(point, bounds)

    if noise_floor is None:
        noise = _resolve_noise_floors(
            function,
            point,
            [direction],
            noise_floor_strategy,
            executor,
            bounds,
        )[0]
    else:
        noise = NoiseFloor(
            sigma=noise_floor, level=None, confident=True, sigmas=[]
        )
    # Only the first (flattened) output component is estimated here (see
    # module docstring) -- `estimate_jacobian` is the multi-output entry
    # point. Slicing the noise floor down to component 0 keeps a
    # multi-output raw function's *other* components from affecting this
    # single-output estimate's step size/tolerance.
    noise = _noise_floor_component(noise, 0)
    effective_sigma = max(noise.sigma, nondet_tol)

    if tol is None:
        tol = _default_tol(effective_sigma)

    ladder = build_step_ladder(
        point,
        direction,
        effective_sigma,
        n_rungs=n_rungs,
        step_ratio=step_ratio,
        bounds=bounds,
    )
    far_ladder = build_step_ladder(
        point,
        direction,
        np.finfo(float).eps,
        n_rungs=n_rungs_far,
        step_ratio=step_ratio_far,
        bounds=bounds,
    )

    # Every point the whole ladder (main and far) needs is decided upfront
    # and dispatched together as one batch through `executor` -- this is
    # what makes parallelizing the ladder a matter of swapping the
    # executor, not restructuring this function.
    n = len(ladder)
    n_far = len(far_ladder)
    batch_points = (
        [point]
        + [point + h * direction for h in ladder]
        + [point - h * direction for h in ladder]
        + [point + h * direction for h in far_ladder]
        + [point - h * direction for h in far_ladder]
    )
    batch_results = np.array(
        [np.asarray(v) for v in executor(function, batch_points)]
    ).reshape(len(batch_points), -1)[:, :1]
    f_0 = batch_results[0]
    f_plus = batch_results[1 : 1 + n]
    f_minus = batch_results[1 + n : 1 + 2 * n]
    far_f_plus = batch_results[1 + 2 * n : 1 + 2 * n + n_far]
    far_f_minus = batch_results[1 + 2 * n + n_far :]

    return _estimate_from_ladder(
        ladder,
        f_0,
        f_plus,
        f_minus,
        noise,
        tol,
        nondet_tol,
        far_ladder=far_ladder,
        far_f_plus=far_f_plus,
        far_f_minus=far_f_minus,
    )[0]


def estimate_gradient(
    function: Type.FUNCTION,
    point: Type.POINT,
    directions: list[Type.DIRECTION] | None = None,
    tol: float | None = None,
    noise_floor: float | None = None,
    nondet_tol: float = 0.0,
    n_rungs: int = 8,
    step_ratio: float = 2.0,
    n_rungs_far: int = 4,
    step_ratio_far: float = 10.0,
    bounds: Type.BOUNDS | None = None,
    noise_floor_strategy: str = "auto",
    executor: Executor | None = None,
) -> list[DerivativeEstimate]:
    """Estimate derivatives along several directions at once.

    Shares one noise-floor probe across every direction by default (see
    :func:`fiddy.noise.estimate_model_noise_floor`) and dispatches *every*
    direction's ladder evaluations as a single combined batch through
    `executor` -- checking N directions costs one batch dispatch, not N.

    :param function: The blackbox function.
    :param point: The point to estimate the gradient at.
    :param directions: Defaults to the standard basis (one direction per
        component of `point`), i.e. the full gradient.
    :param tol: See :func:`estimate_directional_derivative` -- applied
        identically to every direction.
    :param noise_floor: See :func:`estimate_directional_derivative` --
        applied identically to every direction.
    :param nondet_tol: See :func:`estimate_directional_derivative` --
        applied identically to every direction.
    :param n_rungs: See :func:`estimate_directional_derivative` -- applied
        identically to every direction.
    :param step_ratio: See :func:`estimate_directional_derivative` --
        applied identically to every direction.
    :param n_rungs_far: See :func:`estimate_directional_derivative` --
        applied identically to every direction.
    :param step_ratio_far: See :func:`estimate_directional_derivative` --
        applied identically to every direction.
    :param bounds: See :func:`estimate_directional_derivative` -- applied
        identically to every direction.
    :param noise_floor_strategy: How to estimate the noise floor(s) this
        gradient's directions are checked with:

        - ``"shared"``: one probe along :func:`fiddy.noise.
          default_probe_direction`, reused for every direction (today's
          only behavior, cheapest -- one 15-point probe regardless of
          how many directions are checked).
        - ``"per_direction"``: an independent, bounds-clamped probe along
          *each* direction, from one combined batch dispatch (not one
          `executor` round per direction). Costs roughly `32/17 ~ 1.9x`
          the evaluations of `"shared"` for `n_rungs=8` as the number of
          directions grows (the per-direction step-size *ladder* --
          already one per direction even under `"shared"` -- dominates
          total cost once there are more than a handful of directions;
          this is not the "Nx" blowup a naive per-parameter noise floor
          might suggest).
        - ``"auto"`` (the default): try `"shared"` first, at no extra
          cost; if it comes back unconfident or degenerate (see
          :func:`fiddy.noise.noise_floor_is_confident`), transparently
          re-probe `"per_direction"` instead of silently returning a
          crushed, spuriously tight tolerance for every direction.

        Found necessary in practice: a *single* parameter close to its
        own declared `bounds` can crush the *shared* probe's step down
        to near-zero for *every* direction at once (not just that
        parameter's own), even though the parameter itself may be
        perfectly valid -- confirmed on real PEtab benchmark models,
        where this turned an otherwise-correct check into 100%
        "noise_dominated" results. `"per_direction"`/escalated `"auto"`
        isolates this: only the genuinely bound-constrained directions
        end up honestly uncertain, every other direction resolves
        cleanly and confidently.
    :param executor: See :func:`estimate_directional_derivative` --
        applied identically to every direction.
    :return: One `DerivativeEstimate` per direction, in the same order.
    :raises ValueError: If `point` violates `bounds`, or
        `noise_floor_strategy` is invalid.
    """
    if executor is None:
        executor = SequentialExecutor()
    function = _ensure_function(function)
    point = np.asarray(point, dtype=float)
    if directions is None:
        directions = list(np.eye(len(point)))
    directions = [np.asarray(d, dtype=float) for d in directions]
    _validate_point_in_bounds(point, bounds)

    if noise_floor is None:
        noises = _resolve_noise_floors(
            function, point, directions, noise_floor_strategy, executor, bounds
        )
    else:
        shared = NoiseFloor(
            sigma=noise_floor, level=None, confident=True, sigmas=[]
        )
        noises = [shared] * len(directions)
    # Only the first (flattened) output component is estimated here (see
    # module docstring) -- `estimate_jacobian` is the multi-output entry
    # point.
    noises = [_noise_floor_component(n, 0) for n in noises]
    effective_sigmas = [max(n.sigma, nondet_tol) for n in noises]

    if tol is None:
        tols = [_default_tol(s) for s in effective_sigmas]
    else:
        tols = [tol] * len(directions)

    ladders = [
        build_step_ladder(
            point,
            d,
            effective_sigmas[i],
            n_rungs=n_rungs,
            step_ratio=step_ratio,
            bounds=bounds,
        )
        for i, d in enumerate(directions)
    ]
    far_ladders = [
        build_step_ladder(
            point,
            d,
            np.finfo(float).eps,
            n_rungs=n_rungs_far,
            step_ratio=step_ratio_far,
            bounds=bounds,
        )
        for d in directions
    ]

    # f(x0) does not depend on direction, so it is evaluated once and
    # shared across every direction's ladder (main and far), not once per
    # direction.
    batch_points = [point]
    for d, ladder, far_ladder in zip(
        directions, ladders, far_ladders, strict=True
    ):
        batch_points += [point + h * d for h in ladder]
        batch_points += [point - h * d for h in ladder]
        batch_points += [point + h * d for h in far_ladder]
        batch_points += [point - h * d for h in far_ladder]

    batch_results = np.array(
        [np.asarray(v) for v in executor(function, batch_points)]
    ).reshape(len(batch_points), -1)[:, :1]

    f_0 = batch_results[0]
    results = []
    offset = 1
    for i, (ladder, far_ladder) in enumerate(
        zip(ladders, far_ladders, strict=True)
    ):
        n = len(ladder)
        n_far = len(far_ladder)
        f_plus = batch_results[offset : offset + n]
        f_minus = batch_results[offset + n : offset + 2 * n]
        offset += 2 * n
        far_f_plus = batch_results[offset : offset + n_far]
        far_f_minus = batch_results[offset + n_far : offset + 2 * n_far]
        offset += 2 * n_far
        results.append(
            _estimate_from_ladder(
                ladder,
                f_0,
                f_plus,
                f_minus,
                noises[i],
                tols[i],
                nondet_tol,
                far_ladder=far_ladder,
                far_f_plus=far_f_plus,
                far_f_minus=far_f_minus,
            )[0]
        )
    return results


@dataclass
class JacobianEstimate:
    """Per-(output component, direction) derivative estimates from one
    shared batch of evaluations -- multi-output support.

    Checking every output component of a bundled multi-output function
    (e.g. a model's `x`/`y`/`sigmay`/`llh` sensitivities) costs the same
    batch of perturbed-point evaluations as checking one, since that
    sharing is the entire reason such a function bundles several outputs
    into one evaluation in the first place (see :func:`estimate_jacobian`).

    Indexed as ``estimates[output_index][direction_index]``. If the
    checked function returned a named dict (see :class:`fiddy.Function`),
    `schema` lets a named output be looked up by name via :meth:`output`
    instead of a raw flat integer index.
    """

    estimates: list[list[DerivativeEstimate]]
    """Indexed as ``estimates[output_index][direction_index]``."""
    schema: OutputSchema | None = None
    """The checked function's output schema, or `None` if it returned a
    plain array (not a dict)."""

    @property
    def n_outputs(self) -> int:
        return len(self.estimates)

    @property
    def n_directions(self) -> int:
        return len(self.estimates[0]) if self.estimates else 0

    def _resolve_flat_index(self, name_or_index: str | int) -> int:
        if not isinstance(name_or_index, str):
            return int(name_or_index)
        if self.schema is None or not self.schema.is_structured:
            raise KeyError(
                f"No named output {name_or_index!r}: the checked function "
                "did not return a named (dict) output."
            )
        if name_or_index not in self.schema.names:
            raise KeyError(
                f"Unknown output name {name_or_index!r}; available: "
                f"{self.schema.names}"
            )
        # A named output may itself bundle more than one component (e.g. a
        # vector-valued `y`); this returns the *first* component's row --
        # index `.estimates` directly for a specific component of a
        # multi-component named output.
        return self.schema.slices[name_or_index].start

    def output(self, name_or_index: str | int) -> list[DerivativeEstimate]:
        """One output component's derivative estimate for every direction
        (one row of the Jacobian).

        :param name_or_index: A named output (if the checked function
            returned a dict, see :class:`fiddy.output.OutputSchema`), or
            a flat integer index into the bundled output.
        :return: The derivative estimates for that output component, one
            per direction.
        :raises KeyError: If `name_or_index` names an output the checked
            function did not return, or the function's output wasn't
            named at all.
        """
        return self.estimates[self._resolve_flat_index(name_or_index)]

    @property
    def values(self) -> np.ndarray:
        """Jacobian of point-estimate values, shape
        ``(n_outputs, n_directions)``."""
        return np.array([[e.value for e in row] for row in self.estimates])

    @property
    def statuses(self) -> list[list[str]]:
        """Convergence status per (output, direction), same shape as
        :attr:`values`."""
        return [[e.status for e in row] for row in self.estimates]


def estimate_jacobian(
    function: Type.FUNCTION,
    point: Type.POINT,
    directions: list[Type.DIRECTION] | None = None,
    tol: float | None = None,
    noise_floor: float | None = None,
    nondet_tol: float = 0.0,
    n_rungs: int = 8,
    step_ratio: float = 2.0,
    n_rungs_far: int = 4,
    step_ratio_far: float = 10.0,
    bounds: Type.BOUNDS | None = None,
    noise_floor_strategy: str = "auto",
    executor: Executor | None = None,
) -> JacobianEstimate:
    """Estimate derivatives of every output component along several
    directions at once -- a full Jacobian, not just a gradient of one
    scalar output.

    Checking all of a model's forward sensitivities together (e.g. a
    bundled state/observable/likelihood sensitivities `sx`/`sy`/`sllh`,
    not just its scalar objective) is a central use case this engine is
    meant to support well, not an edge case. `function` may return a
    plain array or a named dict (see :class:`fiddy.Function`/
    :mod:`fiddy.output`); either way, every output component shares the
    *same* batch of perturbed-point evaluations per direction -- the very
    reason a multi-output function bundles several outputs into one
    evaluation in the first place -- so checking N outputs costs no more
    function evaluations than checking 1.

    The main step-size ladder for a given direction is necessarily shared
    across every output component (it drives the one batch of evaluations
    that produces every output's value at once), built from the *largest*
    per-output noise floor so it stays safe for the noisiest component;
    each output's own convergence classification -- including its
    discontinuity check -- still uses its own noise floor and tolerance,
    not the shared ladder-driving one (a component whose own noise floor
    is much smaller than that shared value having its kink-detection
    budget calibrated to the shared value instead was a real false
    positive found via multi-output validation; `curvature_rtol`'s
    relative budget term, not sigma-sharing, is what now guards against
    the *original* concern -- genuine truncation curvature at oversized
    steps being mistaken for a kink). The far ladder (see `n_rungs_far`
    below) is likewise shared across output components per direction, for
    the same batching reason.

    :param function: The blackbox function.
    :param point: The point to estimate the Jacobian at.
    :param directions: Defaults to the standard basis (one direction per
        component of `point`), i.e. the full Jacobian.
    :param tol: Optional absolute-error tolerance hint applied to every
        output component; bypasses each component's own noise-derived
        tolerance.
    :param noise_floor: See :func:`estimate_directional_derivative` --
        applied identically to every direction and output component (if
        given, treated as already shared across every output).
    :param nondet_tol: See :func:`estimate_directional_derivative` --
        applied identically to every direction and output component.
    :param n_rungs: See :func:`estimate_directional_derivative` -- applied
        identically to every direction and output component.
    :param step_ratio: See :func:`estimate_directional_derivative` --
        applied identically to every direction and output component.
    :param n_rungs_far: See :func:`estimate_directional_derivative` --
        applied identically to every direction and output component; the
        far ladder is shared across output components per direction, same
        as the main ladder.
    :param step_ratio_far: See :func:`estimate_directional_derivative` --
        applied identically to every direction and output component.
    :param bounds: See :func:`estimate_directional_derivative` -- applied
        identically to every direction and output component.
    :param noise_floor_strategy: See :func:`estimate_gradient` -- applied
        identically to every direction; each direction's own probe (under
        `"per_direction"`/escalated `"auto"`) measures every output
        component at once, same as the shared probe already does, so
        multi-output support costs nothing extra here either.
    :param executor: See :func:`estimate_directional_derivative` --
        applied identically to every direction and output component.
    :return: The Jacobian estimate, indexed
        `[output_index][direction_index]` (or by output name, if
        `function` returned a named dict -- see
        :meth:`JacobianEstimate.output`).
    :raises ValueError: If `point` violates `bounds`, or
        `noise_floor_strategy` is invalid.
    """
    if executor is None:
        executor = SequentialExecutor()
    function = _ensure_function(function)
    point = np.asarray(point, dtype=float)
    _validate_point_in_bounds(point, bounds)
    # Evaluated directly (not through `executor`) so `function.schema` is
    # guaranteed to be populated on *this* (main-process) object: with
    # `JoblibExecutor`, batch evaluations run in separate worker
    # processes, each with its own pickled copy of `function` -- their
    # schema never makes it back to the object referenced here, but a
    # direct call always does. Also serves as `f_0`, so this costs no
    # extra evaluation (just relocates the unperturbed-point evaluation
    # out of the batch).
    f_0 = np.asarray(function(point), dtype=float)
    n_outputs = f_0.shape[0]

    if directions is None:
        directions = list(np.eye(len(point)))
    directions = [np.asarray(d, dtype=float) for d in directions]

    if noise_floor is None:
        noises = _resolve_noise_floors(
            function, point, directions, noise_floor_strategy, executor, bounds
        )
    else:
        shared = NoiseFloor(
            sigma=noise_floor, level=None, confident=True, sigmas=[]
        )
        noises = [shared] * len(directions)

    # The step-size ladder is shared across every output component of a
    # given direction (one batch of evaluations serves them all -- see
    # docstring above), so it must be driven by one scalar noise estimate
    # even when individual outputs' noise floors differ hugely: the max
    # keeps the ladder safe (large enough) for the noisiest component, at
    # the cost of some sub-optimality for quieter ones.
    effective_sigmas_for_ladder = [
        max(float(np.max(np.atleast_1d(n.sigma))), nondet_tol) for n in noises
    ]

    ladders = [
        build_step_ladder(
            point,
            d,
            effective_sigmas_for_ladder[i],
            n_rungs=n_rungs,
            step_ratio=step_ratio,
            bounds=bounds,
        )
        for i, d in enumerate(directions)
    ]
    far_ladders = [
        build_step_ladder(
            point,
            d,
            np.finfo(float).eps,
            n_rungs=n_rungs_far,
            step_ratio=step_ratio_far,
            bounds=bounds,
        )
        for d in directions
    ]

    batch_points = []
    for d, ladder, far_ladder in zip(
        directions, ladders, far_ladders, strict=True
    ):
        batch_points += [point + h * d for h in ladder]
        batch_points += [point - h * d for h in ladder]
        batch_points += [point + h * d for h in far_ladder]
        batch_points += [point - h * d for h in far_ladder]

    batch_results = np.array(
        [np.asarray(v) for v in executor(function, batch_points)]
    ).reshape(len(batch_points), -1)

    tol_arrs = []
    for n_floor in noises:
        noise_sigma_arr = np.broadcast_to(
            np.atleast_1d(n_floor.sigma), (n_outputs,)
        ).astype(float)
        if tol is None:
            tol_arrs.append(
                _default_tol(np.maximum(noise_sigma_arr, nondet_tol))
            )
        else:
            tol_arrs.append(np.full(n_outputs, float(tol)))

    per_direction: list[list[DerivativeEstimate]] = []
    offset = 0
    for i, (ladder, far_ladder) in enumerate(
        zip(ladders, far_ladders, strict=True)
    ):
        n = len(ladder)
        n_far = len(far_ladder)
        f_plus = batch_results[offset : offset + n]
        f_minus = batch_results[offset + n : offset + 2 * n]
        offset += 2 * n
        far_f_plus = batch_results[offset : offset + n_far]
        far_f_minus = batch_results[offset + n_far : offset + 2 * n_far]
        offset += 2 * n_far
        per_direction.append(
            _estimate_from_ladder(
                ladder,
                f_0,
                f_plus,
                f_minus,
                noises[i],
                tol_arrs[i],
                nondet_tol,
                # Each output's own noise floor (`noises[i].sigma`,
                # already the default), not the shared ladder-driving
                # value -- see the docstring above for why sigma-sharing
                # is no longer needed here.
                far_ladder=far_ladder,
                far_f_plus=far_f_plus,
                far_f_minus=far_f_minus,
            )
        )

    # per_direction[i][j] is (direction i, output j); JacobianEstimate's
    # contract is estimates[output][direction], so transpose.
    estimates = [
        [per_direction[i][j] for i in range(len(directions))]
        for j in range(n_outputs)
    ]
    return JacobianEstimate(estimates=estimates, schema=function.schema)
