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
from typing import Any

import numpy as np

from .constants import Type
from .discontinuity import DiscontinuityCheck, check_discontinuity
from .executor import Executor, SequentialExecutor
from .extrapolation import ExtrapolationResult, extrapolate_central_differences
from .function import Function
from .noise import NoiseFloor, estimate_model_noise_floor
from .output import OutputSchema
from .step_size import build_step_ladder

__all__ = [
    "DerivativeEstimate",
    "JacobianEstimate",
    "estimate_directional_derivative",
    "estimate_gradient",
    "estimate_jacobian",
]


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
    """The kink/discontinuity cross-check result."""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Additional diagnostics (``ladder``, ``central_values``,
    ``gap_values``, ``tol``, ``relative_error``), for plotting/debugging."""


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
    margin. `fiddy.check.check_gradient`'s `k` parameter provides a
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


def _estimate_from_ladder(
    ladder: np.ndarray,
    f_0: np.ndarray,
    f_plus: np.ndarray,
    f_minus: np.ndarray,
    noise: NoiseFloor,
    tol: float | np.ndarray,
    nondet_tol: float,
    discontinuity_noise_sigma: float | np.ndarray | None = None,
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
    value_arr = np.atleast_1d(extrapolation.value)
    error_estimate_arr = np.atleast_1d(extrapolation.error_estimate)
    relative_error_arr = error_estimate_arr / np.maximum(
        np.abs(value_arr), np.finfo(float).eps
    )
    default_rtol = 0.5
    converged_arr = (error_estimate_arr <= tol_arr) & (
        relative_error_arr <= default_rtol
    )
    suspected_arr = np.atleast_1d(discontinuity.suspected)

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

        results.append(
            DerivativeEstimate(
                value=float(value_arr[j]),
                error_estimate=float(error_estimate_arr[j]),
                status=status,
                noise=_noise_floor_component(noise, j),
                extrapolation=_extrapolation_component(extrapolation, j),
                discontinuity=_discontinuity_component(discontinuity, j),
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
    :param executor: How to dispatch the batch of ``2 * n_rungs + 1``
        ladder evaluations (``f(x0)``, ``f(x0 +/- h)`` per rung) -- e.g.
        :class:`fiddy.executor.JoblibExecutor` to run them in parallel.
        Defaults to :class:`fiddy.executor.SequentialExecutor`. Also
        forwarded to noise-floor estimation (its own, separate probe
        batch). Every point needed is decided upfront and dispatched as a
        single batch per phase, so switching executors changes wall-clock
        time only, never the result.
    :return: The directional derivative estimate.
    """
    if executor is None:
        executor = SequentialExecutor()
    function = _ensure_function(function)
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if noise_floor is None:
        noise = estimate_model_noise_floor(function, point, executor=executor)
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
    )

    # Every point the whole ladder needs is decided upfront and dispatched
    # together as one batch through `executor` -- this is what makes
    # parallelizing the ladder a matter of swapping the executor, not
    # restructuring this function.
    n = len(ladder)
    batch_points = (
        [point]
        + [point + h * direction for h in ladder]
        + [point - h * direction for h in ladder]
    )
    batch_results = np.array(
        [np.asarray(v) for v in executor(function, batch_points)]
    ).reshape(len(batch_points), -1)[:, :1]
    f_0 = batch_results[0]
    f_plus = batch_results[1 : 1 + n]
    f_minus = batch_results[1 + n :]

    return _estimate_from_ladder(
        ladder, f_0, f_plus, f_minus, noise, tol, nondet_tol
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
    executor: Executor | None = None,
) -> list[DerivativeEstimate]:
    """Estimate derivatives along several directions at once.

    Shares one noise-floor probe across every direction (see
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
    :param executor: See :func:`estimate_directional_derivative` --
        applied identically to every direction.
    :return: One `DerivativeEstimate` per direction, in the same order.
    """
    if executor is None:
        executor = SequentialExecutor()
    function = _ensure_function(function)
    point = np.asarray(point, dtype=float)
    if directions is None:
        directions = list(np.eye(len(point)))
    directions = [np.asarray(d, dtype=float) for d in directions]

    if noise_floor is None:
        noise = estimate_model_noise_floor(function, point, executor=executor)
    else:
        noise = NoiseFloor(
            sigma=noise_floor, level=None, confident=True, sigmas=[]
        )
    # Only the first (flattened) output component is estimated here (see
    # module docstring) -- `estimate_jacobian` is the multi-output entry
    # point.
    noise = _noise_floor_component(noise, 0)
    effective_sigma = max(noise.sigma, nondet_tol)

    if tol is None:
        tol = _default_tol(effective_sigma)

    ladders = [
        build_step_ladder(
            point, d, effective_sigma, n_rungs=n_rungs, step_ratio=step_ratio
        )
        for d in directions
    ]

    # f(x0) does not depend on direction, so it is evaluated once and
    # shared across every direction's ladder, not once per direction.
    batch_points = [point]
    for d, ladder in zip(directions, ladders, strict=True):
        batch_points += [point + h * d for h in ladder]
        batch_points += [point - h * d for h in ladder]

    batch_results = np.array(
        [np.asarray(v) for v in executor(function, batch_points)]
    ).reshape(len(batch_points), -1)[:, :1]

    f_0 = batch_results[0]
    results = []
    offset = 1
    for ladder in ladders:
        n = len(ladder)
        f_plus = batch_results[offset : offset + n]
        f_minus = batch_results[offset + n : offset + 2 * n]
        offset += 2 * n
        results.append(
            _estimate_from_ladder(
                ladder, f_0, f_plus, f_minus, noise, tol, nondet_tol
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

    The step-size ladder for a given direction is necessarily shared
    across every output component (it drives the one batch of evaluations
    that produces every output's value at once), built from the *largest*
    per-output noise floor so it stays safe for the noisiest component;
    each output's own convergence classification still uses its own
    noise floor and tolerance, not the shared ladder-driving one.

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
    :param executor: See :func:`estimate_directional_derivative` --
        applied identically to every direction and output component.
    :return: The Jacobian estimate, indexed
        `[output_index][direction_index]` (or by output name, if
        `function` returned a named dict -- see
        :meth:`JacobianEstimate.output`).
    """
    if executor is None:
        executor = SequentialExecutor()
    function = _ensure_function(function)
    point = np.asarray(point, dtype=float)
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
        noise = estimate_model_noise_floor(function, point, executor=executor)
    else:
        noise = NoiseFloor(
            sigma=noise_floor, level=None, confident=True, sigmas=[]
        )

    # The step-size ladder is shared across every output component of a
    # given direction (one batch of evaluations serves them all -- see
    # docstring above), so it must be driven by one scalar noise estimate
    # even when individual outputs' noise floors differ hugely: the max
    # keeps the ladder safe (large enough) for the noisiest component, at
    # the cost of some sub-optimality for quieter ones.
    ladder_sigma = float(np.max(np.atleast_1d(noise.sigma)))
    effective_sigma_for_ladder = max(ladder_sigma, nondet_tol)

    ladders = [
        build_step_ladder(
            point,
            d,
            effective_sigma_for_ladder,
            n_rungs=n_rungs,
            step_ratio=step_ratio,
        )
        for d in directions
    ]

    batch_points = []
    for d, ladder in zip(directions, ladders, strict=True):
        batch_points += [point + h * d for h in ladder]
        batch_points += [point - h * d for h in ladder]

    batch_results = np.array(
        [np.asarray(v) for v in executor(function, batch_points)]
    ).reshape(len(batch_points), -1)

    noise_sigma_arr = np.broadcast_to(
        np.atleast_1d(noise.sigma), (n_outputs,)
    ).astype(float)
    if tol is None:
        tol_arr = _default_tol(np.maximum(noise_sigma_arr, nondet_tol))
    else:
        tol_arr = np.full(n_outputs, float(tol))

    per_direction: list[list[DerivativeEstimate]] = []
    offset = 0
    for ladder in ladders:
        n = len(ladder)
        f_plus = batch_results[offset : offset + n]
        f_minus = batch_results[offset + n : offset + 2 * n]
        offset += 2 * n
        per_direction.append(
            _estimate_from_ladder(
                ladder,
                f_0,
                f_plus,
                f_minus,
                noise,
                tol_arr,
                nondet_tol,
                discontinuity_noise_sigma=effective_sigma_for_ladder,
            )
        )

    # per_direction[i][j] is (direction i, output j); JacobianEstimate's
    # contract is estimates[output][direction], so transpose.
    estimates = [
        [per_direction[i][j] for i in range(len(directions))]
        for j in range(n_outputs)
    ]
    return JacobianEstimate(estimates=estimates, schema=function.schema)
