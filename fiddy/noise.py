"""Empirical, model-agnostic noise-floor estimation.

The core idea is inspired by the "ECNoise" algorithm (``MoreWild2011`` in
``doc/references.bib``): evaluate a handful of equally spaced points and
use the finite-difference table to find the order at which a function's
smooth (Taylor) part has been differenced away and only noise remains.
This is a *simplified* plateau-detection variant of their idea, not a
verbatim port -- see :func:`estimate_noise_floor` for the exact criterion
used.

A caller that already knows a trustworthy noise floor (e.g. an ODE
solver's own ``abstol``/``reltol``) can skip estimation entirely by
constructing a :class:`NoiseFloor` directly: ``NoiseFloor(sigma=value,
level=None, confident=True, sigmas=[])``. Do this with care: a solver's
own tolerances can be many orders of magnitude looser than the function's
*actual*, empirically measurable noise (e.g. an ODE solver's likelihood
output can be orders of magnitude noisier than its state/observable
tolerances alone would suggest) -- empirical estimation is the safer
default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import Type
from .executor import Executor, SequentialExecutor
from .step_size import clamp_step_to_bounds

__all__ = [
    "NoiseFloor",
    "default_probe_direction",
    "estimate_noise_floor",
    "estimate_model_noise_floor",
    "noise_floor_is_confident",
]


@dataclass
class NoiseFloor:
    sigma: float | np.ndarray
    """Estimated noise standard deviation (same units as the function
    output). A plain float for a scalar-output function; shape
    ``(n_outputs,)`` when the probed function returns several bundled
    outputs -- different bundled outputs (e.g. state trajectories vs. a
    likelihood) can have noise floors many orders of magnitude apart, so
    each gets its own estimate from the very same probe batch, at no
    extra evaluation cost."""
    level: int | None | np.ndarray
    """Finite-difference table order the estimate was taken at, or ``None``
    if no noise-dominated level was found (low-confidence fallback). For
    multi-output, an ``(n_outputs,)`` int array with ``-1`` standing in for
    "no plateau found" (one order per output component)."""
    confident: bool | np.ndarray
    """Whether a noise-dominated level was actually identified. An
    ``(n_outputs,)`` bool array for multi-output."""
    sigmas: list[float | np.ndarray | None] = field(
        default_factory=list, repr=False
    )
    """Per-order estimates ``sigma_level(k)`` for every order ``k`` that had
    enough table entries to compute, in order (``sigmas[0]`` is level 1,
    etc.). Kept for diagnostics/plotting (:func:`fiddy.plotting.
    plot_noise_floor`), not needed for the estimate itself. Each entry is
    an ``(n_outputs,)`` array for multi-output."""


def _noise_floor_probe_points(
    point: Type.POINT,
    direction: Type.DIRECTION,
    h0: float | None,
    n_points: int,
    bounds: Type.BOUNDS | None,
) -> list[np.ndarray]:
    """Build one direction's probe points for :func:`estimate_noise_floor`,
    bounds-clamped -- factored out so callers needing *several* directions'
    worth of probe points (e.g. per-direction noise-floor estimation in
    :mod:`fiddy.estimate`) can concatenate them into one combined batch
    and dispatch it through a single `executor` call, per fiddy's
    batch-then-analyze design, rather than one `executor` round per
    direction.

    :param point: The point to probe around.
    :param direction: The direction to probe along.
    :param h0: The spacing between probe points, or `None` to default to
        1% of the point's magnitude along `direction` (falling back to an
        absolute step of ``0.01`` near the origin).
    :param n_points: Number of equally spaced probe points.
    :param bounds: Optional per-parameter valid domain; see
        :func:`fiddy.step_size.clamp_step_to_bounds`.
    :return: The `n_points` probe points, in order.
    """
    if h0 is None:
        h0 = max(abs(float(np.dot(point, direction))), 1.0) * 1e-2

    max_offset = (n_points - 1) / 2 * h0
    clamped_max_offset = clamp_step_to_bounds(
        point, direction, max_offset, bounds
    )
    if max_offset > 0:
        h0 = h0 * (clamped_max_offset / max_offset)

    offsets = (np.arange(n_points) - (n_points - 1) / 2) * h0
    return [point + t * direction for t in offsets]


def _analyze_noise_table(
    values: np.ndarray, plateau_ratio: float
) -> NoiseFloor:
    """Turn one direction's probe evaluations into a :class:`NoiseFloor`,
    via the plateau-detection criterion described in
    :func:`estimate_noise_floor`'s own docstring -- factored out so the
    same analysis can run on results gathered from a combined,
    multi-direction batch dispatch (see :func:`_noise_floor_probe_points`).

    :param values: The probe evaluations, shape ``(n_points, n_outputs)``.
    :param plateau_ratio: See :func:`estimate_noise_floor`.
    :return: The estimated noise floor.
    """
    n_points, n_outputs = values.shape

    table = [values]
    for _ in range(n_points - 1):
        prev = table[-1]
        table.append(prev[1:] - prev[:-1])

    sigmas: list[np.ndarray | None] = []
    for level in range(1, n_points - 1):
        row = table[level]
        if row.shape[0] < 1:
            sigmas.append(None)
            continue
        variance_factor = math.comb(2 * level, level)
        sigmas.append(np.sqrt(np.mean(row**2, axis=0) / variance_factor))

    sigma = np.full(n_outputs, np.nan)
    level_arr = np.full(n_outputs, -1, dtype=int)
    confident = np.zeros(n_outputs, dtype=bool)

    for i in range(len(sigmas) - 1):
        current, following = sigmas[i], sigmas[i + 1]
        if current is None or following is None:
            continue
        unresolved = ~confident
        # `following` can be exactly 0 for a component with no measurable
        # noise at this order (e.g. a constant/fixed value at every rung) --
        # the `following > 0` mask already excludes that case from `found`,
        # but the division below would otherwise still warn (0/0 -> NaN)
        # before masking discards it.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = current / following
        found = unresolved & (following > 0) & (ratio < plateau_ratio)
        sigma[found] = (current[found] + following[found]) / 2
        level_arr[found] = i + 1
        confident[found] = True

    # No clear plateau found for some components (e.g. the function is
    # smoother than h0 can resolve within n_points levels, or dominated by
    # a non-smooth feature). Fall back to the deepest usable level for
    # those, flagged as low-confidence -- this is a conservative choice
    # biased towards *underestimating* the noise floor for genuinely
    # smooth functions (harmless: callers must not treat an unconfident
    # estimate as a hard guarantee).
    for s in reversed(sigmas):
        if s is None:
            continue
        unresolved = np.isnan(sigma)
        sigma[unresolved] = s[unresolved]

    # Anything still unset (no usable levels at all for that component)
    # falls back to machine epsilon.
    sigma[np.isnan(sigma)] = np.finfo(float).eps

    if n_outputs == 1:
        return NoiseFloor(
            sigma=float(sigma[0]),
            level=(int(level_arr[0]) if confident[0] else None),
            confident=bool(confident[0]),
            sigmas=[None if s is None else float(s[0]) for s in sigmas],
        )
    return NoiseFloor(
        sigma=sigma,
        level=level_arr,
        confident=confident,
        sigmas=sigmas,
    )


def estimate_noise_floor(
    function: Type.FUNCTION,
    point: Type.POINT,
    direction: Type.DIRECTION,
    h0: float | None = None,
    n_points: int = 15,
    plateau_ratio: float = 3.0,
    executor: Executor | None = None,
    bounds: Type.BOUNDS | None = None,
) -> NoiseFloor:
    """Estimate a function's per-evaluation noise level along a direction.

    Evaluates ``n_points`` equally spaced points centered on ``point`` and
    builds the finite-difference table. For i.i.d. per-evaluation noise with
    standard deviation ``sigma``, the ``k``-th order forward difference has
    standard deviation ``sqrt(C(2k, k)) * sigma``, so ``sigma_level(k) =
    sqrt(mean(row_k**2) / C(2k, k))`` is an unbiased noise estimate *once
    the smooth part of the function has been differenced away*. Below that
    order, the smooth (Taylor) part of the function dominates the row and
    ``sigma_level(k)`` keeps shrinking rapidly (by roughly a factor of
    ``h`` per order); once noise dominates, differencing further barely
    changes ``sigma_level(k)`` any more (it plateaus). This looks for the
    lowest order at which that rapid shrinkage stops -- i.e. the first
    ``k`` with ``sigma_level(k) / sigma_level(k+1) < plateau_ratio`` -- and
    reports the plateau value as the noise floor.

    :param function: The blackbox function.
    :param point: The point to probe around.
    :param direction: The direction to probe along.
    :param h0: The spacing between probe points. Defaults to 1% of the
        point's magnitude along ``direction`` (falling back to an
        absolute step of ``0.01`` near the origin).
    :param n_points: Number of equally spaced probe points. More points
        give more headroom for the plateau to appear within resolvable
        orders, at the cost of more function evaluations.
    :param plateau_ratio: How close two consecutive orders' estimates
        must be (``sigma_level(k) / sigma_level(k+1) < plateau_ratio``)
        to be treated as "the shrinkage has stopped."
    :param executor: How to dispatch the ``n_points`` probe evaluations
        -- e.g. :class:`fiddy.executor.JoblibExecutor` to run them in
        parallel. Defaults to :class:`fiddy.executor.SequentialExecutor`.
        All ``n_points`` probe points are decided upfront and dispatched
        as a single batch, so switching executors changes wall-clock time
        only, never the result.
    :param bounds: Optional per-parameter valid domain; forwarded to
        :func:`fiddy.step_size.clamp_step_to_bounds`, applied to the
        probe's outermost point -- every interior probe point is then
        automatically safe too, since it sits strictly closer to `point`.
        `None` (the default) disables clamping entirely.
    :return: The estimated noise floor.
    """
    if executor is None:
        executor = SequentialExecutor()
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    probe_points = _noise_floor_probe_points(
        point, direction, h0, n_points, bounds
    )
    values = np.array(
        [np.asarray(v) for v in executor(function, probe_points)]
    )
    # Keep every output component: a multi-output (fiddy.output-flattened)
    # function's noise floor is estimated component-wise, from the very
    # same probe batch -- no extra evaluations needed to cover every
    # bundled output at once.
    values = values.reshape(n_points, -1).astype(float)
    return _analyze_noise_table(values, plateau_ratio)


def noise_floor_is_confident(noise: NoiseFloor) -> bool:
    """Whether a (possibly multi-output) :class:`NoiseFloor` is usable
    as-is, or whether it looks like a *degenerate* estimate that should
    not be trusted -- used to drive the `"auto"` noise-floor-strategy
    escalation in :mod:`fiddy.estimate` (see
    :func:`fiddy.estimate.estimate_gradient`'s `noise_floor_strategy`).

    Checks both `noise.confident` (the plateau-detection heuristic's own
    signal) and `noise.sigma` being nonzero: a bounds-clamped probe step
    crushed down to (numerically) zero can make every probe point
    evaluate to the same value, which the plateau-detection heuristic
    can misread as a confident zero-noise plateau rather than "the probe
    itself was too small to measure anything" -- found via a real,
    bounds-constrained PEtab model (see
    :func:`fiddy.noise.default_probe_direction`'s own docstring for the
    underlying shared-probe cross-contamination this guards against).

    :param noise: The noise floor to check.
    :return: `True` iff every output component is confident and has a
        strictly positive `sigma`.
    """
    confident = bool(np.all(noise.confident))
    nonzero = bool(np.all(np.atleast_1d(noise.sigma) > 0))
    return confident and nonzero


def default_probe_direction(point: Type.POINT) -> np.ndarray:
    """A generic direction for a single, shared, per-point noise estimate.

    Estimating a noise floor along every parameter direction separately
    would multiply :func:`estimate_noise_floor`'s evaluation cost by the
    number of parameters -- exactly the kind of per-parameter blowup fiddy
    is meant to avoid. Instead, one noise floor is estimated along a
    single generic direction (perturbing every parameter component at
    once) and reused for every parameter's own step ladder via
    :func:`fiddy.step_size.build_step_ladder`.

    This is a deliberate few-evaluations/precision trade-off, not a free
    lunch: real noise floors can vary by many orders of magnitude between
    directions/parameters of the same model (observed empirically on a
    real ODE-based likelihood: ~1e-9 for most parameters vs. ~1e-14 for
    others). Callers that need per-direction precision should call
    :func:`estimate_noise_floor` directly with their own direction
    instead.

    A point whose components span many orders of magnitude (e.g. some
    PEtab models' free parameters on their unscaled/linear
    parameterization, spanning ~1e-5 to ~1e5) can still defeat this
    direction: the shared probe step is sized to the largest components,
    then applied identically to the smallest ones too, perturbing them
    far outside any sane range. A per-component-magnitude-scaled
    direction was tried as a fix and reverted -- it traded this failure
    for a different one (its probe became dominated by whichever
    parameter happens to be largest, which for some models has enough
    real curvature to fool the plateau-detection heuristic into reading
    it as noise). Root-caused: both failure modes are really about a
    finite-difference step leaving a parameter's *known, bounded* valid
    domain -- the actual fix is bounds-aware step clamping (an optional
    per-parameter valid range, never stepped outside of -- see the
    `bounds` parameter of :func:`estimate_noise_floor`,
    :func:`fiddy.step_size.build_step_ladder`, and the public
    `fiddy.check_gradient`/`fiddy.check_jacobian` entry points), not a
    cleverer probe direction.

    :param point: The point a probe direction is needed for (only its
        dimensionality is used).
    :return: A normalized all-ones vector of the same dimensionality as
        `point`.
    """
    point = np.asarray(point, dtype=float)
    ones = np.ones_like(point)
    norm = np.linalg.norm(ones)
    return ones / norm if norm > 0 else ones


def estimate_model_noise_floor(
    function: Type.FUNCTION, point: Type.POINT, **kwargs
) -> NoiseFloor:
    """Estimate one noise floor for a point, shared across all directions.

    Convenience wrapper around :func:`estimate_noise_floor` using
    :func:`default_probe_direction`; see both for the rationale and
    trade-off.

    :param function: The blackbox function.
    :param point: The point to probe around.
    :param kwargs: Forwarded to :func:`estimate_noise_floor` (``h0``,
        ``n_points``, ``plateau_ratio``, ``executor``, ``bounds``).
    :return: The estimated noise floor.
    """
    direction = default_probe_direction(point)
    return estimate_noise_floor(function, point, direction, **kwargs)
