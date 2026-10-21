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
import warnings
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
    extra evaluation cost. Estimated values are floored at the rounding
    level (see :func:`estimate_noise_floor`), so `sigma` can exceed
    every entry of `sigmas`."""
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
    an ``(n_outputs,)`` array for multi-output. If `one_sided` is set,
    `level`, `confident` and `sigmas` refer to the half of the probe
    points the estimate was taken from. The half has fewer orders; the
    missing entries at the end are ``None`` (``NaN`` for multi-output)."""
    one_sided: bool | np.ndarray = False
    """Whether `sigma` was estimated from only one half of the probe
    points, because the estimate from all points was much larger, e.g. due
    to a jump or kink in the other half or at the center (see
    :func:`estimate_noise_floor`). An ``(n_outputs,)``
    bool array for multi-output."""


_ROUNDING_FLOOR_FACTOR = 1.0
"""Lower bound on the noise floor, in units of ``eps * median|f|`` over the
probe points (see `_apply_rounding_floor`)."""

_DEFAULT_RELATIVE_SPACING = 1e-4
"""Default probe spacing relative to the point's magnitude along the probe
direction (see `_noise_floor_probe_points`)."""

_SPLIT_RATIO = 10.0
"""Factor by which the estimate from all probe values must exceed the one
from a half to use the latter (see `_analyze_noise_table`)."""

_MIN_HALF_POINTS = 5
"""Minimum number of points per half for the split analysis."""


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
    :param h0: The spacing between probe points, or `None` for the default
        (see :func:`estimate_noise_floor`).
    :param n_points: Number of equally spaced probe points.
    :param bounds: Optional per-parameter valid domain; see
        :func:`fiddy.step_size.clamp_step_to_bounds`.
    :return: The `n_points` probe points, in order.
    """
    if h0 is None:
        h0 = (
            max(abs(float(np.dot(point, direction))), 1.0)
            * _DEFAULT_RELATIVE_SPACING
        )

    max_offset = (n_points - 1) / 2 * h0
    clamped_max_offset = clamp_step_to_bounds(
        point, direction, max_offset, bounds
    )
    if max_offset > 0:
        h0 = h0 * (clamped_max_offset / max_offset)

    offsets = (np.arange(n_points) - (n_points - 1) / 2) * h0
    points = [point + t * direction for t in offsets]
    if bounds is not None:
        # The clamped outermost points can exceed the bounds by rounding.
        points = [np.clip(p, *bounds) for p in points]
    return points


def _analyze_noise_table(
    values: np.ndarray,
    plateau_ratio: float,
    split_ratio: float = _SPLIT_RATIO,
) -> NoiseFloor:
    """Turn one direction's probe evaluations into a :class:`NoiseFloor`.

    Applies the plateau-detection criterion described in
    :func:`estimate_noise_floor` to all probe values, and to the two
    halves left and right of the center point. If the estimate from all
    values exceeds the smaller half's by more than `split_ratio`, a jump
    or kink is assumed in the other half or at the center, and the smaller
    half's estimate is used. All estimates are floored at the rounding
    level before comparing. The middle point (two for even ``n_points``) is
    excluded from both halves, so that an outlier exactly at the probed
    point (e.g., a solver evaluated exactly at an event time) affects
    neither.

    Factored out so the same analysis can run on results gathered from a
    combined, multi-direction batch dispatch (see
    :func:`_noise_floor_probe_points`).

    :param values: The probe evaluations, shape ``(n_points, n_outputs)``.
    :param plateau_ratio: See :func:`estimate_noise_floor`.
    :param split_ratio: See above.
    :return: The estimated noise floor.
    """
    n_points, n_outputs = values.shape
    sigma, level_arr, confident, sigmas = _plateau_estimate(
        values, plateau_ratio
    )
    sigma = _apply_rounding_floor(sigma, values)

    one_sided = np.zeros(n_outputs, dtype=bool)
    n_half = (n_points - 1) // 2
    if n_half >= _MIN_HALF_POINTS:
        left, right = (
            _plateau_estimate(half, plateau_ratio)
            for half in (values[:n_half], values[n_points - n_half :])
        )
        left_sigma = _apply_rounding_floor(left[0], values[:n_half])
        right_sigma = _apply_rounding_floor(
            right[0], values[n_points - n_half :]
        )
        use_right = right_sigma < left_sigma
        half_sigma = np.where(use_right, right_sigma, left_sigma)
        one_sided = sigma > split_ratio * half_sigma
        use_left = one_sided & ~use_right
        use_right &= one_sided
        sigma = np.where(one_sided, half_sigma, sigma)
        sigma = _apply_rounding_floor(sigma, values)
        for mask, (_, h_level, h_confident, h_sigmas) in (
            (use_left, left),
            (use_right, right),
        ):
            level_arr = np.where(mask, h_level, level_arr)
            confident = np.where(mask, h_confident, confident)
            sigmas = [
                full
                if full is None
                else np.where(
                    mask, h_sigmas[k] if k < len(h_sigmas) else np.nan, full
                )
                for k, full in enumerate(sigmas)
            ]

    if n_outputs == 1:
        return NoiseFloor(
            sigma=float(sigma[0]),
            level=(int(level_arr[0]) if level_arr[0] > 0 else None),
            confident=bool(confident[0]),
            sigmas=[
                None if s is None or np.isnan(s[0]) else float(s[0])
                for s in sigmas
            ],
            one_sided=bool(one_sided[0]),
        )
    return NoiseFloor(
        sigma=sigma,
        level=level_arr,
        confident=confident,
        sigmas=sigmas,
        one_sided=one_sided,
    )


def _apply_rounding_floor(sigma: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Floor noise estimates at the rounding level of the function values.

    The floor of a few ulps covers rounding errors accumulated in typical
    computations, and the sampling error of the table estimate itself,
    which otherwise makes downstream budgets platform-dependent.
    Non-finite probe values are ignored.

    :param sigma: Noise estimates, shape ``(n_outputs,)``.
    :param values: The probe evaluations, shape ``(n_points, n_outputs)``.
    :return: The floored estimates.
    """
    magnitude = np.where(np.isfinite(values), np.abs(values), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        magnitude = np.nan_to_num(np.nanmedian(magnitude, axis=0), nan=0.0)
    rounding_floor = _ROUNDING_FLOOR_FACTOR * np.finfo(float).eps * magnitude
    return np.maximum(sigma, rounding_floor)


def _plateau_estimate(
    values: np.ndarray, plateau_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Plateau-detection noise estimate from equally spaced values.

    See :func:`estimate_noise_floor` for the criterion.

    :param values: The probe evaluations, shape ``(n_points, n_outputs)``.
    :param plateau_ratio: See :func:`estimate_noise_floor`.
    :return: ``(sigma, level, confident, sigmas)``: per-component estimate
        (not floored at the rounding level), plateau level (``-1`` if
        none), whether a plateau was found, and per-level estimates.
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
    return sigma, level_arr, confident, sigmas


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

    A jump or kink within the probe span dominates every order of the table
    and would be mistaken for noise. Therefore, the same estimate is also
    computed from the ``(n_points - 1) // 2`` points on either side,
    excluding the middle point(s), if ``n_points >= 11``. If the estimate
    from all points exceeds the smaller of these by more than a factor of
    10, the smaller one is used (see :attr:`NoiseFloor.one_sided`). This
    handles a single jump or kink within the probe span. A noise level that
    changes within the probe span has the same effect, and the noise is then
    underestimated if the point is on the noisier side. The default probe
    spacing is small to make multiple jumps unlikely: in ODE models with
    events, jumps occur whenever an event time crosses an output time, which
    can happen many times within a few percent of a parameter's value.

    The estimate is never smaller than ``eps * median|f|`` over the probe
    points, i.e. about two ulps of the function values. This conservative
    floor covers rounding errors accumulated in typical computations and
    the sampling error of the table estimate itself, which can otherwise
    reach tens of percent and make the noise floor (and every step size
    and tolerance derived from it) platform-dependent at the rounding
    level.

    :param function: The blackbox function.
    :param point: The point to probe around.
    :param direction: The direction to probe along.
    :param h0: The spacing between probe points. Defaults to ``1e-4``
        times the point's magnitude along ``direction`` (falling back to
        an absolute step of ``1e-4`` near the origin).
    :param n_points: Number of equally spaced probe points. More points
        give more headroom for the plateau to appear within resolvable
        orders, at the cost of more function evaluations. At least 11
        are needed for the estimate from either half (see above); with
        fewer than 15, these are often not confident.
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
    signal) and `noise.sigma` being nonzero. A bounds-clamped probe step
    crushed down to (numerically) zero makes every probe point evaluate
    to the same value: no plateau is found (`confident` is false), and
    `sigma` is only the rounding floor, or zero for an all-zero output --
    "the probe itself was too small to measure anything", e.g. if `point`
    is on a bound in every component, or close to a bound in one. A
    `NoiseFloor` constructed directly can still be confident with zero
    `sigma`, hence the second check.

    :param noise: The noise floor to check.
    :return: `True` iff every output component is confident and has a
        strictly positive `sigma`.
    """
    confident = bool(np.all(noise.confident))
    nonzero = bool(np.all(np.atleast_1d(noise.sigma) > 0))
    return confident and nonzero


def default_probe_direction(
    point: Type.POINT, bounds: Type.BOUNDS | None = None
) -> np.ndarray:
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

    Components where `point` lies on (or outside) a bound are excluded
    (set to zero): a central-difference probe cannot move them at all,
    and including them would make
    :func:`fiddy.step_size.clamp_step_to_bounds` shrink the probe to
    zero for *all* components.

    :param point: The point a probe direction is needed for.
    :param bounds: Optional ``(lower, upper)`` valid domain, as for
        :func:`estimate_noise_floor`.
    :return: A normalized vector of the same dimensionality as `point`,
        with equal entries for all components not on a bound, and zero
        entries for components on a bound. All zeros if every component
        is on a bound.
    """
    point = np.asarray(point, dtype=float)
    direction = np.ones_like(point)
    if bounds is not None:
        lower, upper = bounds
        on_bound = (point <= np.asarray(lower, dtype=float)) | (
            point >= np.asarray(upper, dtype=float)
        )
        direction[on_bound] = 0.0
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction


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
    direction = default_probe_direction(point, kwargs.get("bounds"))
    return estimate_noise_floor(function, point, direction, **kwargs)


_LADDER_NOISE_MIN_RUNGS = 6
"""Minimum ladder length for `_estimate_ladder_noise` (at least 3
combinations per difference sequence)."""


def _annihilating_weights(
    step_ratio: float, powers: tuple[int, ...]
) -> np.ndarray:
    """Weights ``w`` with ``sum_j w[j] * h[k + j]**p == 0`` for every ``p`` in
    `powers`, on a geometric ladder ``h[k + 1] = h[k] / step_ratio``.

    These are the coefficients of ``prod_p (1 - step_ratio**p * z)``.
    """
    weights = np.array([1.0])
    for p in powers:
        weights = np.convolve(weights, [1.0, -(step_ratio**p)])
    return weights


def _estimate_ladder_noise(
    ladder: np.ndarray,
    f_0: np.ndarray,
    f_plus: np.ndarray,
    f_minus: np.ndarray,
) -> np.ndarray:
    """Estimate the noise level along a direction from evaluations on a
    step-size ladder.

    Uses the second differences ``s_k = f(x + h_k) - 2 f(x) + f(x - h_k)``
    and the first differences ``d_k = f(x + h_k) - f(x - h_k)``. Over a few
    consecutive rungs of a geometric ladder, fixed linear combinations
    cancel the terms of ``s_k`` that are constant, proportional to ``h`` or
    to ``h**2`` (for a smooth function: curvature; the leading terms of a
    jump or kink at `x`), and the terms of ``d_k`` that are constant,
    proportional to ``h`` (the slope) or to ``h**3``. What remains is
    noise; the noise in ``f(x)`` cancels as well. The estimate is the
    median absolute value of these combinations over all windows of
    consecutive rungs, scaled to the standard deviation of independent
    Gaussian noise. It is a median of only a few correlated values; for
    independent Gaussian noise and 8 rungs, 90% of the estimates lie within
    0.5 to 1.9 times the noise level. Features that the combinations do not
    cancel, such as a jump or kink at a distance from `x` that lies within
    the ladder, or higher-order terms at coarse rungs, inflate it.

    :param ladder: Step sizes, decreasing by a constant ratio.
    :param f_0: Function values at the point, shape ``(n_outputs,)``.
    :param f_plus: Function values at ``x + ladder[k] * direction``, shape
        ``(len(ladder), n_outputs)``.
    :param f_minus: Function values at ``x - ladder[k] * direction``, same
        shape as `f_plus`.
    :return: The estimated noise standard deviation per output component,
        shape ``(n_outputs,)``. NaN where it cannot be estimated (fewer
        than `_LADDER_NOISE_MIN_RUNGS` rungs, a ladder that is not
        geometric, or non-finite values).
    """
    ladder = np.asarray(ladder, dtype=float)
    n_rungs = len(ladder)
    f_plus = np.asarray(f_plus, dtype=float).reshape(n_rungs, -1)
    f_minus = np.asarray(f_minus, dtype=float).reshape(n_rungs, -1)
    f_0 = np.asarray(f_0, dtype=float).reshape(-1)
    no_estimate = np.full(f_plus.shape[1], np.nan)
    if n_rungs < _LADDER_NOISE_MIN_RUNGS or not np.all(ladder > 0):
        return no_estimate
    ratios = ladder[:-1] / ladder[1:]
    if not np.allclose(ratios, ratios[0], rtol=1e-8):
        return no_estimate
    finite = (
        np.isfinite(f_0)
        & np.all(np.isfinite(f_plus), axis=0)
        & np.all(np.isfinite(f_minus), axis=0)
    )

    with np.errstate(invalid="ignore", over="ignore"):
        residuals = []
        for differences, cancelled_powers in (
            (f_plus - 2 * f_0[None, :] + f_minus, (0, 1, 2)),
            (f_plus - f_minus, (0, 1, 3)),
        ):
            weights = _annihilating_weights(ratios[0], cancelled_powers)
            n = len(weights)
            combinations = np.stack(
                [
                    weights @ differences[i : i + n]
                    for i in range(n_rungs - n + 1)
                ]
            )
            # Each difference carries the noise of two independent
            # evaluations.
            residuals.append(
                combinations / (np.sqrt(2) * np.linalg.norm(weights))
            )
        # Median absolute value -> standard deviation, for zero-mean
        # Gaussian noise.
        sigma = np.median(np.abs(np.concatenate(residuals)), axis=0) / 0.6745
    return np.where(finite, sigma, np.nan)
