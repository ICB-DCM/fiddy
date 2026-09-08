"""Step-size ladder generation from a noise floor.

Picks an initial step size from the classical central-difference
truncation/rounding-error trade-off, ``h* ~ noise_floor**(1/3)``
(``DennisSchnabel1983`` in ``doc/references.bib``; and the practical
hybrid ``max(relstep*|x|, absstep)`` scaling documented for FiniteDiff.jl,
``Finitediffjl``), then lays out a fixed geometric ladder of step sizes
around it -- a single upfront batch, not an iterative retry loop,
following the DERIVEST/numdifftools recipe (``Derrico2006derivest``).
"""

from __future__ import annotations

import numpy as np

from .constants import Type

__all__ = ["build_step_ladder", "clamp_step_to_bounds"]


def clamp_step_to_bounds(
    point: Type.POINT,
    direction: Type.DIRECTION,
    h: float,
    bounds: Type.BOUNDS | None,
) -> float:
    """Shrink a step size so a central-difference pair never leaves a
    known valid domain.

    A finite-difference step evaluated at a mathematically or physically
    invalid point is not a hypothetical concern: e.g. a PEtab model
    parameter on a ``log10`` scale requires its linear value to stay
    strictly positive for the scale conversion itself to be defined --
    perturbing it non-positive is not "possibly fine", it is an
    unconditional domain violation (``log10`` of a non-positive number).
    Other cases (e.g. an ODE solver failing to converge for a
    parameter far outside the range its model was ever validated for)
    are softer -- crossing the declared bound does not *guarantee*
    failure there -- but the same clamp is a reasonable, already-
    available safeguard for both, since a domain the modeller declared
    valid is the best information available about where evaluation is
    expected to work.

    Finds the largest ``h' <= h`` such that ``point + t * direction``
    stays within ``bounds`` for every ``t`` in ``[-h', h']`` (safe for
    both the forward and backward evaluation of a central difference),
    following scipy's own ``optimize.approx_derivative`` convention for
    `bounds`, without its one-sided-scheme fallback: if a symmetric step
    this small already cannot fit, this simply shrinks it -- potentially
    all the way to a step size that reports a direction as noise-
    dominated rather than converged -- rather than adding a second,
    asymmetric evaluation scheme purely to extract a few more
    directions' worth of confident answers. Consistent with fiddy's
    "honest uncertainty over confident wrongness" design.

    :param point: The point the step is taken from.
    :param direction: The direction the step is taken along.
    :param h: The desired step size.
    :param bounds: ``(lower, upper)``, each the same shape as `point`;
        ``-inf``/``inf`` marks an unbounded component. `None` disables
        clamping entirely (returns `h` unchanged) -- the default
        everywhere this is used, reproducing today's behavior exactly
        for callers who don't supply a domain.
    :return: The clamped step size, ``0 <= h' <= h``.
    """
    if bounds is None:
        return h
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    constrained = direction != 0
    if not np.any(constrained):
        return h

    d = direction[constrained]
    p = point[constrained]
    lo = lower[constrained]
    hi = upper[constrained]

    positive = d > 0
    # point + h*d stays in [lo, hi]; point - h*d stays in [lo, hi].
    forward_limit = np.where(positive, (hi - p) / d, (lo - p) / d)
    backward_limit = np.where(positive, (p - lo) / d, (p - hi) / d)

    return float(
        max(0.0, min(h, np.min(forward_limit), np.min(backward_limit)))
    )


def build_step_ladder(
    point: Type.POINT,
    direction: Type.DIRECTION,
    noise_floor: float,
    n_rungs: int = 8,
    step_ratio: float = 2.0,
    max_relative_step: float = 1.0,
    bounds: Type.BOUNDS | None = None,
) -> np.ndarray:
    """Build a fixed geometric ladder of central-difference step sizes.

    The initial (largest) step follows the classical central-difference
    optimum ``h* ~ noise_floor**(1/3)``, scaled by the point's magnitude
    along ``direction`` (falling back to an absolute step near the
    origin). The ladder then descends by ``step_ratio`` each rung -- one
    fixed batch, not an adaptive retry loop.

    :param point: The point the ladder is built around.
    :param direction: The direction the ladder's steps are taken along.
    :param noise_floor: The noise floor to derive the initial step from,
        e.g. from :func:`fiddy.noise.estimate_model_noise_floor`.
    :param n_rungs: Number of step sizes in the ladder.
    :param step_ratio: Ratio between consecutive rungs
        (``h_k = h_0 / step_ratio**k``).
    :param max_relative_step: Safety clamp: the largest rung never exceeds
        `max_relative_step` times the point's own magnitude along
        `direction` (the same `scale` the ``h*`` formula above is scaled
        by). ``h* ~ noise_floor**(1/3)`` implicitly assumes `noise_floor`
        is a "small" quantity; when it is not, the bare formula can
        produce a step thousands of times larger than the point itself,
        evaluating the function at nonsensical perturbed points. This is
        not hypothetical: on a real, stiff epidemiological ODE model, the
        shared, generic-direction noise probe (see
        :func:`fiddy.noise.default_probe_direction`) measured a "noise
        floor" *larger than the function's own value* at that point,
        because perturbing every parameter at once pushed the model into
        a genuinely different dynamical regime that the plateau-detection
        heuristic mistook for noise -- every direction's derivative
        estimate came back `NaN` until this clamp was added. Clamping the
        largest rung to a sane fraction of the point's own scale keeps
        the ladder local regardless of how large (or wrong) the noise
        estimate turns out to be; the default of `1.0` only ever engages
        for such pathological cases, never for the well-behaved
        ``noise_floor << 1`` regime this formula was designed for.
    :param bounds: Optional per-parameter valid domain; forwarded to
        :func:`clamp_step_to_bounds`, applied to the largest rung after
        the `max_relative_step` clamp above -- every smaller rung is then
        automatically safe too, since a domain violation only gets less
        likely as the step shrinks. `None` (the default) disables
        clamping entirely, reproducing today's exact behavior.
    :return: The step-size ladder, decreasing.
    """
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    scale = max(abs(float(np.dot(point, direction))), 1.0)
    # A component whose value is exactly constant along `direction` (e.g. one
    # bundled output in a multi-output check that just doesn't depend on the
    # perturbed parameters) has a true noise floor of exactly 0, not merely
    # small. Left unfloored, `h0` (and thus the entire ladder) collapses to
    # literal 0.0, which turns the central-difference formula downstream
    # into a 0/0 divide (found via the AMICI SBML semantic test suite).
    # Flooring at machine epsilon keeps every rung strictly positive while
    # leaving the well-behaved `noise_floor >> eps` regime untouched.
    noise_floor = max(noise_floor, np.finfo(float).eps)
    h0 = min(scale * noise_floor ** (1 / 3), max_relative_step * scale)
    h0 = clamp_step_to_bounds(point, direction, h0, bounds)
    return h0 / step_ratio ** np.arange(n_rungs)
