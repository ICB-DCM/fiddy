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

__all__ = ["build_step_ladder"]


def build_step_ladder(
    point: Type.POINT,
    direction: Type.DIRECTION,
    noise_floor: float,
    n_rungs: int = 8,
    step_ratio: float = 2.0,
    max_relative_step: float = 1.0,
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
    return h0 / step_ratio ** np.arange(n_rungs)
