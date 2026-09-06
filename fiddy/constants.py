"""Shared type-annotation aliases used throughout the package."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["Type"]


class Type:
    """Type annotation variables."""

    SCALAR = np.float64
    ARRAY = NDArray[SCALAR]
    DIRECTION = ARRAY
    POINT = ARRAY
    #: The flat, canonical form returned by :meth:`fiddy.Function.__call__`.
    FUNCTION_OUTPUT = ARRAY
    #: What a *raw*, user-supplied function may return before it is wrapped
    #: by :class:`fiddy.Function`: either a plain array, or a dict of named
    #: arrays (e.g. ``{"x": ..., "y": ..., "llh": ...}``), which gets
    #: flattened into `FUNCTION_OUTPUT` -- see :mod:`fiddy.output`.
    RAW_FUNCTION_OUTPUT = ARRAY | dict[str, ARRAY]
    FUNCTION = Callable[[POINT], FUNCTION_OUTPUT]
    #: Anything :func:`numpy.random.default_rng` accepts to seed a fresh
    #: `Generator`, per `SPEC 7
    #: <https://scientific-python.org/specs/spec-0007/>`_.
    SEED_LIKE = int | np.integer | Sequence[int] | np.random.SeedSequence
    #: An already-constructed random generator, per SPEC 7.
    RNG_LIKE = np.random.Generator | np.random.BitGenerator
    #: A per-parameter valid domain, ``(lower, upper)``, each the same
    #: shape as `POINT` -- following scipy's `optimize.approx_derivative`
    #: convention: ``-inf``/``inf`` marks an unbounded component. See
    #: :func:`fiddy.step_size.clamp_step_to_bounds`.
    BOUNDS = tuple[ARRAY, ARRAY]
