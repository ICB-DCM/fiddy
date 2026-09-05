"""Executor abstraction for batched (optionally parallel) function evaluation.

Every phase of the engine (:mod:`fiddy.noise`, :mod:`fiddy.estimate`)
that needs several function evaluations decides the *whole batch* of
points it needs upfront, then dispatches them through one `Executor`
call -- this is what makes parallelization possible without changing any
of the numerical logic: the batch-then-analyze architecture (see the
:mod:`fiddy.extrapolation`/:mod:`fiddy.step_size` module docstrings,
following the DERIVEST/numdifftools/`pnd` recipe -- ``Derrico2006derivest``/
``Pnd`` in ``doc/references.bib``) already produces a fixed, independent
set of evaluations per phase, so dispatching them concurrently changes
only wall-clock time, never results.

Two executors are provided: :class:`SequentialExecutor` (the default,
plain a Python loop) and :class:`JoblibExecutor` (using `joblib`, already
a core fiddy dependency via :class:`fiddy.function.CachedFunction`). Any
other callable with the same ``(function, points) -> list`` signature can
be passed instead -- e.g. one backed by MPI or a cluster scheduler for
larger-scale use -- without fiddy needing to depend on it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .constants import Type

__all__ = ["Executor", "SequentialExecutor", "JoblibExecutor"]

Executor = Callable[[Type.FUNCTION, Sequence[Any]], list]
"""An executor is any callable ``executor(function, points) -> list`` that
evaluates ``function`` at every point in ``points`` and returns the
results in the same order -- sequentially, in parallel, or however else it
likes, as long as the order is preserved (callers rely on it to line up
results with the points/step sizes that produced them)."""


class SequentialExecutor:
    """Evaluate every point one at a time, in order. The default executor."""

    def __call__(self, function: Type.FUNCTION, points: Sequence[Any]) -> list:
        return [function(point) for point in points]


class JoblibExecutor:
    """Evaluate points in parallel using `joblib.Parallel`.

    Uses joblib's own default backend (``"loky"``, process-based), not
    ``"threading"`` -- **deliberately**, based on a real correctness
    failure, not just a style preference. Validated against a real,
    stateful blackbox function wrapping an external simulator: with
    `backend="threading"`, several directions' results silently *differed*
    from the sequential-executor result, even though wall-clock was
    competitive with sequential (unlike `"loky"`, which was ~20x slower
    for this small/fast case due to per-worker process-startup and
    repickling overhead). Root cause: the wrapped simulator held mutable
    internal state that was *not* thread-safe -- concurrent calls from
    multiple threads corrupted each other's results. This is exactly the
    failure any executor's exit criterion ("switching executors changes
    wall-clock only, never results") exists to catch, so `"loky"`
    (process-isolated, safe for any picklable function regardless of
    internal state) stays the default. Pass `backend="threading"`
    explicitly only for functions verified to be thread-safe (e.g.
    pure/stateless Python functions), where it can avoid `"loky"`'s
    per-worker startup cost.

    :param n_jobs: Forwarded to `joblib.Parallel`. `-1` (default) uses
        all available CPUs.
    :param kwargs: Additional keyword arguments forwarded to
        `joblib.Parallel` (e.g. `backend`, `prefer`).
    """

    def __init__(self, n_jobs: int = -1, **kwargs):
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def __call__(self, function: Type.FUNCTION, points: Sequence[Any]) -> list:
        import joblib

        return joblib.Parallel(n_jobs=self.n_jobs, **self.kwargs)(
            joblib.delayed(function)(point) for point in points
        )
