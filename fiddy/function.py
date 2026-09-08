"""Wraps a raw blackbox function: bundles/flattens its output (see
:mod:`fiddy.output`), validates that output's structure stays consistent
across calls, reports a clear error if the function itself fails, and
optionally adds joblib-based disk/RAM caching.
"""

import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np

from .constants import Type
from .output import OutputSchema, flatten_output

default_memory_kwargs = {
    "location": "cache_fiddy",
    "verbose": 0,
}
# TODO change to multiprocessing.shared_memory
ram_cache_parent_path = Path("/dev/shm")  # noqa: S108


class FunctionEvaluationError(RuntimeError):
    """The function being differentiated failed to produce a usable value
    at some point fiddy needed to evaluate it -- either it raised an
    exception, or it returned a non-finite (``NaN``/``inf``) value (e.g.
    an ODE solver failing to converge at a perturbed parameter value).

    This is (almost always) a problem with that function/model, not with
    fiddy's finite-difference algorithm: there is no meaningful way to
    estimate, or even classify as noise-dominated, a derivative from a
    point where the function itself could not produce a valid value.
    Callers hitting this should first check that the function evaluates
    successfully (and returns finite output) at the reported point,
    before treating it as a fiddy bug.
    """


class Function:
    """Wrapper for functions.

    The wrapped ``function`` may return either a plain array-like, or a
    dict of named array-likes (e.g. ``{"x": ..., "y": ..., "llh": ...}``,
    as a bundle of named simulation outputs might). Either way, calling
    this wrapper always returns a single flat array (see
    :mod:`fiddy.output`), so every named output shares one finite-
    difference sweep -- no extra function evaluations are needed to check
    several outputs at once. Use :meth:`unbundle` to split a flat result
    (e.g. a computed derivative that shares the function output's layout)
    back into the named pieces.

    :ivar function: The wrapped function.
    :ivar schema: The :class:`~fiddy.output.OutputSchema` recorded from
        the first call, or `None` before the function has been called.
    """

    def __init__(
        self,
        function: Type.FUNCTION,
    ):
        """Construct a function.

        :param function: The function to wrap.
        """
        self.function = function
        self.schema: OutputSchema | None = None

    def __call__(self, point: Type.POINT) -> Type.FUNCTION_OUTPUT:
        try:
            raw = self.function(point)
        except Exception as error:
            raise FunctionEvaluationError(
                "The function being differentiated raised "
                f"{type(error).__name__} when evaluated at point "
                f"{point!r}: {error}\n"
                "This is very likely a problem with that function (e.g. "
                "it fails to evaluate/converge at this input), not with "
                "fiddy's finite-difference algorithm -- check that the "
                "function evaluates successfully here before treating "
                "this as a fiddy bug."
            ) from error
        flat, schema = flatten_output(raw)
        if not np.all(np.isfinite(flat)):
            raise FunctionEvaluationError(
                "The function being differentiated returned a non-"
                f"finite (NaN/inf) value when evaluated at point "
                f"{point!r}.\n"
                "This is very likely a problem with that function (e.g. "
                "an ODE solver silently failing to converge at this "
                "input rather than raising), not with fiddy's finite-"
                "difference algorithm -- fiddy has no way to estimate a "
                "derivative from an invalid function value."
            )
        if self.schema is None:
            self.schema = schema
        elif (
            schema.names != self.schema.names
            or schema.shapes != self.schema.shapes
        ):
            raise ValueError(
                "The wrapped function returned a different output "
                "structure than on its first call. A blackbox function's "
                "output structure (names and shapes) must stay the same "
                "at every point evaluated during a derivative check.\n"
                f"  first call:  names={self.schema.names}, "
                f"shapes={self.schema.shapes}\n"
                f"  this call:   names={schema.names}, shapes={schema.shapes}"
            )
        return flat

    def unbundle(self, flat: Type.FUNCTION_OUTPUT):
        """Split a flat array back into named outputs, per :attr:`schema`.

        :param flat: An array with the same flat layout as this
            function's output -- typically the output itself, or a
            derivative computed from it.
        """
        if self.schema is None:
            raise RuntimeError(
                "Call the function at least once before unbundling, so "
                "its output structure is known."
            )
        return self.schema.unbundle(flat)


def _disambiguate_closure(function: Type.FUNCTION) -> Type.FUNCTION:
    """Give a closure a unique identity before handing it to
    `joblib.Memory`, so independently-created closures that share the
    same defining code (module/qualname/source) can never collide in
    the cache.

    `joblib.Memory` identifies a cached function by
    ``(module, qualname, source text, call arguments)`` -- it has no
    visibility into a closure's captured (free) variables. A factory
    function that returns a closure (e.g. one built fresh per call with
    different captured configuration) therefore produces a *new* Python
    function object every call, but one `joblib` cannot tell apart from
    any other closure returned by the same factory: two closures with
    different captured state but the same defining code, both called
    with the same explicit arguments, would silently share one cache
    entry. This was found and confirmed via a real bug: a factory
    function returning a `derivative(point)` closure whose captured
    configuration differed between two factory calls -- calling both
    with the same `point` returned the *first* call's cached result for
    the second, wrong silently rather than loudly.

    `function.__closure__` is `None` exactly when a function captures no
    enclosing-scope variables (true for every plain, top-level function,
    and for a nested `def` that happens not to close over anything) --
    exactly the functions for which `joblib`'s assumption already holds,
    so those are left untouched (including their legitimate persistence
    across process restarts). Only an actual closure gets a fresh,
    per-wrapping unique name, which means a closure's cache no longer
    persists across separate `CachedFunction(...)` wrappings (including
    across process restarts) -- an acceptable trade, since that
    persistence was never safe to rely on in the first place.

    :param function: The function to give a safe-to-cache identity.
    :return: `function`, mutated in place if it is a closure.
    """
    if function.__closure__ is not None:
        suffix = uuid.uuid4().hex
        function.__qualname__ = f"{function.__qualname__}#{suffix}"
        function.__name__ = f"{function.__name__}#{suffix}"
    return function


class CachedFunction(Function):
    """Wrapper for functions to enable caching.

    Cached data may persist, but can be removed by calling
    `CachedFunction.delete_cache()`. If the wrapped function is a closure
    (captures variables from an enclosing scope), it is automatically
    given a unique identity before caching -- `joblib.Memory` identifies
    a cached function only by ``(module, qualname, source text, call
    arguments)``, blind to a closure's captured variables, so two
    closures sharing the same defining code but different captured state
    could otherwise silently collide in the cache.

    :ivar function: The function.
    :ivar cache_path: The path to the cache (disk or RAM).
    """

    def __init__(
        self,
        function: Type.FUNCTION,
        ram_cache: bool = False,
        **kwargs,
    ):
        """Construct a cached function.

        :param function: The function to wrap and cache.
        :param ram_cache: Whether to cache in RAM. If `False`, disk is
            used instead.
        :param kwargs: Passed on to `joblib.Memory`.
        """
        self.cache_path = kwargs.get(
            "location", default_memory_kwargs["location"]
        )
        if ram_cache:
            if "location" in kwargs:
                raise ValueError(
                    "Do not supply a location when using `ram_cache`."
                )
            if not ram_cache_parent_path.is_dir():
                raise FileNotFoundError(
                    "The standard Linux shared memory location '/dev/shm' "
                    "does not exist."
                )
            self.cache_path = (
                ram_cache_parent_path / default_memory_kwargs["location"]
            )
        self.cache_path = Path(self.cache_path).resolve()
        kwargs["location"] = str(self.cache_path)

        memory = joblib.Memory(**{**default_memory_kwargs, **kwargs})
        # Caching wraps the *raw* function, before output flattening, so
        # the cache is unaffected by (and shared regardless of) whether
        # the wrapped function returns a plain array or a structured dict.
        super().__init__(memory.cache(_disambiguate_closure(function)))

    def delete_cache(self):
        shutil.rmtree(self.cache_path)
