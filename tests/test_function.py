import numpy as np
import pytest

from fiddy import CachedFunction, Function
from fiddy.function import FunctionEvaluationError


def test_function_wraps_plain_array_output():
    def raw(point):
        return point * 2

    function = Function(raw)
    point = np.array([1.0, 2.0, 3.0])

    result = function(point)

    np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])
    assert function.schema is not None
    assert not function.schema.is_structured
    np.testing.assert_array_equal(function.unbundle(result), [2.0, 4.0, 6.0])


def test_function_wraps_structured_output_with_no_extra_evaluations():
    call_count = 0

    def raw(point):
        nonlocal call_count
        call_count += 1
        return {"x": point * 2, "llh": np.array(point.sum())}

    function = Function(raw)
    point = np.array([1.0, 2.0])

    flat = function(point)

    assert call_count == 1
    restored = function.unbundle(flat)
    np.testing.assert_array_equal(restored["x"], [2.0, 4.0])
    np.testing.assert_array_equal(restored["llh"], 3.0)


def test_function_rejects_inconsistent_output_structure_across_calls():
    def raw(point):
        if point[0] < 0:
            return {"x": np.zeros(2)}
        return {"x": np.zeros(3)}

    function = Function(raw)
    function(np.array([1.0]))

    with pytest.raises(ValueError, match="different output structure"):
        function(np.array([-1.0]))


def test_cached_function_flattens_structured_output_and_caches():
    call_count = 0

    def raw(point):
        nonlocal call_count
        call_count += 1
        return {"x": point * 2, "y": point.sum(keepdims=True)}

    function = CachedFunction(raw)
    point = np.array([1.0, 2.0])

    first = function(point)
    second = function(point)

    assert call_count == 1  # second call was served from the cache
    np.testing.assert_array_equal(first, second)
    restored = function.unbundle(first)
    np.testing.assert_array_equal(restored["x"], [2.0, 4.0])
    np.testing.assert_array_equal(restored["y"], [3.0])

    function.delete_cache()


def test_function_wraps_exception_with_point_and_guidance():
    """A raw function that fails to evaluate (e.g. an ODE solver failing
    to converge at some perturbed point) must not surface as a bare,
    unattributed traceback -- the wrapped error should carry the point
    and make clear this is very likely the wrapped function's own fault,
    not fiddy's finite-difference algorithm."""

    def raw(point):
        raise RuntimeError("solver did not converge")

    function = Function(raw)

    with pytest.raises(
        FunctionEvaluationError, match="not with fiddy"
    ) as exc_info:
        function(np.array([1.0, 2.0]))

    assert "solver did not converge" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_function_rejects_non_finite_output():
    """A function silently returning NaN/inf (rather than raising) must
    also be caught here, not allowed to flow into noise/extrapolation
    statistics where it would otherwise be indistinguishable from a
    legitimate but unresolved ("noise_dominated") result."""

    def raw(point):
        return np.array([point[0], np.nan])

    function = Function(raw)

    with pytest.raises(FunctionEvaluationError, match="non-finite"):
        function(np.array([1.0, 2.0]))


def test_cached_function_disambiguates_colliding_closures():
    """Regression test for a real bug (found migrating AMICI's fiddy
    adapter): two closures produced by the same factory function share
    the same module/qualname/source, differing only in invisible
    captured state -- `joblib.Memory` (which `CachedFunction` is built
    on) keys its cache purely on `(module, qualname, source, arguments)`,
    so without disambiguation, calling both closures with the same point
    would silently return whichever closure's result was cached first."""

    def make(offset):
        def f(point):
            return point + offset

        return f

    f1 = CachedFunction(make(100))
    f2 = CachedFunction(make(999))

    np.testing.assert_array_equal(f1(np.array([1.0])), [101.0])
    # Not [101.0] (f1's cached result) -- a real, distinct computation.
    np.testing.assert_array_equal(f2(np.array([1.0])), [1000.0])

    f1.delete_cache()


def test_cached_function_still_caches_within_one_closure_instance():
    """The disambiguation fix must not defeat caching's actual point --
    repeated calls to the *same* wrapped closure still avoid
    recomputation."""
    call_count = 0

    def make():
        def f(point):
            nonlocal call_count
            call_count += 1
            return point * 2

        return f

    function = CachedFunction(make())
    point = np.array([1.0, 2.0])

    first = function(point)
    second = function(point)

    assert call_count == 1  # second call was served from the cache
    np.testing.assert_array_equal(first, second)

    function.delete_cache()


def test_disk_cache_avoids_recomputation():
    call_count = 0

    def function_uncached(array: np.ndarray):
        nonlocal call_count
        call_count += 1
        return array.flatten().sum()

    point = np.array([[1, 2], [3, 4]])
    function_cached = CachedFunction(function_uncached)

    first = function_cached(point)
    second = function_cached(point)

    assert call_count == 1  # second call was served from the cache
    np.testing.assert_array_equal(first, second)

    function_cached.delete_cache()


def test_ram_cache_avoids_recomputation():
    call_count = 0

    def function_uncached(array: np.ndarray):
        nonlocal call_count
        call_count += 1
        return array.flatten().sum()

    point = np.array([[1, 2], [3, 4]])
    function_cached = CachedFunction(function_uncached, ram_cache=True)

    first = function_cached(point)
    second = function_cached(point)

    assert call_count == 1  # second call was served from the cache
    np.testing.assert_array_equal(first, second)

    function_cached.delete_cache()


def test_delete_cache_forces_recomputation():
    call_count = 0

    def function_uncached(array: np.ndarray):
        nonlocal call_count
        call_count += 1
        return array.flatten().sum()

    point = np.array([[1, 2], [3, 4]])
    function_cached = CachedFunction(function_uncached)

    function_cached(point)
    function_cached.delete_cache()
    function_cached(point)

    assert call_count == 2  # deleting the cache forced a fresh computation

    function_cached.delete_cache()
