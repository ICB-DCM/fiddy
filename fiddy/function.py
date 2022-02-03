from pathlib import Path
import shutil

import joblib

from .constants import TYPE_FUNCTION, TYPE_POINT, TYPE_OUTPUT


default_memory_kwargs = {
    'location': 'cache_fiddy',
    'verbose': 0,
}
ram_cache_parent_path = Path("/dev/shm")


class Function:
    """Wrapper for functions."""
    def __init__(
        self,
        function: TYPE_FUNCTION,
    ):
        """Construct a cached function.

        `kwargs` is passed on to `joblib.Memory`.

        Args:
            ram_cache:
                Whether to cache in RAM. If `False`, disk is used instead.
        """
        self.function = function

    def __call__(self, point: TYPE_POINT) -> TYPE_OUTPUT:
        return self.function(point)


class CachedFunction(Function):
    """Wrapper for functions to enable caching.

    Cached data may persist, but can be removed by calling
    `CachedFunction.delete_cache()`.

    Attributes:
        function:
            The function.
        ram_cache_path:
            The path to the RAM cache, if requested.
    """

    def __init__(
        self,
        function: TYPE_FUNCTION,
        ram_cache: bool = False,
        **kwargs,
    ):
        """Construct a cached function.

        `kwargs` is passed on to `joblib.Memory`.

        Args:
            ram_cache:
                Whether to cache in RAM. If `False`, disk is used instead.
        """
        self.cache_path = \
            kwargs.get('location', default_memory_kwargs['location'])
        if ram_cache:
            if 'location' in kwargs:
                raise ValueError(
                    "Do not supply a location when using `ram_cache`."
                )
            if not ram_cache_parent_path.is_dir():
                raise FileNotFoundError(
                    "The standard Linux shared memory location '/dev/shm' "
                    "does not exist."
                )
            self.cache_path = \
                ram_cache_parent_path / default_memory_kwargs['location']
        self.cache_path = Path(self.cache_path).resolve()
        kwargs['location'] = str(self.cache_path)

        memory = joblib.Memory(**{**default_memory_kwargs, **kwargs})
        self.function = memory.cache(function)

    def delete_cache(self):
        shutil.rmtree(self.cache_path)
