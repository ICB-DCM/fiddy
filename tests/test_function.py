import os
import time

import numpy as np

from fiddy import CachedFunction


def test_cache():
    point = np.array([[1, 2], [3, 4]])

    def function_uncached(array: np.ndarray):
        time.sleep(0.01)
        return array.flatten().sum()

    function_cached_disk = CachedFunction(function_uncached)
    function_cached_ram = CachedFunction(function_uncached, ram_cache=True)

    n_repeats = int(1e3)

    time_uncached = 0
    for _ in range(n_repeats):
        time_start = time.time()
        function_uncached(point)
        time_uncached += time.time() - time_start

    time_cached_disk = 0
    for _ in range(n_repeats):
        time_start = time.time()
        function_cached_disk(point)
        time_cached_disk += time.time() - time_start

    time_cached_ram = 0
    for _ in range(n_repeats):
        time_start = time.time()
        function_cached_ram(point)
        time_cached_ram += time.time() - time_start

    function_cached_disk.delete_cache()
    function_cached_ram.delete_cache()

    assert time_uncached > 5 * time_cached_disk
    assert time_uncached > 5 * time_cached_ram
    # Fails with GitHub Actions, possibly because disk is ram there.
    if not os.environ.get("GITHUB_ACTIONS", False):
        assert time_cached_disk > time_cached_ram
