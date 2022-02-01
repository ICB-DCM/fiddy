from functools import partial
from pathlib import Path
import pytest
import time
from typing import Dict, Iterable, List

import finite_difference_methods as fdm
from finite_difference_methods import CachedFunction
import numpy as np
import sympy as sp


def test_cache():
    point = np.array([[1,2],[3,4]])
    
    def function_uncached(array: np.ndarray):
        time.sleep(0.01)
        return array.flatten().sum()

    function_cached_disk = CachedFunction(function_uncached)
    function_cached_ram = CachedFunction(function_uncached, ram_cache=True)

    n_repeats = int(1e3)
   
    time_uncached = 0
    for i in range(n_repeats):
        time_start = time.time()
        function_uncached(point)
        time_uncached += time.time() - time_start

    time_cached_disk = 0
    for i in range(n_repeats):
        time_start = time.time()
        function_cached_disk(point)
        time_cached_disk += time.time() - time_start

    time_cached_ram = 0
    for i in range(n_repeats):
        time_start = time.time()
        function_cached_ram(point)
        time_cached_ram += time.time() - time_start

    function_cached_disk.delete_cache()
    function_cached_ram.delete_cache()

    assert time_uncached > 10*time_cached_disk
    assert time_uncached > 10*time_cached_ram
    assert time_cached_disk > time_cached_ram
