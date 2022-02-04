from typing import Tuple

import numpy as np


def numpy_array_to_tuple(array: np.ndarray) -> Tuple:
    """Convert a NumPy array to a tuple.

    Useful when arrays should be hashable.

    Args:
        array:
            The array.

    Returns:
        The tuple.
    """
    # Iterate over the array, turning iterable
    # elements also into tuples, recursively.
    try:
        return tuple(numpy_array_to_tuple(element) for element in array)
    except TypeError:
        return array
