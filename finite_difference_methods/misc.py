import numpy as np


def numpy_array_to_tuple(array: np.ndarray):
    """Convert a NumPy array to a tuple.

    Useful when arrays should be hashable.

    Args:
        array:
            The array.

    Returns:
    tuple
        The tuple.
    """
    try:
        return tuple(
            numpy_array_to_tuple(element)
            for element in array
        )
    except TypeError:
        return array
