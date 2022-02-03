from .constants import (
    TYPE_FUNCTION,
    TYPE_POINT,
)


def forward(
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    step: TYPE_POINT,
):
    """Compute a forward difference.

    .. _forward difference args:

    Args:
        function:
            The function.
        point:
            The point at which to compute the difference.
        step:
            The finite difference step to take away from the point.

    Returns:
        The forward difference.
    """
    return function(point + step) - function(point)


def backward(
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    step: TYPE_POINT,
):
    """Compute a backward difference.

    Arguments are documented in the
    :ref:`forward difference <forward difference args>` method.

    Returns:
        The backward difference.
    """
    return function(point) - function(point - step)


def central(
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    step: TYPE_POINT,
):
    """Compute a central difference.

    NB: The size is halved before stepping forward and backward.

    Arguments are documented in the
    :ref:`forward difference <forward difference args>` method.

    Returns:
        The central difference.
    """
    return function(point + step/2) - function(point - step/2)
