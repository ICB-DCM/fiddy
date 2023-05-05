from functools import partial

import numpy as np
import pytest
from more_itertools import one
from scipy.optimize import rosen, rosen_der

from fiddy import MethodId, get_derivative, methods
from fiddy.analysis import ApproximateCentral
from fiddy.derivative import Computer
from fiddy.derivative_check import NumpyIsCloseDerivativeCheck
from fiddy.success import Consistency

RTOL = 1e-2
ATOL = 1e-15


def rosenbrock(input_value, output_shape):
    size = np.product(output_shape)
    values = [rosen(input_value + i * 0.01) for i in range(size)]
    output = np.array(values).reshape(output_shape)
    return output


def rosenbrock_der(input_value, output_shape):
    size = np.product(output_shape)
    input_shape = input_value.shape
    values = [rosen_der(input_value + i * 0.01) for i in range(size)]
    # The input shape is the "deepest" dimension(s), i.e. expect
    # that a single input-value-dimensional-point in the output space
    # of the derivative function is the derivative vector for the
    # output value at this point in function output space, w.r.t. all
    # parameters.
    output = np.array(values).reshape(output_shape + input_shape)
    return output


@pytest.mark.parametrize(
    "point, direction, method",
    [
        (np.array(point), np.array(direction), method)
        for point in [
            (1, 0),
            (0, 1),
            (1, 0.5),
            (-0.5, 0.25),
        ]
        for direction in [
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 0.001),
            (0.5, -0.5),
        ]
        for method in methods
    ],
)
def test_default_directional_derivative_results(
    point, direction, method, size=1e-10
):
    function = rosen
    expected_gradient = rosen_der

    computer = Computer(
        function=function,
        point=point,
        direction=direction,
        size=size,
        method=method,
    )

    test_value = one(computer.results).value
    expected_value = expected_gradient(point).dot(
        direction / np.linalg.norm(direction)
    )

    assert np.isclose(test_value, expected_value, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "point, sizes, output_shape",
    [
        (np.array(point), sizes, output_shape)
        for point in [
            (1, 0, 0),
            (0.9, 0.1, 0.2, 0.4),
        ]
        for sizes in [
            [1e-10, 1e-5],
        ]
        for output_shape in [
            (1,),
            (1, 2),
            (5, 3, 6, 2, 4),
        ]
    ],
)
def test_get_derivative(point, sizes, output_shape):
    function = partial(rosenbrock, output_shape=output_shape)
    expected_derivative_function = partial(
        rosenbrock_der, output_shape=output_shape
    )
    derivative = get_derivative(
        function=function,
        point=point,
        # FIXME default?
        sizes=[1e-10, 1e-5],
        # FIXME default?
        method_ids=[MethodId.FORWARD, MethodId.BACKWARD],
        # FIXME default?
        analysis_classes=[ApproximateCentral],
        # FIXME default? not just "True" ...
        success_checker=Consistency(),
    )
    expected_value = expected_derivative_function(point)

    check = NumpyIsCloseDerivativeCheck(
        derivative=derivative,
        expectation=expected_value,
        point=point,
    )
    result = check(rtol=1e-2)
    assert result.success


def test_get_derivative_relative():
    point = np.array((3, 4, 0))
    size = 1e-1
    output_shape = (1,)

    direction = np.array([1, 0, 0])

    success_checker = Consistency(atol=1e-2)

    function = partial(rosenbrock, output_shape=output_shape)

    # Expected finite difference derivatives
    f_0 = function(point)
    f_a = function(point + direction * size)
    f_r = function(
        point + point * direction * size
    )  # cardinal direction, simplifies to this, but usually need dot product

    g_a = (f_a - f_0) / size
    g_r = (f_r - f_0) / (
        point * direction * size
    ).sum()  # cardinal direction, simplifies to this, but usually need dot product

    # Fiddy finite difference derivatives
    kwargs = {
        "function": function,
        "point": point,
        "sizes": [size],
        "method_ids": [MethodId.FORWARD],
        "directions": [direction],
        "success_checker": success_checker,
    }
    fiddy_r = float(
        np.squeeze(get_derivative(**kwargs, relative_sizes=True).value)
    )
    fiddy_a = float(
        np.squeeze(get_derivative(**kwargs, relative_sizes=False).value)
    )

    # Relative step sizes work
    assert np.isclose(fiddy_r, g_r)
    assert np.isclose(fiddy_a, g_a)

    # Same thing, now with non-cardinal direction
    def function(x):
        return (x[0] - 2) ** 2 + (x[1] + 3) ** 2

    point = np.array([3, 4])
    direction = np.array([1, 1])
    unit_direction = direction / np.linalg.norm(direction)
    kwargs["function"] = function
    kwargs["directions"] = [direction]
    kwargs["point"] = point

    size_r = size * np.dot(point, unit_direction)

    f_0 = function(point)
    f_a = function(point + unit_direction * size)
    f_r = function(point + unit_direction * size_r)

    g_a = (f_a - f_0) / size
    g_r = (f_r - f_0) / size_r

    fiddy_r = float(
        np.squeeze(get_derivative(**kwargs, relative_sizes=True).value)
    )
    fiddy_a = float(
        np.squeeze(get_derivative(**kwargs, relative_sizes=False).value)
    )

    assert np.isclose(fiddy_r, g_r)
    assert np.isclose(fiddy_a, g_a)
