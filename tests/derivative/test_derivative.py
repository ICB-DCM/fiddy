import pytest

from more_itertools import one
import numpy as np
import pandas as pd
from scipy.optimize import rosen, rosen_der

from fiddy import MethodId, get_derivative, methods
from fiddy.derivative import Computer
from fiddy.analysis import ApproximateCentral
from fiddy.success import Consistency


RTOL = 1e-2
ATOL = 1e-15


@pytest.mark.parametrize("point, direction, method", [
    (np.array(point), np.array(direction), method)
    for point in [
        [1, 0],
        [0, 1],
        [1, 0.5],
        [-0.5, 0.25],
    ]
    for direction in [
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0.001],
        [0.5, -0.5],
    ]
    for method in methods
])
def test_default_directional_derivative_results(point, direction, method, size=1e-10):
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
    expected_value = rosen_der(point).dot(direction/np.linalg.norm(direction))

    assert np.isclose(test_value, expected_value, rtol=RTOL, atol=ATOL)


def test_get_derivative():
    point = np.array([1,0,0])
    sizes = [1e-10, 1e-5]
    derivative = get_derivative(
        function=rosen,
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
    assert np.isclose(derivative.values, rosen_der(point)).all()
