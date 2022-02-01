import math
import pytest
from typing import Dict, Iterable, List

import finite_difference_methods as fdm
from more_itertools import one
import numpy as np
import sympy as sp


@pytest.fixture
def case1():
    parameter_ids = ['k1', 'k2']
    equation = sp.sympify('3 + 5 * k1 + 2 * k2^2 + k1 ^ k2')
    derivatives = [equation.diff(parameter_id) for parameter_id in parameter_ids]

    def function(
        parameters: Iterable[float],
        equation: sp.Expr = equation,
        parameter_ids: List[str] = parameter_ids,
    ) -> float:
        return equation.subs(dict(zip(parameter_ids, parameters))).evalf()

    def gradient(
        parameters: Iterable[float],
        derivatives: List[sp.Expr] = derivatives,
        parameter_ids: List[str] = parameter_ids,
    ):
        return [
            derivative.subs(dict(zip(parameter_ids, parameters))).evalf()
            for derivative in derivatives
        ]

    return {
        'function': function,
        'gradient': gradient,
        #'point': np.array([3, 4]),
        #'size': 1e-10,
    }


def test_gradient_check(case1):
    point = np.array([3,4])
    size = 1e-10
    rel_tol = 1e-1

    expected_gradient = case1['gradient'](parameters=point)

    success_forward, result_df_forward = fdm.gradient_check(
        function=case1['function'],
        point=point,
        gradient=case1['gradient'],
        sizes=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
        fd_gradient_method='forward',
    )
    success_backward, result_df_backward = fdm.gradient_check(
        function=case1['function'],
        point=point,
        gradient=case1['gradient'],
        sizes=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
        fd_gradient_method='backward',
    )
    success_central, result_df_central = fdm.gradient_check(
        function=case1['function'],
        point=point,
        gradient=case1['gradient'],
        sizes=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
        fd_gradient_method='central',
    )

    # All gradient checks were successful.
    assert success_forward
    assert success_backward
    assert success_central

    # The test gradient is close to the expected gradient for both parameters and all methods.
    # Only one result is returned for each dimension.
    dimensions = [0, 1]
    results = [result_df_forward, result_df_backward, result_df_central]
    for result in results:
        for dimension in dimensions:
            assert math.isclose(
                one(result.loc[result["dimension"] == dimension]["test_gradient"]),
                expected_gradient[dimension],
                rel_tol=rel_tol,
            )

    # Errors with central method are far lower than errors with forward or backward methods.
    results = [result_df_forward, result_df_backward]
    errors = ["|aerr|", "|rerr|"]
    for result in results:
        for dimension in dimensions:
            for error in errors:
                assert (
                    one(result.loc[result["dimension"] == dimension][error])
                    > 10*
                    one(result_df_central.loc[result_df_central["dimension"] == dimension][error])
                )
