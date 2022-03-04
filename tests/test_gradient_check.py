import math
import pytest
from typing import Iterable, List

import fiddy
from fiddy.gradient_check import simplify_results_df
from more_itertools import one
import numpy as np
import sympy as sp


@pytest.fixture
def case1():
    parameter_ids = ["k1", "k2"]
    equation = sp.sympify("3 + 5 * k1 + 2 * k2^2 + k1 ^ k2")
    derivatives = [
        equation.diff(parameter_id) for parameter_id in parameter_ids
    ]

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
        "function": function,
        "gradient": gradient,
        # 'point': np.array([3, 4]),
        # 'size': 1e-10,
    }


def test_gradient_check(case1):
    point = np.array([3, 4])
    rel_tol = 1e-1

    sizes = [
        1e-1,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
        1e-10,
        1e-11,
        1e-12,
        1e-13,
    ]

    expected_gradient = case1["gradient"](parameters=point)

    success_forward, results_df_forward = fiddy.gradient_check(
        function=case1["function"],
        point=point,
        gradient=case1["gradient"],
        sizes=sizes,
        fd_gradient_method="forward",
    )
    success_backward, results_df_backward = fiddy.gradient_check(
        function=case1["function"],
        point=point,
        gradient=case1["gradient"],
        sizes=sizes,
        fd_gradient_method="backward",
    )
    success_central, results_df_central = fiddy.gradient_check(
        function=case1["function"],
        point=point,
        gradient=case1["gradient"],
        sizes=sizes,
        fd_gradient_method="central",
    )  
    success_hybrid, results_df_central = fiddy.gradient_check(
        function=case1["function"],
        point=point,
        gradient=case1["gradient"],
        sizes=sizes,
        fd_gradient_method="hybrid",
    )
 
    # All gradient checks were successful.
    assert success_forward
    assert success_backward
    assert success_central 
    # Test should intentionally fail, after implementation test should succeed
    assert success_hybrid

    simplified_results_df_forward = simplify_results_df(results_df_forward)
    simplified_results_df_backward = simplify_results_df(results_df_backward)
    simplified_results_df_central = simplify_results_df(results_df_central)

    # The test gradient is close to the expected gradient for both parameters
    # and all methods.
    # Only one result is returned for each dimension.
    dimensions = [0, 1]
    simplified_results_dfs = [
        simplified_results_df_forward,
        simplified_results_df_backward,
        simplified_results_df_central,
    ]
    for simplified_results_df in simplified_results_dfs:
        for dimension in dimensions:
            assert math.isclose(
                one(
                    simplified_results_df.loc[
                        simplified_results_df["dimension"] == dimension
                    ]["test_gradient"]
                ),
                expected_gradient[dimension],
                rel_tol=rel_tol,
            )

    # Errors with central method are far lower than errors with forward or
    # backward methods.
    simplified_results_dfs = [
        simplified_results_df_forward,
        simplified_results_df_backward,
    ]
    errors = ["|aerr|", "|rerr|"]
    for simplified_results_df in simplified_results_dfs:
        for dimension in dimensions:
            for error in errors:
                assert one(
                    simplified_results_df.loc[
                        simplified_results_df["dimension"] == dimension
                    ][error]
                ) > 10 * one(
                    simplified_results_df_central.loc[
                        simplified_results_df_central["dimension"] == dimension
                    ][error]
                )
