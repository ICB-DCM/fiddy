"""Tests for petab_objective.py."""

from functools import partial
from pathlib import Path

import amici
import amici.petab_import
import amici.petab_objective
import math
from more_itertools import one
import numpy as np
import petab
import pytest

import fiddy
from fiddy.extensions.amici import (
    simulate_petab_to_cached_functions,
)


# Absolute and relative tolerances for finite difference gradient checks.
ATOL: float = 1e-3
RTOL: float = 1e-3


def lotka_volterra() -> petab.Problem:
    petab_problem = petab.Problem.from_yaml(str(
        Path(__file__).parent
        / 'petab_test_problems'
        / 'lotka_volterra'
        / 'petab'
        / 'problem.yaml'
    ))
    point = np.array([2, 3])
    return petab_problem, point


def simple() -> petab.Problem:
    petab_problem = petab.Problem.from_yaml(str(
        Path(__file__).parent
        / 'petab_test_problems'
        / 'simple'
        / 'petab'
        / 'problem.yaml'
    ))
    point = np.array([1])
    return petab_problem, point


@pytest.mark.parametrize("problem_generator", [simple, lotka_volterra])
def test_simulate_petab_to_functions(problem_generator):
    petab_problem, point = problem_generator()
    amici_model = amici.petab_import.import_petab_problem(petab_problem)
    amici_solver = amici_model.getSolver()

    amici_solver.setSensitivityOrder(amici.SensitivityOrder_first)

    function, gradient = simulate_petab_to_cached_functions(
        simulate_petab=amici.petab_objective.simulate_petab,
        parameter_ids=petab_problem.parameter_df.index,
        petab_problem=petab_problem,
        amici_model=amici_model,
        solver=amici_solver,
    )

    expected_gradient = gradient(point)

    gradient_check_partial = partial(
        fiddy.gradient_check,
        function=function,
        point=point,
        gradient=gradient,
        sizes=[
            1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11,
            1e-12, 1e-13,
        ],
    )

    success_forward, results_df_forward = gradient_check_partial(
        fd_gradient_method='forward',
    )
    success_backward, results_df_backward = gradient_check_partial(
        fd_gradient_method='backward',
    )
    success_central, results_df_central = gradient_check_partial(
        fd_gradient_method='central',
    )

    # All gradient checks were successful.
    assert success_forward
    assert success_backward
    assert success_central

    # The test gradient is close to the expected gradient for both parameters and all methods.
    # Only one result is returned for each dimension.
    dimensions = [i for i in range(len(point))]
    rel_tol = 1e-1
    results_dfs = [
        results_df_forward,
        results_df_backward,
        results_df_central,
    ]
    for results_df in results_dfs:
        for dimension in dimensions:
            best_test_index = (results_df.loc[results_df["dimension"] == dimension, "test_gradient"] - expected_gradient[dimension]).abs().idxmin()
            best_test_gradient = results_df.iloc[best_test_index]["test_gradient"]
            assert math.isclose(
                best_test_gradient,
                expected_gradient[dimension],
                rel_tol=rel_tol,
            )

    # Errors with central method are far lower than errors with forward or backward methods.
    errors = ["|aerr|", "|rerr|"]
    results_dfs = [
        results_df_forward,
        results_df_backward,
    ]
    for results_df in results_dfs:
        for dimension in dimensions:
            # Sufficent for this test to pick a single size, even though different sizes may be better for different errors.
            # So, test at the minimum absolute error.
            best_test_index = (results_df.loc[results_df["dimension"] == dimension, "test_gradient"] - expected_gradient[dimension]).abs().idxmin()
            best_test_index_central = (results_df_central.loc[results_df_central["dimension"] == dimension, "test_gradient"] - expected_gradient[dimension]).abs().idxmin()
            for error in errors:
                assert (
                    results_df.iloc[best_test_index][error]
                    > 5*
                    results_df_central.iloc[best_test_index_central][error]
                )
