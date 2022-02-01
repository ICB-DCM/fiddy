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

import finite_difference_methods as fdm
from finite_difference_methods.extensions.amici import (
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


# TODO switch to lotka-volterra
@pytest.mark.parametrize("problem_generator", [simple, lotka_volterra])
def test_simulate_petab_to_functions(problem_generator):
    petab_problem, point = problem_generator()
    #petab_problem = petab_problem_fixture
    #breakpoint()
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
        fdm.gradient_check,
        function=function,
        point=point,
        gradient=gradient,
        sizes=[
            1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11,
            1e-12, 1e-13,
        ],
    )

    success_forward, result_df_forward = gradient_check_partial(
        fd_gradient_method='forward',
    )
    success_backward, result_df_backward = gradient_check_partial(
        fd_gradient_method='backward',
    )
    success_central, result_df_central = gradient_check_partial(
        fd_gradient_method='central',
    )

    print(result_df_forward)
    print(result_df_backward)
    print(result_df_central)

    # All gradient checks were successful.
    assert success_forward
    assert success_backward
    assert success_central

    # The test gradient is close to the expected gradient for both parameters and all methods.
    # Only one result is returned for each dimension.
    dimensions = [i for i in range(len(point))]
    rel_tol = 1e-1
    results = [result_df_forward, result_df_backward, result_df_central]
    for result in results:
        for dimension in dimensions:

            try:
                assert math.isclose(
                    one(result.loc[result["dimension"] == dimension]["test_gradient"]),
                    expected_gradient[dimension],
                    rel_tol=rel_tol,
                )
            except:
                breakpoint()

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
                    one(
                        result_df_central
                        .loc[result_df_central["dimension"] == dimension]
                        [error]
                    )
                )
