from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from amici.petab_objective import LLH, SLLH
from amici import SensitivityOrder
import petab
from petab.C import PARAMETER_SCALE

from ...constants import TYPE_FUNCTION, TYPE_POINT
from ...function import CachedFunction


LOG_E_10 = np.log(10)


def transform_gradient_lin(gradient_value, parameter_value):
    return gradient_value


def transform_gradient_log(gradient_value, parameter_value):
    return gradient_value / parameter_value


def transform_gradient_log10(gradient_value, parameter_value):
    return gradient_value / (parameter_value * LOG_E_10)


transforms = {
    "lin": transform_gradient_lin,
    "log": transform_gradient_log,
    "log10": transform_gradient_log10,
}


def simulate_petab_to_cached_functions(
    simulate_petab: Callable[[Any], Dict[str, Any]],
    petab_problem: petab.Problem,
    parameter_ids: List[str] = None,
    cache: bool = True,
    *args,
    **kwargs,
) -> Tuple[TYPE_FUNCTION, TYPE_FUNCTION]:
    r"""Convert AMICI output to compatible gradient check functions.

    Note that all gradients are provided on linear scale. The correction from
    `'log10'` scale is automatically done.

    Args:
        simulate_petab:
            A method to simulate PEtab problems with AMICI, e.g.
            `amici.petab_objective.simulate_petab`.
        parameter_ids:
            The IDs of the parameters, in the order that parameter values will
            be supplied. Defaults to `petab_problem.parameter_df.index`.
        petab_problem:
            The PEtab problem.
        \*args, \*\*kwargs:
            Passed to `simulate_petab`.

    Returns:
        tuple
            1: A method to compute the function at a point.
            2: A method to compute the gradient at a point.
    """
    if parameter_ids is None:
        parameter_ids = list(petab_problem.parameter_df.index)
    gradient_transformations = [
        transforms[
            petab_problem.parameter_df.loc[parameter_id, PARAMETER_SCALE]
        ]
        for parameter_id in parameter_ids
    ]

    solver = kwargs.pop('solver')

    simulate_petab_partial = partial(
        simulate_petab, petab_problem=petab_problem, *args, **kwargs
    )

    def simulate_petab_full(point: TYPE_POINT, order: SensitivityOrder):
        problem_parameters = dict(zip(parameter_ids, point))
        solver.setSensitivityOrder(order)
        result = simulate_petab_partial(problem_parameters=problem_parameters,
                                        solver=solver)
        return result

    def function(point: TYPE_POINT):
        result = simulate_petab_full(point, SensitivityOrder.none)
        return result[LLH]

    def gradient(point: TYPE_POINT) -> TYPE_POINT:
        result = simulate_petab_full(point, SensitivityOrder.first)
        sllh = np.array(
            [
                gradient_transformations[parameter_index](
                    gradient_value=result[SLLH][parameter_id],
                    parameter_value=point[parameter_index],
                )
                for parameter_index, parameter_id in enumerate(parameter_ids)
            ]
        )
        return sllh

    if cache:
        function = CachedFunction(function)
        gradient = CachedFunction(gradient)

    return function, gradient
