from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from amici.petab_objective import (
    LLH,
    SLLH,
    create_edatas,
    create_parameter_mapping,
)
import petab
from petab.C import PARAMETER_SCALE, LIN, LOG, LOG10

from ...constants import TYPE_FUNCTION, TYPE_POINT
from ...function import CachedFunction


LOG_E_10 = np.log(10)


def transform_gradient_lin_to_lin(gradient_value, parameter_value):
    return gradient_value


def transform_gradient_lin_to_log(gradient_value, parameter_value):
    return gradient_value * parameter_value


def transform_gradient_lin_to_log10(gradient_value, parameter_value):
    return gradient_value * (parameter_value * LOG_E_10)


transforms = {
    LIN:   transform_gradient_lin_to_lin,
    LOG:   transform_gradient_lin_to_log,
    LOG10: transform_gradient_lin_to_log10,
}


def simulate_petab_to_cached_functions(
    simulate_petab: Callable[[Any], Dict[str, Any]],
    petab_problem: petab.Problem,
    *args,
    parameter_ids: List[str] = None,
    cache: bool = True,
    precreate_edatas: bool = True,
    precreate_parameter_mapping: bool = True,
    scaled_gradients: bool = True,
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
        cache:
            Whether to cache the function call.
        precreate_edatas:
            Whether to create the AMICI measurements object in advance, to save
            time.
        precreate_parameter_mapping:
            Whether to create the AMICI parameter mapping object in advance, to
            save time.
        scaled_gradients:
            Whether to return gradients on the scale of the parameters.
        \*args, \*\*kwargs:
            Passed to `simulate_petab`.

    Returns:
        tuple
            1: A method to compute the function at a point.
            2: A method to compute the gradient at a point.
    """
    if parameter_ids is None:
        parameter_ids = list(petab_problem.parameter_df.index)
    if scaled_gradients:
        gradient_transformations = [
            transforms[
                petab_problem.parameter_df.loc[parameter_id, PARAMETER_SCALE]
            ]
            for parameter_id in parameter_ids
        ]
    else:
        gradient_transformations = [transforms[LIN] for _ in parameter_ids]

    edatas = None
    if precreate_edatas:
        if 'amici_model' not in kwargs:
            raise ValueError(
                'Please supply the AMICI model to precreate ExpData.'
            )
        edatas = create_edatas(
            amici_model=kwargs['amici_model'],
            petab_problem=petab_problem,
            simulation_conditions=\
                petab_problem.get_simulation_conditions_from_measurement_df(),
        )

    parameter_mapping = None
    if precreate_parameter_mapping:
        if 'amici_model' not in kwargs:
            raise ValueError(
                'Please supply the AMICI model to precreate ExpData.'
            )
        parameter_mapping = create_parameter_mapping(
            petab_problem=petab_problem,
            simulation_conditions=\
                petab_problem.get_simulation_conditions_from_measurement_df(),
            scaled_parameters=kwargs.get(
                'scaled_parameters',
                (
                    signature(simulate_petab)
                    .parameters['scaled_parameters']
                    .default
                ),
            ),
            amici_model=kwargs['amici_model'],
        )

    precreated_kwargs = {
        'edatas': edatas,
        'parameter_mapping': parameter_mapping,
        'petab_problem': petab_problem,
    }
    precreated_kwargs = {
        k: v
        for k, v in precreated_kwargs.items()
        if v is not None
    }

    simulate_petab_partial = partial(
        simulate_petab,
        *args,
        **precreated_kwargs,
        **kwargs,
    )

    def simulate_petab_full(point: TYPE_POINT):
        problem_parameters = dict(zip(parameter_ids, point))
        result = simulate_petab_partial(problem_parameters=problem_parameters)
        return result

    simulate_petab_full_cached = simulate_petab_full
    if cache:
        simulate_petab_full_cached = CachedFunction(simulate_petab_full)

    def function(point: TYPE_POINT):
        result = simulate_petab_full_cached(point)
        return result[LLH]

    def gradient(point: TYPE_POINT) -> TYPE_POINT:
        result = simulate_petab_full_cached(point)
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

    return function, gradient
