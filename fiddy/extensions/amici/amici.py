from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import amici
from amici.petab_objective import (
    LLH,
    SLLH,
    RDATAS,
    create_edatas,
    create_parameter_mapping,
)
import amici.petab_objective
import petab
from petab.C import PARAMETER_SCALE, LIN, LOG, LOG10

from ...constants import Type
from ...function import CachedFunction
from ...numpy import fiddy_array


LOG_E_10 = np.log(10)


def transform_gradient_lin_to_lin(gradient_value, parameter_value):
    return gradient_value


def transform_gradient_lin_to_log(gradient_value, parameter_value):
    return gradient_value * parameter_value


def transform_gradient_lin_to_log10(gradient_value, parameter_value):
    return gradient_value * (parameter_value * LOG_E_10)


transforms = {
    LIN: transform_gradient_lin_to_lin,
    LOG: transform_gradient_lin_to_log,
    LOG10: transform_gradient_lin_to_log10,
}


all_rdata_derivatives = {
    "x": "sx",
    "x0": "sx0",
    "x_ss": "sx_ss",
    "y": "sy",
    "sigmay": "ssigmay",
    "z": "sz",
    "rz": "srz",
    "sigmaz": "ssigmaz",
    "llh": "sllh",
    "sllh": "s2llh",
    "res": "sres",
}

# The dimension of the AMICI ReturnData that contains parameters.
# Should be shifted to the last dimension to be compatible with fiddy.
derivative_parameter_dimension = {
    "sx": 1,
    "sx0": 0,
    "sx_ss": 0,
    "sy": 1,
    "ssigmay": 1,
    #'sz'      : ???,
    "srz": 2,
    #'ssigmaz' : ???,
    "sllh": 0,
    "s2llh": 1,
    "sres": 1,
}


def rdata_array_transpose(array: np.ndarray, variable: str) -> Tuple[int]:
    original_parameter_dimension = derivative_parameter_dimension[variable]
    try:
        return np.moveaxis(array, original_parameter_dimension, -1)
    except:
        breakpoint()
    return np.moveaxis(array, original_parameter_dimension, -1)


default_derivatives = {
    k: v
    for k, v in all_rdata_derivatives.items()
    if v not in ["sz", "srz", "ssigmaz", "s2llh"]
}


def rdata_to_array(rdata: amici.AmiciReturnData):
    """Convert AMICI return data to fiddy output.

    Args:
        rdata:
            The AMICI return data.

    Returns:
        The converted return data.
    """
    breakpoint()


def output_to_array(output) -> Type.FUNCTION_OUTPUT:
    """Convert AMICI output to fiddy output.

    Output is expected to be from `amici.petab_objective.simulate_petab`.

    Args:
        output:
            The output.

    Returns:
        The converted output.
    """
    condition_results = [rdata_to_array(rdata) for rdata in output[RDATAS]]
    array = np.array(condition_results)
    breakpoint()


def run_amici_simulation_to_cached_functions(
    amici_model: amici.AmiciModel,
    *args,
    cache: bool = True,
    output_keys: List[str] = None,
    parameter_ids: List[str] = None,
    amici_solver: amici.AmiciSolver = None,
    amici_edata: amici.AmiciExpData = None,
    derivative_variables: List[str] = None,
    **kwargs,
):
    """Convert `amici.runAmiciSimulation` to fiddy functions.

    Args:
        derivative_variables:
            The variables that derivatives will be computed or approximated for. See the keys of `all_rdata_derivatives` for options.
        parameter_ids:
            The IDs that correspond to the values in the parameter vector that is simulated.

    tmp:
        derivative_shapes = None
        derivatives = []
        for rdata in result['rdatas']:
            rdata_derivatives = [np.array(rdata.get(derivative_key), dtype=Type.SCALAR) for derivative_key in output_keys.values()]
            if derivative_shapes is None:
                derivative_shapes = [derivative.shape for derivative in rdata_derivatives]
            derivatives.append(rdata_derivatives)

        # The shapes of output in some dimensions should be equal (not timepoints since difference conditions have different measurements)
        assert [rdata_derivatives[index].shape == derivatives[0][index].shape for rdata_derivatives in derivatives for index in range(len(rdata_derivatives))]

        # TODO (1) sum derivative values across all measurements within a condition
        # TODO (2) then sum across all conditions
        #grad_flat = np.ravel(dv).astype(Type.SCALAR)
        grad_flat = np.ravel(derivatives)

        def unravel_derivatives(raveled_derivatives: Type.DERIVATIVE, derivative_shapes=derivative_shapes) -> List[Type.DERIVATIVE]:
            head = 0
            unraveled = []
            for shape in derivative_shapes:
                length = np.product(shape)
                piece = raveled_derivatives[head:head+length].reshape(shape)
                unraveled.append(piece)
                head += length
            return unraveled

        unraveled = unravel_derivatives(raveled_derivatives=grad_flat)
        breakpoint()
        Returns:
            function, derivatives and structure
    """
    if amici_solver is None:
        amici_solver = amici_model.getSolver()
    if parameter_ids is None:
        parameter_ids = amici_model.getParameterIds()
    if amici_edata is not None:
        raise NotImplementedError(
            "Customization of parameter values inside AMICI ExpData."
        )
    chosen_derivatives = default_derivatives
    if derivative_variables is not None:
        chosen_derivatives = {
            k: all_rdata_derivatives[k] for k in derivative_variables
        }

    def run_amici_simulation(point: Type.POINT, order: amici.SensitivityOrder):
        problem_parameters = dict(zip(parameter_ids, point))
        amici_model.setParameterById(problem_parameters)
        amici_solver.setSensitivityOrder(order)
        rdata = amici.runAmiciSimulation(
            model=amici_model, solver=amici_solver, edata=amici_edata
        )
        return rdata

    def function(point: Type.POINT):
        rdata = run_amici_simulation(
            point=point, order=amici.SensitivityOrder.none
        )
        outputs = {
            variable: fiddy_array(getattr(rdata, variable))
            for variable in chosen_derivatives
        }
        rdata_flat = np.concatenate(
            [output.flat for output in outputs.values()]
        )
        return rdata_flat

    def derivative(point: Type.POINT, return_dict: bool = False):
        rdata = run_amici_simulation(
            point=point, order=amici.SensitivityOrder.first
        )
        outputs = {
            variable: rdata_array_transpose(
                array=fiddy_array(getattr(rdata, derivative_variable)),
                variable=derivative_variable,
            )
            for variable, derivative_variable in chosen_derivatives.items()
        }
        rdata_flat = np.concatenate(
            [
                output_array.reshape(-1, output_array.shape[-1])
                for output_array in outputs.values()
            ],
            axis=0,
        )
        if return_dict:
            return outputs
        return rdata_flat

    if cache:
        function = CachedFunction(function)
        derivative = CachedFunction(derivative)

    # Get structure
    dummy_point = fiddy_array(amici_model.getParameters())
    dummy_rdata = run_amici_simulation(
        point=dummy_point, order=amici.SensitivityOrder.first
    )

    structures = {
        "function": {variable: None for variable in chosen_derivatives},
        "derivative": {variable: None for variable in chosen_derivatives},
    }
    function_position = 0
    derivative_position = 0
    for variable, derivative_variable in chosen_derivatives.items():
        function_array = fiddy_array(getattr(dummy_rdata, variable))
        derivative_array = fiddy_array(
            getattr(dummy_rdata, derivative_variable)
        )
        structures["function"][variable] = (
            function_position,
            function_position + function_array.size,
            function_array.shape,
        )
        structures["derivative"][variable] = (
            derivative_position,
            derivative_position + derivative_array.size,
            derivative_array.shape,
        )
        function_position += function_array.size
        derivative_position += derivative_array.size

    return function, derivative, structures


# (start, stop, shape)
TYPE_STRUCTURE = Tuple[int, int, Tuple[int, ...]]


def flatten(arrays: Dict[str, Type.ARRAY]) -> Type.ARRAY:
    flattened_value = np.concatenate([array.flat for array in arrays.values()])
    return flattened_value


def reshape(
    array: Type.ARRAY, structure: TYPE_STRUCTURE
) -> Dict[str, Type.ARRAY]:
    reshaped_value = {
        variable: array[start:stop].reshape(shape)
        for variable, (start, stop, shape) in structure.items()
    }
    return reshaped_value


def simulate_petab_to_cached_functions(
    petab_problem: petab.Problem,
    amici_model: amici.Model,
    parameter_ids: List[str] = None,
    cache: bool = True,
    precreate_edatas: bool = True,
    precreate_parameter_mapping: bool = True,
    simulate_petab: Callable[[Any], Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Type.FUNCTION, Type.FUNCTION]:
    r"""Convert `amici.petab_objective.simulate_petab` to fiddy functions.

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
        \*\*kwargs:
            Passed to `simulate_petab`.

    Returns:
        tuple
            1: A method to compute the function at a point.
            2: A method to compute the gradient at a point.
    """
    if parameter_ids is None:
        parameter_ids = list(petab_problem.parameter_df.index)

    if simulate_petab is None:
        simulate_petab = amici.petab_objective.simulate_petab

    edatas = None
    if precreate_edatas:
        edatas = create_edatas(
            amici_model=amici_model,
            petab_problem=petab_problem,
            simulation_conditions=petab_problem.get_simulation_conditions_from_measurement_df(),
        )

    parameter_mapping = None
    if precreate_parameter_mapping:
        parameter_mapping = create_parameter_mapping(
            petab_problem=petab_problem,
            simulation_conditions=petab_problem.get_simulation_conditions_from_measurement_df(),
            scaled_parameters=kwargs.get(
                "scaled_parameters",
                (
                    signature(simulate_petab)
                    .parameters["scaled_parameters"]
                    .default
                ),
            ),
            amici_model=amici_model,
        )

    precreated_kwargs = {
        "edatas": edatas,
        "parameter_mapping": parameter_mapping,
        "petab_problem": petab_problem,
    }
    precreated_kwargs = {
        k: v for k, v in precreated_kwargs.items() if v is not None
    }

    amici_solver = kwargs.pop("solver", amici_model.getSolver())

    simulate_petab_partial = partial(
        simulate_petab,
        amici_model=amici_model,
        **precreated_kwargs,
        **kwargs,
    )

    def simulate_petab_full(point: Type.POINT, order: amici.SensitivityOrder):
        problem_parameters = dict(zip(parameter_ids, point))
        amici_solver.setSensitivityOrder(order)
        result = simulate_petab_partial(
            problem_parameters=problem_parameters,
            solver=amici_solver,
        )
        return result

    simulate_petab_full_cached = simulate_petab_full
    if cache:
        simulate_petab_full_cached = CachedFunction(simulate_petab_full)

    def function(point: Type.POINT):
        output = simulate_petab_full(point, order=amici.SensitivityOrder.none)
        result = output[LLH]
        return np.array(result)

    def derivative(point: Type.POINT) -> Type.POINT:
        result = simulate_petab_full(point, order=amici.SensitivityOrder.first)
        sllh = np.array(
            [result[SLLH][parameter_id] for parameter_id in parameter_ids]
        )
        return sllh

    if cache:
        function = CachedFunction(function)
        derivative = CachedFunction(derivative)

    return function, derivative
