from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, List, Tuple, Union

import pandas as pd

from . import quotient
from .constants import (
    TYPE_DIMENSION,
    TYPE_FUNCTION,
    TYPE_POINT,
    GradientCheckMethod,
)
from .step import dstep


def gradient_check(
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    gradient: TYPE_FUNCTION,
    sizes: Iterable[float] = None,
    relative_sizes = False,
    dimensions: Iterable[TYPE_DIMENSION] = None,
    stop_at_success: bool = True,
    # TODO or custom Callable
    fd_gradient_method: GradientCheckMethod = None,
    # atol: float = 1e-2,  # TODO
    # rtol: float = 1e-2,  # TODO
    check_protocol: List[Callable[[pd.DataFrame], None]] = None,
    postprocessor_protocol: List[Callable[[pd.DataFrame], None]] = None,
) -> Tuple[bool, pd.DataFrame]:
    """Manage a gradient check.

    Args:
        point:
            The point about which to check the gradient.
        function:
            The function.
        gradient:
            A function to compute the expected gradient at a point.
        sizes:
            The sizes of the steps to take.
            Defaults to `[1e-1, 1e-3, 1e-5, 1e-7, 1e-9]`.
        relative_sizes:
            If true sizes are interpreted as relative to point value,
            otherwise as absolute.
        dimensions:
            The dimensions along which to step.
            Defaults to all dimensions of the point.
        stop_at_success:
            Whether to stop gradient checks for a specific parameter
            as soon as a tolerance is satisfied for its corresponding
            gradient.
        method:
            The method by which to check the gradient.
        atol:
            The absolute tolerance. REMOVE?
        rtol:
            The relative tolerance. REMOVE?
        check_protocol:
            These methods are applied to the results, to perform the checks.
            Defaults to `default_check_protocol`.
            Methods in this protocol should set the `"success"` column to
            `True` if the check passes, and put the reason for success in the
            `"success_reason"` column.
        postprocessor_protocol:
            Similar to `check_protocol`, but applied after `check_protocol`.

    Returns:
        (1) Whether the gradient check passed, and (2) full results,
        for debugging incorrect gradients and further analysis.
    """
    # Setup, default values
    results: Iterable[Result] = []
    if dimensions is None:
        dimensions = [index for index, _ in enumerate(point)]
    if sizes is None:
        sizes = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]
    if fd_gradient_method is None:
        fd_gradient_method = "central"
    if check_protocol is None:
        check_protocol = default_check_protocol
    if postprocessor_protocol is None:
        # or? postprocessor_protocol = default_postprocessor_protocol
        postprocessor_protocol = []

    # TODO allow supplying this, optionally instead of `gradient` callable
    expected_gradient = gradient(point)

    # Create a method to approximate the gradient. Should only require a
    # step as its only argument (use as kwarg).
    # `fd_gradient_callable` should only require a step to run.
    if fd_gradient_method == "forward":
        fd_gradient_callable = partial(
            quotient.forward, function=function, point=point
        )
    elif fd_gradient_method == "backward":
        fd_gradient_callable = partial(
            quotient.backward, function=function, point=point
        )
    elif fd_gradient_method == "central":
        fd_gradient_callable = partial(
            quotient.central, function=function, point=point
        )
    elif fd_gradient_method == "hybrid":
        fd_gradient_callable = partial(
            quotient.hybrid, function=function, point=point
        )
    else:
        raise NotImplementedError(f"Method: {fd_gradient_method}")

    for size in sizes:
        for dimension in dimensions:
            step = dstep(point=point, dimension=dimension, size=size,
                         relative=relative_sizes)
            test_gradient = fd_gradient_callable(step=step)
            results.append(
                Result(
                    dimension=dimension,
                    size=size,
                    test_gradient=test_gradient,
                    expected_gradient=expected_gradient[dimension],
                    method=fd_gradient_method,
                )
            )

    results_df = pd.DataFrame(results)
    results_df["success"] = False
    results_df["success_reason"] = None
    for check in check_protocol:
        check(results_df)
    if postprocessor_protocol is not None:
        for postprocessor in postprocessor_protocol:
            postprocessor(results_df)

    # Success is true if each dimension has at least one success.
    # TODO better name than "simplify" for this
    simplified_results_df = simplify_results_df(results_df=results_df)
    success = simplified_results_df["success"].all()

    return success, results_df


# FIXME refactor to some `gradient.py` where these FD methods can be used to
#       compute gradients.
#       would result in or require a similar method
def simplify_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only one row per successful dimension, in the dataframe.

    Can be useful for debugging problematic dimensions.

    Args:
        results_df:
            The results.

    Returns:
        pd.DataFrame
            The simplified results.
    """
    dimension_result_dfs = []
    for dimension, df in results_df.groupby("dimension"):
        # If any checks were successful for this dimension, only include the
        # first successful check.
        if df["success"].any():
            # TODO someone only include "best of" successes?
            dimension_result_dfs.append(df[df["success"]].head(1))
        # Else include all checks.
        else:
            # TODO somehow only include "best of" failures?
            dimension_result_dfs.append(df)
    simplified_results_df = pd.concat(dimension_result_dfs)
    return simplified_results_df


@dataclass
class Result:
    """Information about a single finite difference gradient computation."""

    size: float
    """The size of the step taken."""
    dimension: TYPE_DIMENSION
    """The dimension along which the gradient was checked."""
    method: GradientCheckMethod
    """The method used to compute the gradient."""
    test_gradient: float
    """The (finite difference) gradient."""
    expected_gradient: float
    """The expected gradient."""


# FIXME string literals
def add_absolute_error(results_df):
    results_df["|aerr|"] = abs(
        results_df["test_gradient"] - results_df["expected_gradient"]
    )


def check_absolute_error(results_df, tolerance: float = 1e-2):
    success = results_df["|aerr|"] < tolerance
    set_success(results_df, success, reason="|aerr|")


def add_relative_error(results_df):
    if "|aerr|" not in results_df.columns:
        add_absolute_error(results_df)
    epsilon = results_df["|aerr|"].min() * 1e-10

    results_df["|rerr|"] = abs(
        results_df["|aerr|"] / (results_df["expected_gradient"] + epsilon)
    )


def check_relative_error(results_df, tolerance: float = 1e-2):
    success = results_df["|rerr|"] < tolerance
    set_success(results_df, success, reason="|rerr|")


def set_success(results_df: pd.DataFrame, success: pd.Series, reason: str):
    new_success = success & ~results_df["success"]
    results_df["success"] = results_df["success"] | new_success
    results_df["success_reason"].mask(
        new_success,
        reason,
        inplace=True,
    )


default_check_protocol = [
    # set_all_failed,
    add_absolute_error,
    add_relative_error,
    check_absolute_error,
    check_relative_error,
]


def keep_lowest_error(
    results_df,
    error: str = "|rerr|",
    inplace: bool = True,
) -> Union[None, pd.DataFrame]:
    keep_indices = []

    for dimension, df in results_df.groupby("dimension"):
        # Keep best success from each dimension.
        if df["success"].any():
            keep_index = df.loc[df["success"]][error].idxmin()
            keep_indices.append(keep_index)
        # Include all results if there is no success.
        else:
            keep_indices += df.index

    minimal_results_df = results_df.drop(
        [index for index in results_df.index if index not in keep_indices],
        inplace=inplace,
    )

    # Kept like this for readability, but simply `return minimal_results_df`
    # should be equivalent.
    if not inplace:
        return minimal_results_df


default_postprocessor_protocol = [
    keep_lowest_error,
]
