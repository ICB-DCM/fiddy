import abc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import Type
from .derivative import Derivative


@dataclass
class DirectionalDerivativeCheckResult:
    direction_id: str
    """The direction."""
    method_id: str
    """The method that determined whether the directional derivative is correct."""
    test: Type.DIRECTIONAL_DERIVATIVE
    """The value that was tested."""
    expectation: Type.DIRECTIONAL_DERIVATIVE
    """The expected value."""
    success: bool
    """Whether the check passed."""
    output: dict[str, Any] = None
    """Miscellaneous output from the method."""


def _worst_element(value) -> float:
    """The largest error across all outputs for a single directional derivative."""
    return float(np.max(np.atleast_1d(value)))


def _get_printable_value(value) -> str:
    """Round scalar(s) for printing."""
    array = np.atleast_1d(value)
    if array.size == 1:
        return f"{float(array.reshape(-1)[0]):.6g}"
    return np.array2string(
        array, precision=6, suppress_small=True, threshold=6
    )


def _wide_display():
    """A ``pd.option_context`` for rendering full, readably-formatted tables.

    Intended for reports that may only be seen as plain-text CI log output,
    where a truncated/wrapped table or a mix of full-precision and
    scientific-notation floats is hard to read.
    """
    return pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_rows",
        None,
        "display.float_format",
        "{:.6g}".format,
    )


@dataclass
class DerivativeCheckResult:
    method_id: str
    """The method that determined whether the directional derivative is correct."""
    directional_derivative_check_results: list[
        DirectionalDerivativeCheckResult
    ]
    """The results from checking individual directions."""
    test: Type.DERIVATIVE
    """The value that was tested."""
    expectation: Type.DERIVATIVE
    """The expected value."""
    success: bool
    """Whether the check passed."""
    atol: float = None
    """The absolute tolerance used by the check, if any."""
    rtol: float = None
    """The relative tolerance used by the check, if any."""
    derivative: Derivative = None
    """The test derivative, including whether it was computed successfully."""
    output: dict[str, Any] = None
    """Miscellaneous output from the method."""

    @property
    def df(self):
        df = pd.DataFrame(self.directional_derivative_check_results)
        # FIXME string literal
        df.set_index("direction_id", inplace=True)
        # The direction's position in the original (unsorted) direction
        # order, so patterns (e.g. "the first direction is always off") can
        # still be spotted after reports re-sort by failure magnitude.
        df.insert(0, "direction_index", range(len(df)))
        df["abs_diff"] = np.abs(df["expectation"] - df["test"])
        df["rel_diff"] = df["abs_diff"] / np.abs(df["expectation"])
        if self.atol is not None:
            df["atol_success"] = df["abs_diff"] <= self.atol
        if self.rtol is not None:
            df["rtol_success"] = df["rel_diff"] <= self.rtol
        return df

    def assert_success(self, always_print: bool = False) -> None:
        """Assert that this derivative check succeeded.

        Args:
            always_print:
                Print the summary even if the check succeeded.
        """
        if self.derivative is not None:
            derivative_df = self.derivative.df
            if not derivative_df["success"].all():
                failed_ids = list(
                    derivative_df.index[~derivative_df["success"]]
                )
                summary_columns = [
                    column
                    for column in ("value", "success", "completed")
                    if column in derivative_df.columns
                ]
                summary_df = derivative_df[summary_columns].copy()
                if "value" in summary_df.columns:
                    summary_df["value"] = summary_df["value"].map(
                        _get_printable_value
                    )
                with _wide_display():
                    details = str(summary_df)
                raise AssertionError(
                    "Derivative check aborted: failed to compute the test "
                    "derivative via finite differences for "
                    f"{len(failed_ids)}/{len(derivative_df)} direction(s): "
                    f"{failed_ids}\n\n"
                    f"{details}\n\n"
                    "(inspect `derivative.df` for full per-direction "
                    "computer/analysis details)"
                )

        if self.success and not always_print:
            return

        df = self.df
        severity = df["rel_diff"].map(_worst_element)
        df = df.loc[severity.sort_values(ascending=False).index]
        n_total = len(df)
        n_failed = int((~df["success"]).sum())
        status = "PASSED" if self.success else "FAILED"

        header = (
            f"Derivative check {status} "
            f"({n_total - n_failed}/{n_total} direction(s) passed)"
        )
        rule = "=" * len(header)
        lines = [rule, header, rule]

        if n_failed:
            failed_ids = list(df.index[~df["success"]])
            lines.append(f"Failed direction(s): {failed_ids}")
            lines.append("")
            lines.append(
                "Details (failed directions, sorted by relative "
                "difference, worst first):"
            )
            display_columns = [
                column
                for column in (
                    "direction_index",
                    "test",
                    "expectation",
                    "abs_diff",
                    "rel_diff",
                    "atol_success",
                    "rtol_success",
                    "success",
                )
                if column in df.columns
            ]
            failed_df = df.loc[failed_ids, display_columns].copy()
            for column in ("test", "expectation", "abs_diff", "rel_diff"):
                failed_df[column] = failed_df[column].map(_get_printable_value)
            with _wide_display():
                lines.append(str(failed_df))
            lines.append("")

        abs_tol = (
            f" (tolerance: {self.atol:.6g})" if self.atol is not None else ""
        )
        rel_tol = (
            f" (tolerance: {self.rtol:.6g})" if self.rtol is not None else ""
        )
        max_adiff = max(df["abs_diff"].map(_worst_element))
        max_rdiff = max(df["rel_diff"].map(_worst_element))
        lines.append(f"Maximum absolute difference: {max_adiff:.6g}{abs_tol}")
        lines.append(f"Maximum relative difference: {max_rdiff:.6g}{rel_tol}")
        lines.append(rule)
        message = "\n".join(lines)

        if not self.success:
            raise AssertionError(message)

        print(message)


class DerivativeCheck(abc.ABC):
    """Check whether a derivative is correct.

    Args:
        derivative:
            The test derivative.
        expectation:
            The expected derivative.
        point:
            The point where the test derivative was computed.
        output_indices:
            The derivative can be a multi-dimensional object that has dimensions
            associated with the multiple outputs of a function, and dimensions
            associated with the derivative of these multiple outputs with respect
            to multiple directions.
    """

    method_id: str
    """The name of the derivative check method."""

    def __init__(
        self,
        derivative: Derivative,
        expectation: Type.DERIVATIVE,
        point: Type.POINT,
    ):
        self.derivative = derivative
        self.expectation = expectation
        self.point = point

        self.output_indices = self.expectation.shape[: -len(self.point.shape)]

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    @abc.abstractmethod
    def method(self, *args, **kwargs):
        raise NotImplementedError


class NumpyIsCloseDerivativeCheck(DerivativeCheck):
    method_id = "np.isclose"

    def method(self, *args, **kwargs):
        directional_derivative_check_results = []
        for direction_index, directional_derivative in enumerate(
            self.derivative.directional_derivatives
        ):
            test_value = np.asarray(directional_derivative.value)

            expected_value = []
            for output_index in np.ndindex(self.output_indices):
                element = self.expectation[output_index][direction_index]
                expected_value.append(element)
            expected_value = np.array(expected_value).reshape(test_value.shape)

            test_result = np.isclose(
                test_value,
                expected_value,
                *args,
                **kwargs,
            )
            directional_derivative_check_result = (
                DirectionalDerivativeCheckResult(
                    direction_id=directional_derivative.id,
                    method_id=self.method_id,
                    test=test_value,
                    expectation=expected_value,
                    output={"return": test_result},
                    success=test_result.all(),
                )
            )

            directional_derivative_check_results.append(
                directional_derivative_check_result
            )

        success = all(r.success for r in directional_derivative_check_results)
        derivative_check_result = DerivativeCheckResult(
            method_id=self.method_id,
            directional_derivative_check_results=directional_derivative_check_results,
            test=self.derivative.value,
            expectation=self.expectation,
            success=success,
            # `np.isclose` defaults, in case the caller didn't override them.
            atol=kwargs.get("atol", 1e-8),
            rtol=kwargs.get("rtol", 1e-5),
            derivative=self.derivative,
        )
        return derivative_check_result
