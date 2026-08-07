import abc
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from . import analysis, directional_derivative
from .constants import Type
from .directional_derivative import DirectionalDerivative


class Success:
    id: str = None

    def __init__(self, method: Type.SUCCESS_CHECKER = None):
        if method is not None:
            self.method = method
        self.value = None
        self.success = None
        if self.id is None:
            raise ValueError("Please set the success checker ID in its class.")

    def __call__(self, directional_derivative: DirectionalDerivative):
        self.success, self.value = self.method(
            directional_derivative=directional_derivative
        )
        return self.success, self.value

    @abc.abstractmethod
    def method(self, directional_derivative: DirectionalDerivative) -> Any:
        raise NotImplementedError


class Consistency(Success):
    """Consistency-based success checker.

    For each step size, checks whether the requested methods
    (e.g. forward/backward/central) agree with each other ("self-consistent").
    Self-consistency alone is not sufficient though: a step size can be small
    enough that all methods sample points within the target function's
    floating-point noise floor, and become correlated (affected by the same
    rounding/cancellation error) -- self-consistent, yet biased away from the
    truth. Symmetrically, a step size can also be large enough that all
    methods are biased the same way by higher-order/truncation effects.

    To guard against this, self-consistent step sizes are additionally
    required to agree with the majority of other self-consistent step sizes,
    via iterative outlier rejection (order-independent; step size magnitude
    is not used as a proxy for trustworthiness): repeatedly compute the
    median and a robust (MAD-based) spread of the current candidates, and
    drop the single worst-deviating one if it exceeds ``trend_n_sigma``
    scaled MADs from the median, until nothing looks anomalous. This only
    activates once there are at least ``min_trend_samples`` self-consistent
    step sizes; below that, there isn't enough data to estimate a spread, and
    all self-consistent step sizes are used, as before.

    Note that this is a majority-vote style method: like any check based
    purely on the agreement of the values themselves (no independent ground
    truth), it has a breakdown point of roughly 50% (a property of the
    underlying median/MAD statistics) -- if close to half (or more) of the
    self-consistent step sizes are corrupted, this check cannot reliably
    tell which subset is trustworthy. This is a fundamental limitation of
    any purely data-driven consistency check, not something this
    implementation can detect or work around; sufficient step sizes with a
    real chance of being individually trustworthy should be provided.
    """

    # FIXME string literal
    id = "consistency"
    only_at_completion: bool = True

    def __init__(
        self,
        computer_parser: Callable[
            ["directional_derivative.Computer"], float | None
        ] = None,
        analysis_parser: Callable[["analysis.Analysis"], float | None] = None,
        rtol: float = 0.2,
        atol: float = 1e-15,
        equal_nan: bool = True,
        trend_n_sigma: float = 5.0,
        min_trend_samples: int = 3,
    ):
        """Construct.

        Args:
            rtol:
                Relative tolerance for the self-consistency check of methods
                at the same step size, and for the final check of the
                blended value against the trusted per-size means.
            atol:
                Absolute tolerance, analogous to `rtol`. Also used as a
                floor for the robust spread estimate in the cross-step-size
                outlier rejection (see class docstring).
            equal_nan:
                Whether `NaN`s are considered equal in tolerance checks.
            trend_n_sigma:
                The number of scaled median-absolute-deviations a
                self-consistent step size's estimate may deviate from the
                median of the other trusted step sizes' estimates, before it
                is rejected as an outlier.
            min_trend_samples:
                The minimum number of self-consistent step sizes required
                before the cross-step-size outlier rejection is attempted.
                Below this, all self-consistent step sizes are trusted, same
                as if `trend_n_sigma`/outlier rejection did not exist.
        """
        super().__init__()
        # if computer_parser is None:
        #     computer_parser = (
        #         lambda computer, size: computer.value
        #         if computer.size == size
        #         else None
        #     )
        # self.computer_parser = computer_parser
        # if analysis_parser is None:
        #     analysis_parser = (
        #         lambda analysis: analysis.value
        #         if computer.size == size
        #         else None
        #     )
        # self.analysis_parser = analysis_parser

        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        self.trend_n_sigma = trend_n_sigma
        self.min_trend_samples = min_trend_samples

    def method(
        self, directional_derivative: DirectionalDerivative
    ) -> tuple[bool, float]:
        # FIXME string literals
        computer_results = directional_derivative.get_computer_results()
        analysis_results = directional_derivative.get_analysis_results()
        results_by_size = {}
        for result in [*computer_results, *analysis_results]:
            size = result.metadata.get("size_absolute", None)
            if size is None:
                continue
            if size not in results_by_size:
                results_by_size[size] = {}
            if result.method_id in results_by_size[size]:
                raise ValueError(
                    f"Duplicate, and possibly conflicting, results for method "
                    f'"{result.method_id}" and size "{size}".',
                )
            results_by_size[size][result.method_id] = result.value

        self_consistent_means = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Mean of empty slice", RuntimeWarning
            )
            for results in results_by_size.values():
                values = list(results.values())
                mean = np.nanmean(values, axis=0)
                is_self_consistent = np.isclose(
                    values,
                    mean,
                    rtol=self.rtol / 2,
                    atol=self.atol / 2,
                    equal_nan=self.equal_nan,
                ).all()
                if is_self_consistent:
                    self_consistent_means.append(mean)

            if not self_consistent_means:
                return False, np.nan

            trusted_means = self._reject_outliers(self_consistent_means)

            if not trusted_means:
                return False, np.nan

            value = np.nanmean(trusted_means, axis=0)

        success = (
            np.isclose(
                trusted_means,
                value,
                rtol=self.rtol,
                atol=self.atol,
                equal_nan=self.equal_nan,
            ).all()
            and not np.isnan(trusted_means).all()
        )
        return success, value

    def _reject_outliers(
        self, means: list[Type.DIRECTIONAL_DERIVATIVE]
    ) -> list[Type.DIRECTIONAL_DERIVATIVE]:
        """Iteratively reject step sizes whose estimate is an outlier.

        See the class docstring for the rationale. Order-independent: does
        not assume larger (or smaller) step sizes are inherently more
        trustworthy.

        Args:
            means:
                The per-step-size mean estimates that passed the
                within-step-size self-consistency check.

        Returns:
            The subset of `means` that are also mutually consistent with
            each other.
        """
        trusted = list(means)
        if len(trusted) < self.min_trend_samples:
            return trusted

        floor = max(self.atol / 2, np.finfo(float).tiny)
        while len(trusted) >= self.min_trend_samples:
            stacked = np.asarray(trusted, dtype=float)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN", RuntimeWarning)
                center = np.nanmedian(stacked, axis=0)
                mad = np.nanmedian(np.abs(stacked - center), axis=0)
                scale = np.maximum(mad * 1.4826, floor)
                # One badness score per candidate, reduced across all
                # output dimensions (a candidate is an outlier if it
                # deviates too much in *any* output element).
                badness = np.nanmax(
                    (np.abs(stacked - center) / scale).reshape(
                        len(trusted), -1
                    ),
                    axis=1,
                )
                worst = int(np.nanargmax(badness))
            if badness[worst] > self.trend_n_sigma:
                trusted.pop(worst)
            else:
                break
        return trusted
