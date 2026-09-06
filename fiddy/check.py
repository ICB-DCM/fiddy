"""Minimal public gradient-checking API.

Two entry points: :func:`check_gradient` -- requiring only a function, a
point, and something to check against -- ties together the whole engine
(:func:`fiddy.estimate.estimate_gradient`) and replaces per-model
tolerance tuning with a per-direction tolerance auto-derived from the
engine's own error estimate. :func:`check_jacobian` does the same for
every output component of a bundled multi-output function at once (e.g. a
model's state/observable/likelihood sensitivities together, not just its
scalar objective) -- checking N outputs this way costs no more function
evaluations than checking one.

The layered machinery underneath (`fiddy.estimate`, `fiddy.extrapolation`,
`fiddy.noise`, ...) stays available directly for advanced/benchmark use
-- this module only adds the comparison-against-an-expected-value and
reporting layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._report import _get_printable_value, _wide_display
from .constants import Type
from .estimate import (
    DerivativeEstimate,
    JacobianEstimate,
    estimate_gradient,
    estimate_jacobian,
)
from .executor import Executor
from .output import OutputSchema

__all__ = [
    "DirectionCheckResult",
    "GradientCheckResult",
    "JacobianCheckResult",
    "check_gradient",
    "check_jacobian",
]


def _check_direction(
    direction_index: int,
    estimate: DerivativeEstimate,
    expectation: float,
    tol: float | None,
    k: float,
) -> DirectionCheckResult:
    """Shared pass/fail/inconclusive logic for one (direction[, output])
    pair -- used by both :func:`check_gradient` and :func:`check_jacobian`
    so the tolerance-flooring rule (see :func:`check_gradient`'s `k`
    parameter) has exactly one implementation.

    :param direction_index: The direction's index, for the result.
    :param estimate: The direction's finite-difference estimate.
    :param expectation: The value being checked against.
    :param tol: A fixed tolerance, or `None` to auto-derive one from `k`.
    :param k: Multiplier on `estimate`'s own error estimate, used when
        `tol` is `None`.
    :return: The pass/fail/inconclusive result for this direction.
    """
    direction_tol = (
        tol
        if tol is not None
        else max(k * estimate.error_estimate, estimate.diagnostics["tol"])
    )
    if estimate.status != "converged":
        outcome = "inconclusive"
    elif abs(estimate.value - expectation) <= direction_tol:
        outcome = "passed"
    else:
        outcome = "failed"
    return DirectionCheckResult(
        direction_index=direction_index,
        test=estimate.value,
        expectation=float(expectation),
        tol=direction_tol,
        outcome=outcome,
        estimate=estimate,
    )


@dataclass
class DirectionCheckResult:
    direction_index: int
    test: float
    """The finite-difference estimate."""
    expectation: float
    """The supplied gradient value being checked."""
    tol: float
    """The tolerance used for this direction."""
    outcome: str
    """One of "passed", "failed", "inconclusive" -- "inconclusive" means
    the FD engine itself could not produce a trustworthy estimate for this
    direction (`estimate.status` was "noise_dominated" or
    "discontinuity_suspected"), not that the supplied gradient was wrong.
    Reported separately from "failed" rather than counted as a check
    failure, following the convention that mismatches near kinks/noise
    floors are expected, not bugs (``Cs231n`` in ``doc/references.bib``)."""
    estimate: DerivativeEstimate
    """The full underlying result, for diagnostics/plotting."""


@dataclass
class GradientCheckResult:
    direction_results: list[DirectionCheckResult]
    success: bool
    """True iff no direction's outcome is "failed". Directions marked
    "inconclusive" do not affect `success` but are reported -- see
    `DirectionCheckResult.outcome`."""

    @property
    def df(self) -> pd.DataFrame:
        rows = [
            {
                "direction_index": r.direction_index,
                "test": r.test,
                "expectation": r.expectation,
                "abs_diff": abs(r.test - r.expectation),
                "tol": r.tol,
                "status": r.estimate.status,
                "outcome": r.outcome,
            }
            for r in self.direction_results
        ]
        return pd.DataFrame(rows).set_index("direction_index")

    def assert_success(self, always_print: bool = False) -> None:
        """Assert that this gradient check succeeded.

        :param always_print: Print the summary even if the check
            succeeded (e.g. to surface inconclusive directions even on
            success).
        :raises AssertionError: If the check failed.
        """
        message = self._build_report()

        if not self.success:
            raise AssertionError(message)

        has_inconclusive = any(
            r.outcome == "inconclusive" for r in self.direction_results
        )
        if has_inconclusive or always_print:
            print(message)

    def _build_report(self) -> str:
        df = self.df
        n_total = len(df)
        n_failed = int((df["outcome"] == "failed").sum())
        n_inconclusive = int((df["outcome"] == "inconclusive").sum())
        n_passed = n_total - n_failed - n_inconclusive
        status = "PASSED" if self.success else "FAILED"

        header = (
            f"Gradient check {status} "
            f"({n_passed}/{n_total} passed, {n_failed} failed, "
            f"{n_inconclusive} inconclusive)"
        )
        rule = "=" * len(header)
        lines = [rule, header, rule]

        notable = df[df["outcome"] != "passed"].copy()
        if len(notable):
            notable = notable.loc[
                notable["abs_diff"].sort_values(ascending=False).index
            ]
            for column in ("test", "expectation", "abs_diff", "tol"):
                notable[column] = notable[column].map(_get_printable_value)
            lines.append("")
            lines.append("Non-passing directions (failed and inconclusive):")
            with _wide_display():
                lines.append(str(notable))
        lines.append(rule)
        return "\n".join(lines)


def check_gradient(
    function: Type.FUNCTION,
    point: Type.POINT,
    expected,
    directions: list[Type.DIRECTION] | None = None,
    random_directions: int | None = None,
    rng: Type.SEED_LIKE | Type.RNG_LIKE | None = None,
    tol: float | None = None,
    k: float = 3.0,
    noise_floor: float | None = None,
    nondet_tol: float = 0.0,
    n_rungs: int = 8,
    step_ratio: float = 2.0,
    bounds: Type.BOUNDS | None = None,
    noise_floor_strategy: str = "auto",
    executor: Executor | None = None,
) -> GradientCheckResult:
    """Check a supplied gradient against a finite-difference estimate.

    Only `function`, `point`, and `expected` are required; no step sizes,
    and (by default) no tolerance.

    :param function: The blackbox function.
    :param point: The point to check the gradient at.
    :param expected: The gradient to check: one value per direction in
        `directions` (or per component of `point`, if `directions` is not
        given, since it then defaults to the standard basis) -- or, in
        `random_directions` mode, the full gradient vector to project.
    :param directions: Defaults to the standard basis (one direction per
        component of `point`), i.e. checking the full gradient.
    :param random_directions: Optional cheap-check mode (inspired by
        `torch.autograd.gradcheck`'s `fast_mode` (``TorchGradcheck``) and
        `jax.test_util.check_grads`'s single-random-direction default
        (``JaxCheckGrads``) -- both in ``doc/references.bib``): instead
        of checking every requested direction exactly, check this
        many random unit directions instead, projecting `expected` (which
        must then be the *full* gradient vector, `len(point)` entries)
        onto each via a dot product. Reduces the check's cost from
        `O(len(point))` to `O(random_directions)` evaluations, at the
        cost of no longer checking any specific parameter exactly --
        **never the default**; only use this for very large parameter
        vectors where checking every parameter individually is
        infeasible. Mutually exclusive with `directions`.
    :param rng: A seed or `numpy.random.Generator` for the random
        directions (only used with `random_directions`), per `SPEC 7
        <https://scientific-python.org/specs/spec-0007/>`_ -- anything
        :func:`numpy.random.default_rng` accepts.
    :param tol: Optional fixed tolerance applied to every direction,
        bypassing the default per-direction auto-derived tolerance.
        Prefer leaving this as `None` -- a fixed tolerance is exactly the
        per-model tuning this API is meant to make unnecessary.
    :param k: Multiplier on each direction's own FD error estimate to get
        its tolerance (``tol_direction = max(k * error_estimate,
        estimate.diagnostics["tol"])``) when `tol` is not given -- one
        small, model-independent constant rather than a per-model
        tolerance. Calibrated against a real ODE-based likelihood: every
        direction's actual error there was already within 1x its own
        reported error estimate, so `k=3` leaves comfortable headroom
        without reintroducing per-model tuning. The engine's own noise-
        derived `tol` is used as a floor because `error_estimate` can
        legitimately come out as exactly 0 (the corroborating chains and
        full ladder all agreeing to double precision by coincidence for
        a genuinely smooth function) -- `k * 0` would otherwise demand an
        unreachable exact match.
    :param noise_floor: See :func:`fiddy.estimate.estimate_gradient`.
    :param nondet_tol: See :func:`fiddy.estimate.estimate_gradient`.
    :param n_rungs: See :func:`fiddy.estimate.estimate_gradient`.
    :param step_ratio: See :func:`fiddy.estimate.estimate_gradient`.
    :param bounds: Optional per-parameter valid domain -- e.g. a model's
        declared parameter bounds -- that no probe or step is ever
        allowed to step outside of; see
        :func:`fiddy.step_size.clamp_step_to_bounds`. `None` (the
        default) disables clamping entirely. `point` itself must already
        satisfy `bounds`. Forwarded to :func:`fiddy.estimate.estimate_gradient`.
    :param noise_floor_strategy: See :func:`fiddy.estimate.estimate_gradient`
        -- `"auto"` (the default) transparently falls back to an
        independent noise-floor probe per direction whenever the cheap
        shared probe comes back unconfident (as it reliably does once a
        parameter sits close to its own `bounds`), so supplying `bounds`
        does not on its own require choosing a strategy here.
    :param executor: See :func:`fiddy.estimate.estimate_gradient`.
    :return: The gradient check result.
    :raises ValueError: If both `directions` and `random_directions` are
        given, if `random_directions` is used with an `expected` that
        isn't the full gradient vector, if `expected` doesn't have one
        entry per direction, if `point` violates `bounds`, or if
        `noise_floor_strategy` is invalid.
    :raises fiddy.function.FunctionEvaluationError: If `function` raises
        an exception, or returns a non-finite (``NaN``/``inf``) value, at
        any point evaluated during the check (e.g. an ODE solver failing
        to converge at a perturbed parameter value). This is a distinct
        failure mode from a `"noise_dominated"`/`"discontinuity_suspected"`
        direction result: those mean the function *did* return usable
        values but the derivative could not be confidently resolved from
        them, whereas this means the function itself could not produce a
        valid value at all -- almost always a problem with `function`
        (or the model/simulator behind it), not with fiddy's algorithm.
    """
    if random_directions is not None:
        if directions is not None:
            raise ValueError(
                "Specify only one of `directions` and `random_directions`."
            )
        point_arr = np.asarray(point, dtype=float)
        full_gradient = np.atleast_1d(np.asarray(expected, dtype=float))
        if len(full_gradient) != len(point_arr):
            raise ValueError(
                "`random_directions` mode requires `expected` to be the "
                f"full gradient vector ({len(point_arr)} entries), got "
                f"{len(full_gradient)}."
            )
        rng = np.random.default_rng(rng)
        raw = rng.standard_normal((random_directions, len(point_arr)))
        directions = [row / np.linalg.norm(row) for row in raw]
        expected = [float(np.dot(full_gradient, d)) for d in directions]

    estimates = estimate_gradient(
        function,
        point,
        directions=directions,
        noise_floor=noise_floor,
        nondet_tol=nondet_tol,
        n_rungs=n_rungs,
        step_ratio=step_ratio,
        bounds=bounds,
        noise_floor_strategy=noise_floor_strategy,
        executor=executor,
    )
    expected = np.atleast_1d(np.asarray(expected, dtype=float))
    if len(expected) != len(estimates):
        raise ValueError(
            f"`expected` has {len(expected)} entries but there are "
            f"{len(estimates)} directions to check."
        )

    # `k * error_estimate` alone is not a safe floor: `error_estimate` can
    # legitimately come out as exactly 0 (the corroborating chains and
    # full ladder all agreeing to double precision by coincidence),
    # understating the FD engine's real achievable precision. Fall back to
    # the engine's own noise-derived `tol` (`fiddy.estimate._default_tol`,
    # already used for its own "converged" classification) as a floor for
    # exactly this case -- see `_check_direction`.
    direction_results = [
        _check_direction(i, estimate, expectation, tol, k)
        for i, (estimate, expectation) in enumerate(
            zip(estimates, expected, strict=True)
        )
    ]

    success = all(r.outcome != "failed" for r in direction_results)
    return GradientCheckResult(
        direction_results=direction_results, success=success
    )


@dataclass
class JacobianCheckResult:
    """Per-output-component :class:`GradientCheckResult`\\ s from one
    shared batch of evaluations (see :func:`check_jacobian`).

    Indexed as ``output_results[output_index]``; if the checked function
    returned a named dict (see :class:`fiddy.Function`), `schema` lets a
    named output be looked up by name via :meth:`output` instead of a raw
    flat index, mirroring :class:`fiddy.estimate.JacobianEstimate`.
    """

    output_results: list[GradientCheckResult]
    schema: OutputSchema | None = None
    success: bool = field(init=False)

    def __post_init__(self) -> None:
        self.success = all(r.success for r in self.output_results)

    def _resolve_flat_index(self, name_or_index: str | int) -> int:
        if not isinstance(name_or_index, str):
            return int(name_or_index)
        if self.schema is None or not self.schema.is_structured:
            raise KeyError(
                f"No named output {name_or_index!r}: the checked function "
                "did not return a named (dict) output."
            )
        if name_or_index not in self.schema.names:
            raise KeyError(
                f"Unknown output name {name_or_index!r}; available: "
                f"{self.schema.names}"
            )
        return self.schema.slices[name_or_index].start

    def output(self, name_or_index: str | int) -> GradientCheckResult:
        """The check result for one output component.

        :param name_or_index: A named output (if the checked function
            returned a dict), or a flat integer index. For a multi-
            component named output, returns its first component --
            index `.output_results` directly for a specific component.
        :return: The output component's gradient check result.
        :raises KeyError: If `name_or_index` names an output the checked
            function did not return, or the function's output wasn't
            named at all.
        """
        return self.output_results[self._resolve_flat_index(name_or_index)]

    def assert_success(self, always_print: bool = False) -> None:
        """Assert that every output component's check succeeded.

        :param always_print: Print every output's summary even if it
            succeeded (e.g. to surface inconclusive directions even on
            success).
        :raises AssertionError: If any output component's check failed.
        """
        failures = []
        for i, result in enumerate(self.output_results):
            try:
                result.assert_success(always_print=always_print)
            except AssertionError as error:
                failures.append(f"--- output {i} ---\n{error}")
        if failures:
            raise AssertionError("\n\n".join(failures))


def _flatten_expected_jacobian(
    expected, schema: OutputSchema | None, n_outputs: int, n_directions: int
) -> np.ndarray:
    """Align a caller-supplied expected Jacobian with `estimate_jacobian`'s
    ``estimates[output_index][direction_index]`` layout.

    `expected` may be a plain array already shaped ``(n_outputs,
    n_directions)``, or (if the checked function returned a named dict --
    see :class:`fiddy.output.OutputSchema`) a dict mapping each output
    name to an array of shape ``(*that output's own shape,
    n_directions)`` -- e.g. per-variable sensitivity arrays after moving
    the parameter axis last, one entry per bundled output variable,
    mirroring how the function's own output was bundled.

    :param expected: The expected Jacobian, in either form above.
    :param schema: The checked function's output schema (`None` if it
        returned a plain array).
    :param n_outputs: The number of output components being checked.
    :param n_directions: The number of directions being checked.
    :return: `expected`, aligned to shape ``(n_outputs, n_directions)``.
    :raises ValueError: If `expected` is a dict but the function's output
        wasn't named, is missing a named output, or doesn't have shape
        ``(n_outputs, n_directions)`` once aligned.
    """
    if isinstance(expected, dict):
        if schema is None or not schema.is_structured:
            raise ValueError(
                "`expected` is a dict, but the checked function did not "
                "return a named (dict) output."
            )
        missing = [name for name in schema.names if name not in expected]
        if missing:
            raise ValueError(
                f"`expected` is missing output(s) {missing} "
                f"(schema names: {schema.names})."
            )
        parts = []
        for name in schema.names:
            component_count = int(np.prod(schema.shapes[name], dtype=int)) or 1
            arr = np.asarray(expected[name], dtype=float).reshape(
                component_count, -1
            )
            parts.append(arr)
        flat = np.concatenate(parts, axis=0)
    else:
        flat = np.atleast_2d(np.asarray(expected, dtype=float))

    if flat.shape != (n_outputs, n_directions):
        raise ValueError(
            f"`expected` has shape {flat.shape} once flattened/aligned, "
            f"but the Jacobian being checked has {n_outputs} output "
            f"component(s) and {n_directions} direction(s)."
        )
    return flat


def check_jacobian(
    function: Type.FUNCTION,
    point: Type.POINT,
    expected,
    directions: list[Type.DIRECTION] | None = None,
    tol: float | None = None,
    k: float = 3.0,
    noise_floor: float | None = None,
    nondet_tol: float = 0.0,
    n_rungs: int = 8,
    step_ratio: float = 2.0,
    bounds: Type.BOUNDS | None = None,
    noise_floor_strategy: str = "auto",
    executor: Executor | None = None,
) -> JacobianCheckResult:
    """Check every output component of a bundled multi-output function at
    once -- e.g. a model's state/observable/likelihood sensitivities
    together, not just its scalar objective.

    Checking all of a model's forward sensitivities together is a central
    use case this engine is meant to support well, not an edge case.
    Built on :func:`fiddy.estimate.estimate_jacobian`, so every output
    component shares the same batch of perturbed-point evaluations per
    direction -- checking N outputs costs no more function evaluations
    than :func:`check_gradient` checking 1.

    :param function: The blackbox function.
    :param point: The point to check the Jacobian at.
    :param expected: The Jacobian to check against -- either a plain
        array of shape ``(n_outputs, n_directions)``, or (if `function`
        returns a named dict) a dict mapping each output name to an
        array of shape ``(*that output's shape, n_directions)`` -- see
        :func:`_flatten_expected_jacobian`.
    :param directions: Defaults to the standard basis (one direction per
        component of `point`), i.e. the full Jacobian.
    :param tol: See :func:`check_gradient` -- applied identically to
        every (output, direction) pair.
    :param k: See :func:`check_gradient` -- applied identically to every
        (output, direction) pair.
    :param noise_floor: Forwarded to
        :func:`fiddy.estimate.estimate_jacobian`.
    :param nondet_tol: Forwarded to
        :func:`fiddy.estimate.estimate_jacobian`.
    :param n_rungs: Forwarded to :func:`fiddy.estimate.estimate_jacobian`.
    :param step_ratio: Forwarded to
        :func:`fiddy.estimate.estimate_jacobian`.
    :param bounds: See :func:`check_gradient`. Forwarded to
        :func:`fiddy.estimate.estimate_jacobian`.
    :param noise_floor_strategy: See :func:`check_gradient`. Forwarded to
        :func:`fiddy.estimate.estimate_jacobian`.
    :param executor: Forwarded to
        :func:`fiddy.estimate.estimate_jacobian`.
    :return: The Jacobian check result.
    :raises ValueError: If `point` violates `bounds`, or
        `noise_floor_strategy` is invalid.
    :raises fiddy.function.FunctionEvaluationError: See
        :func:`check_gradient`.
    """
    jacobian: JacobianEstimate = estimate_jacobian(
        function,
        point,
        directions=directions,
        noise_floor=noise_floor,
        nondet_tol=nondet_tol,
        n_rungs=n_rungs,
        step_ratio=step_ratio,
        bounds=bounds,
        noise_floor_strategy=noise_floor_strategy,
        executor=executor,
    )
    expected_flat = _flatten_expected_jacobian(
        expected, jacobian.schema, jacobian.n_outputs, jacobian.n_directions
    )

    output_results = []
    for j, row in enumerate(jacobian.estimates):
        direction_results = [
            _check_direction(i, estimate, expectation, tol, k)
            for i, (estimate, expectation) in enumerate(
                zip(row, expected_flat[j], strict=True)
            )
        ]
        success = all(r.outcome != "failed" for r in direction_results)
        output_results.append(
            GradientCheckResult(
                direction_results=direction_results, success=success
            )
        )

    return JacobianCheckResult(
        output_results=output_results, schema=jacobian.schema
    )
