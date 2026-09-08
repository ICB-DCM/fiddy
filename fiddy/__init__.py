"""Robust finite-difference gradient checking/computation for blackbox
functions, with a particular focus on functions that are noisy,
expensive to evaluate, and where step sizes/tolerances shouldn't need
hand-tuning per model. Start with :func:`check_gradient` or
:func:`estimate_gradient`; see each submodule for the layered engine
underneath (:mod:`fiddy.noise` through :mod:`fiddy.check`).
"""

from importlib.metadata import PackageNotFoundError, version

from .check import (
    DirectionCheckResult,
    GradientCheckResult,
    JacobianCheckResult,
    check_gradient,
    check_jacobian,
)
from .constants import *
from .discontinuity import DiscontinuityCheck, check_discontinuity
from .estimate import (
    DerivativeEstimate,
    EstimateKwargs,
    JacobianEstimate,
    estimate_directional_derivative,
    estimate_gradient,
    estimate_jacobian,
)
from .executor import Executor, JoblibExecutor, SequentialExecutor
from .extrapolation import ExtrapolationResult, extrapolate_central_differences
from .function import CachedFunction, Function, FunctionEvaluationError
from .noise import NoiseFloor, estimate_model_noise_floor, estimate_noise_floor
from .output import OutputSchema, flatten_output
from .step_size import build_step_ladder

try:
    __version__ = version("fiddy")
except PackageNotFoundError:
    # package is not installed
    pass
