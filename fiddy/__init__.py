# from .gradient_check import gradient_check
from importlib.metadata import PackageNotFoundError, version

from .constants import *
from .derivative import Derivative, get_derivative
from .directional_derivative import methods

# from . import difference
from .function import CachedFunction, Function
from .numpy import fiddy_array

# from . import quotient
from .step import step

try:
    __version__ = version("fiddy")
except PackageNotFoundError:
    # package is not installed
    pass
