# fiddy

[![Test suite](https://github.com/ICB-DCM/fiddy/actions/workflows/test_suite.yml/badge.svg)](https://github.com/ICB-DCM/fiddy/actions/workflows/test_suite.yml)
[![PyPI](https://badge.fury.io/py/fiddy.svg)](https://badge.fury.io/py/fiddy)
[![Documentation](https://readthedocs.org/projects/fiddy/badge/?version=latest)](https://fiddy.readthedocs.io)

Robust [finite difference](https://en.wikipedia.org/wiki/Finite_difference)
gradient checking for blackbox functions -- with a particular focus on
functions that are noisy (e.g. adaptive-step ODE solvers), expensive to
evaluate, and where you don't want to hand-tune step sizes or tolerances
per model.

```python
from fiddy import check_gradient

result = check_gradient(function, point, expected_gradient)
result.assert_success()
```

No step sizes, and by default no tolerance: `fiddy` empirically estimates
the function's own noise floor, builds an appropriate step-size ladder,
extrapolates a value with a corroborated error estimate, and derives each
direction's check tolerance from that error estimate automatically. See
`doc/examples/derivative.ipynb` for a guided walkthrough of the problems
this solves (noise, kinks, near-zero gradients) and how.

## Computing a gradient (not just checking one)

The same engine that backs `check_gradient` is available directly, with no
`expected` gradient required -- useful when you want an FD gradient
computed, not compared against something else:

```python
from fiddy import estimate_gradient
import numpy as np


def function(x):
    return np.sin(x[0]) * np.cos(x[1])


results = estimate_gradient(function, np.array([0.6, -0.3]))
gradient = np.array([r.value for r in results])
```

Each entry in `results` is a full `DerivativeEstimate`, not just a number:
`r.error_estimate` and `r.status` (e.g. `"converged"`, `"noise_dominated"`,
`"discontinuity_suspected"`) let you decide whether to trust a given
gradient component, rather than silently using a value that may be
noise-dominated or meaningless (e.g. at a genuine kink). `estimate_gradient`
also accepts `executor=` (`fiddy.executor.JoblibExecutor()` for
process-based parallelism) and `directions=` (to compute only a subset of
components).

The rest of the layered engine (`fiddy.noise`, `fiddy.extrapolation`, ...)
is available too, for custom tolerances or lower-level diagnostics.

## Checking/computing every output at once (a Jacobian, not just a gradient)

`check_gradient`/`estimate_gradient` estimate the derivative of a single
plain value. If your function instead returns a `dict` of named arrays
(e.g. `{"x": ..., "y": ..., "llh": ...}`), `check_jacobian`/
`estimate_jacobian` check/compute *every* named output's derivative at
once, from the very same batch of perturbed-point evaluations -- checking
N outputs costs no more function evaluations than checking 1:

```python
from fiddy import check_jacobian
import numpy as np


def function(x):
    return {"a": x[0] ** 2 + x[1], "b": np.sin(x[0]) * x[1]}


point = np.array([1.3, 0.6])
# One row per output ("a", "b"), one column per direction (x[0], x[1]).
expected = {
    "a": np.array([2 * point[0], 1.0]),
    "b": np.array([np.cos(point[0]) * point[1], np.sin(point[0])]),
}

result = check_jacobian(function, point, expected)
result.output("a").assert_success()  # a per-output GradientCheckResult
```

`estimate_jacobian` is the "compute, don't check" counterpart, returning a
`JacobianEstimate` indexed `[output][direction]` (or by output name via
`.output(...)`).

# Installation

Currently under development, please install from source.
```bash
pip install -e .
```

Can also be installed from [PyPI](https://pypi.org/project/fiddy/)
```bash
pip install fiddy
```

Optional extras: `fiddy[examples]` (notebook, plotting), `fiddy[tests]`.
