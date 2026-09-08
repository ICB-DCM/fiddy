# fiddy

[![Test suite](https://github.com/ICB-DCM/fiddy/actions/workflows/test_suite.yml/badge.svg)](https://github.com/ICB-DCM/fiddy/actions/workflows/test_suite.yml)
[![PyPI](https://badge.fury.io/py/fiddy.svg)](https://badge.fury.io/py/fiddy)
[![Documentation](https://readthedocs.org/projects/fiddy/badge/?version=latest)](https://fiddy.readthedocs.io)

Robust [finite difference](https://en.wikipedia.org/wiki/Finite_difference)
derivative estimation and gradient checking for blackbox functions -- with
a particular focus on functions that are noisy (e.g. adaptive-step ODE
solvers), expensive to evaluate, and where you don't want to hand-tune
step sizes or tolerances per model.

## Estimating a gradient

No step sizes to choose: fiddy empirically estimates the function's own
noise floor, builds an appropriate step-size ladder, and extrapolates a
value with a corroborated error estimate.

```python
from fiddy import estimate_gradient
import numpy as np


def function(x):
    return np.sin(x[0]) * np.cos(x[1])


results = estimate_gradient(function, np.array([0.6, -0.3]))
gradient = np.array([r.value for r in results])
```

Each entry is a `DerivativeEstimate`, not just a number: `r.error_estimate`/
`r.status` flag noise-dominated or meaningless (e.g. kink) values instead
of silently trusting them, backed by lower-level diagnostics (`fiddy.noise`,
`fiddy.extrapolation`, ...) if you need to dig further.

See `doc/examples/derivative.ipynb` for a guided walkthrough of the
problems this solves (noise, kinks, near-zero gradients) and how.

## Every output at once (a Jacobian, not just a gradient)

If your function instead returns a `dict` of named arrays (e.g. `{"x":
..., "y": ..., "llh": ...}`), `estimate_jacobian` estimates *every* named
output's derivative at once, from the same batch of perturbed-point
evaluations -- N outputs cost no more function evaluations than 1:

```python
from fiddy import estimate_jacobian
import numpy as np


def function(x):
    return {"a": x[0] ** 2 + x[1], "b": np.sin(x[0]) * x[1]}


jacobian = estimate_jacobian(function, np.array([1.3, 0.6]))
jacobian.values  # shape (n_outputs, n_directions)
jacobian.output("a")  # one output's DerivativeEstimate per direction
```

## Checking a gradient against an expected value

`check_gradient`/`check_jacobian` wrap the same engine to compare against
something already computed (e.g. an analytic/adjoint gradient), auto-
deriving each direction's tolerance from its own error estimate -- no
tolerance to tune by default:

```python
from fiddy import check_gradient

result = check_gradient(function, point, expected_gradient)
result.assert_success()
```

`check_jacobian` is the equivalent for a dict-returning function, taking
`expected` as either a plain array or the same named-dict shape as
`function`'s own output.

## Installation

Currently under development -- the API may still change between
releases, so pinning an exact version is advisable. Install from source:
```bash
pip install -e .
```

Can also be installed from [PyPI](https://pypi.org/project/fiddy/)
```bash
pip install fiddy
```

Optional extras: `fiddy[examples]` (notebook, plotting), `fiddy[tests]`.
