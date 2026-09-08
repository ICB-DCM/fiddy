About fiddy
===========

fiddy is a robust `finite difference
<https://en.wikipedia.org/wiki/Finite_difference>`_ derivative estimation
and gradient-checking library for blackbox functions, with a particular
focus on functions that are noisy (e.g. adaptive-step ODE solvers),
expensive to evaluate, and where hand-tuning step sizes or tolerances
per model is undesirable.

No step sizes, and by default no tolerance: fiddy empirically estimates
the function's own noise floor, builds an appropriate step-size ladder,
extrapolates a value with a corroborated error estimate, and (when
checking against an expected value) derives each direction's tolerance
from that error estimate automatically.

Architecture
------------

The engine is a layered chain of modules, each composing the previous:

1. :mod:`fiddy.output` -- bundles a plain array or a dict of named
   arrays (e.g. ``{"x": ..., "y": ..., "llh": ...}``) into one flat
   vector, and unbundles it again later.
2. :mod:`fiddy.function` -- wraps a raw callable, applies the bundling
   above plus output-structure-consistency validation across calls,
   with optional disk/RAM caching.
3. :mod:`fiddy.noise` -- empirical, model-agnostic noise-floor
   estimation (an ECNoise-style plateau-detection heuristic).
4. :mod:`fiddy.step_size` -- builds a geometric step-size ladder from a
   noise floor, with a safety clamp against pathological noise
   estimates.
5. :mod:`fiddy.extrapolation` -- Neville extrapolation over the ladder
   with two independently-corroborating chains, vectorized over an
   output-component axis for multi-output support.
6. :mod:`fiddy.discontinuity` -- kink/discontinuity detection via a
   forward-backward-central "gap" cross-check, also vectorized.
7. :mod:`fiddy.executor` -- batch (optionally parallel) function
   evaluation.
8. :mod:`fiddy.estimate` -- composes all of the above into
   :func:`~fiddy.estimate.estimate_gradient`/
   :func:`~fiddy.estimate.estimate_jacobian`.
9. :mod:`fiddy.check` -- the public comparison/reporting layer on top:
   :func:`~fiddy.check.check_gradient`/:func:`~fiddy.check.check_jacobian`.
10. :mod:`fiddy.plotting` -- optional (matplotlib), diagnostic plots
    mirroring the data structures above.

Design principles
------------------

- **Never framework-specific.**
  While fiddy is developed in the context of sensitivity checking for
  ODE simulation in AMICI, it is designed to be framework-agnostic.
  Integration adapters live in the consumer's own repository.
- **Honest uncertainty over confident wrongness.** Every derivative
  estimate carries a status (``"converged"``, ``"noise_dominated"``,
  ``"discontinuity_suspected"``); a gradient check reports
  non-converged directions as inconclusive, never silently folded into
  a false pass or a false fail.
- **No user-supplied step sizes or tolerances by default.** Everything
  is derived from the function's own empirically measured noise floor.
- **Batch-then-analyze.** Every phase that needs several evaluations
  decides the whole batch upfront and dispatches it through one
  executor call, so parallelizing is a matter of swapping the executor,
  never restructuring the numerics.
- **Robustness by default, not competing on speed.** fiddy's
  noise-probe-and-ladder cost is deliberate overhead, paid even on
  already-well-behaved functions, in exchange for a status it actually
  trusts.
