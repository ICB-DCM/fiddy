fiddy
=====

Robust `finite difference <https://en.wikipedia.org/wiki/Finite_difference>`_
derivative estimation and gradient checking for blackbox functions --
with a particular focus on functions that are noisy (e.g. adaptive-step
ODE solvers), expensive to evaluate, and where you don't want to
hand-tune step sizes or tolerances per model.

Quickstart
----------

No step sizes to choose: fiddy empirically estimates the function's own
noise floor, builds an appropriate step-size ladder, and extrapolates a
value with a corroborated error estimate.

.. code-block:: python

    from fiddy import estimate_gradient
    import numpy as np


    def function(x):
        return np.sin(x[0]) * np.cos(x[1])


    results = estimate_gradient(function, np.array([0.6, -0.3]))
    gradient = np.array([r.value for r in results])

See :doc:`about` for the full picture (multi-output Jacobians, checking
against an expected gradient, ...) and :doc:`examples/derivative` for a
guided walkthrough of the problems this solves (noise, kinks, near-zero
gradients) and how.

Installation
------------

.. code-block:: bash

    pip install fiddy

Currently under development -- the API may still change between
releases, so pinning an exact version is advisable. Optional extras:
``fiddy[examples]`` (notebook, plotting), ``fiddy[tests]``.

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   about
   examples/derivative

.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:

   fiddy

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
