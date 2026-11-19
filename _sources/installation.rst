Installation and running
========================

Dependencies
------------

The code is plain Python (3.14 or newer) on top of JAX:

============== ==========================================================
Package        Used for
============== ==========================================================
``jax``        arrays, automatic differentiation, ``jit``/``vmap``
``ott-jax``    entropic optimal transport (``ott.solvers.linear.solve``)
``lineax``     linear solvers inside OTT's implicit differentiation
``interpax``   Catmull-Rom interpolation in 1D and 2D
``scipy``      SLSQP / L-BFGS-B, ``Bounds``, ``LinearConstraint``
``numpy``      Hessian probing bookkeeping
``h5py``       parameter output
``matplotlib`` plotting (``visualizer``, ``visu``)
``cycler``     line-style cycles in :mod:`W_POD.visualizer`
``tqdm``       progress bars during Hessian probing
============== ==========================================================

``ott-jax`` 0.6 also installs ``jaxopt``, which :mod:`W_POD` itself does not
import. The ``use_danskin`` semantics described in :doc:`numerics` are those
of OTT 0.6, so ``pyproject.toml`` restricts it to ``ott-jax>=0.6,<0.7``.

Installing the package
----------------------

:mod:`W_POD` is a Hatch project (``pyproject.toml``). Its wheel contains
``src/W_POD``, including the 2D subpackage :mod:`W_POD.two_dim`. From the root
of the repository:

.. code-block:: bash

   pip install -e .

To reproduce the exact environment of the paper's runs (:doc:`versions`),
install the pinned versions alongside:

.. code-block:: bash

   pip install -r requirements_cpu.txt -e .   # CPU runs, Python 3.14.2
   pip install -r requirements_gpu.txt -e .   # GPU runs, Python 3.14.6, CUDA 12

Optional extras:

* ``.[cuda]``: ``jax[cuda12]``, the GPU build of JAX;
* ``.[examples]``: ``mpi4py`` (drivers in ``examples/crossing_waves/exp2`` and
  ``examples/global_local``) and ``torch`` (``zorawski.py``);
* ``.[tests]``: ``pytest``, ``openturns`` and ``smt``, for ``tests/``.

``mpi4py`` is built against the MPI library found at install time, so load
your MPI first.

Running the examples
--------------------

Every driver writes its output relative to the current directory, so run it
from a folder where that is wanted.

* **Crossing waves and advection–diffusion** (``examples/crossing_waves``,
  ``examples/global_local``): the drivers generate their own data
  (:class:`~W_POD.util.DataCreation`) and create ``data/<casename>/`` and
  ``img/<casename>/``. Drivers that import ``mpi4py`` share their trials
  among the MPI ranks:

  .. code-block:: bash

     python cw_exp1_formulation_B_budget.py
     mpirun -n 10 python formulation_B.py      # global_local: 10 trials, 10 ranks

* **Wildland fire** (``examples/wildland_fire``): the drivers take the seed
  of the random start as their only argument. They read
  ``Wildlandfire_2d/1D_Grid_full_burela.npy``, ``data_full_burela.npy`` and
  ``data_z_full_burela.npy`` from the current directory, and write
  ``wildfire/seed_<case>_s<seed>.hdf5``. The ``wildfire/`` folder must exist
  beforehand:

  .. code-block:: bash

     mkdir -p wildfire
     python formulation_B_8_4.py 22

* ``zorawski.py`` (sPOD comparison) takes a seed as well, needs ``torch`` and
  queries a CUDA device at start-up.

..
   TODO: say where the wildland fire data (Wildlandfire_2d/*_burela.npy) can
   be obtained; it is not part of the repository.
