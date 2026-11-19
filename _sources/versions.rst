Code and package versions
=========================

The code in this repository (``src/W_POD``, ``examples/``) is the code of the
paper's final runs.

Package versions
----------------

The runs used two Python environments. Each requirements file pins every
dependency of :mod:`W_POD`, plus everything those pull in, at the exact
version of that environment:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Runs
     - File
     - Environment
   * - CPU runs
     - ``requirements_cpu.txt``
     - Python 3.14.2
   * - GPU runs
     - ``requirements_gpu.txt``
     - Python 3.14.6, CUDA 12 (``jax-cuda12-plugin`` and the NVIDIA CUDA 12
       wheels)

The numerical core is the same in both: ``jax``/``jaxlib`` 0.9.2,
``ott-jax`` 0.6.0, ``lineax`` 0.1.0, ``interpax`` 0.3.14, ``scipy`` 1.17.1 and
``numpy`` 2.4.6. They differ only in plotting, I/O and utility packages
(``matplotlib``, ``h5py``, ``optax``, …). See :doc:`installation` for how to
install either environment.

Floating-point precision
------------------------

Every driver in ``examples/`` enables ``jax_enable_x64``, except those in
``examples/crossing_waves/exp1/``, which run in float32.
