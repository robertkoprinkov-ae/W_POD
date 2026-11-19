W_POD
=====

**W_POD** finds low-dimensional, *transported* modal decompositions of
transport-dominated snapshot data. Each snapshot is written as a sum of modes
that are moved (shifted, stretched, radially scaled) and rescaled as a function
of the snapshot parameter :math:`z`:

.. math::

   U(z) \approx \sum_i \alpha_i(z)\, \mathcal{C}_{\gamma_i(z)}(\varphi_i).

The modes :math:`\varphi_i`, the amplitudes :math:`\alpha_i(z)` and the
transport parameters :math:`\gamma_i(z)` are all found non-intrusively, by
minimising a weighted sum of an :math:`L^2` and an entropic Wasserstein
reconstruction error. Each trajectory in :math:`z` is a kernel interpolant
whose RKHS norm can be penalised.

The package :mod:`W_POD` comes in two parts:

* the 1D modules (``W_POD.ansatz``, ``W_POD.modes``, ``W_POD.solver``, …),
  used by the crossing-waves and advection–diffusion experiments;
* the 2D subpackage :mod:`W_POD.two_dim` (wildland fire experiment), which
  reuses the kernel interpolator and the Hessian colouring of the 1D modules.

.. important::

   This documentation describes the code in this repository, which is the code
   of the paper's final runs. :doc:`versions` lists the exact package versions
   the runs used.

.. toctree::
   :maxdepth: 1

   api/index
   theory

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
