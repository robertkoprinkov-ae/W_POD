API reference
=============

Classes and functions of ``src/W_POD`` in this repository.
Private helpers are listed only where they matter for understanding the
numerics.

W_POD (1D)
----------

.. py:module:: W_POD
   :synopsis: Transported modal decomposition of snapshot data (1D modules at the top level).

.. toctree::
   :maxdepth: 1

   w_pod/util
   w_pod/modes
   w_pod/interp
   w_pod/ansatz
   w_pod/solver
   w_pod/hess_diag_probed
   w_pod/guarded_solver
   w_pod/visualizer

W_POD.two_dim (2D wildfire)
---------------------------

.. py:module:: W_POD.two_dim
   :synopsis: 2D wildfire ansatz, losses and Hessian probing.

.. toctree::
   :maxdepth: 1

   wildfire/ansatz
   wildfire/losses
   wildfire/hess_diag_probed_2d
   wildfire/visu

Parameter-vector layout
-----------------------

Both codes optimise one flat vector. Its blocks, in order:

**1D** (:class:`W_POD.ansatz.Ansatz`), one block per mode, concatenated in
mode order:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Mode
     - Block
   * - :class:`~W_POD.modes.GlobalMode`
     - ``pod_global`` (N), ``alpha_coeff``
   * - :class:`~W_POD.modes.TransportedMode`
     - ``pod_local`` (N), ``beta_coeff``, ``gamma_coeff``
   * - :class:`~W_POD.modes.StretchedMode`
     - ``pod_local`` (N), ``beta_coeff``, ``gamma_shift_coeff``,
       ``gamma_stretch_coeff``

N is the number of mesh nodes. Each ``*_coeff`` block has
:meth:`RBF_interpol.n_param() <W_POD.util.RBF_interpol.n_param>` entries:
:math:`n_z`, plus one for the bias.

**2D** (:class:`W_POD.two_dim.ansatz.Ansatz`), one block per coordinate
frame. Inside a frame, the transformation comes first and the modes follow:

* ``coordinate_transformation``: ``radial_0_coeff`` … ``radial_{n-1}_coeff``,
  ``shift_x_coeff``, ``shift_y_coeff`` (empty for the identity);
* ``mode_k``: ``mode`` (one value per pixel, row-major
  :math:`n_y \times n_x`), then ``amplitude_coeff`` unless the amplitude is
  fixed.
