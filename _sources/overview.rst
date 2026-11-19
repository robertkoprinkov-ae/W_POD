Overview
========

The problem
-----------

Linear model reduction (POD) writes a snapshot as
:math:`U(z) = \sum_i \alpha_i(z)\varphi_i` with modes fixed in space. Data in
which a feature *moves* at roughly constant shape has a slowly decaying
Kolmogorov width, so such fixed-mode expansions need many modes. W_POD
instead lets every mode be carried by a parametrised coordinate
transformation:

.. math::

   U(z) \approx \sum_i \alpha_i(z)\, \mathcal{C}_{\gamma_i(z)}(\varphi_i),

where :math:`\mathcal{C}_\gamma` applies a discretised coordinate
transformation (a shift, a shift plus a stretch, or in 2D a radial scaling
around a centre) to the mode :math:`\varphi_i`.

Unknowns
--------

Given snapshots :math:`U_j = U(z_j)`, :math:`j = 1,\dots,n_z`, on a mesh
:math:`x`, the unknowns are:

* the **mode vectors** :math:`\varphi_i`, one value per mesh node;
* the **trajectories** :math:`\alpha_i(z)` (amplitude) and
  :math:`\gamma_i(z)` (transport). Each trajectory is a kernel interpolant
  (:class:`~W_POD.util.RBF_interpol`)

  .. math::

     y(z) = b + \sum_{j=1}^{n_z} a_j\, k(z, z_j),

  and its coefficients :math:`a_j` (plus the bias :math:`b` when
  ``bias=True``) are optimised.

All unknowns are stacked into one flat parameter vector. The ansatz classes
(:class:`W_POD.ansatz.Ansatz`, :class:`W_POD.two_dim.ansatz.Ansatz`) know its
block layout and convert between the flat vector and named dictionaries.

Objective
---------

The 1D objective (:meth:`W_POD.ansatz.AnsatzConfig._cost`) is

.. math::

   J(p) = w_{L^2}\,\frac{L^2(p)}{L^2_0}
        + w_{W}\,\frac{W(p)}{W_0}
        + \sum_{c} \lambda_c\, a_c^\top K_c\, a_c
        + w_{\text{exceed}}\, E(p)
        + w_{\text{pod}}\, P(p),

with

* :math:`L^2(p) = \sum_j \Delta x\, \lVert D(p; z_j) - U_j \rVert_2^2`, the
  squared :math:`L^2` reconstruction error. :math:`D(p;z)` is the
  reconstruction on the data mesh, obtained by pulling the mesh back through
  each mode's transformation and interpolating the mode there (paper
  Remark 3.15);
* :math:`W(p) = \sum_j \mathrm{OT}_\varepsilon(\mu_j(p), \nu_j)`, the entropic
  optimal-transport cost (OTT's ``reg_ot_cost``, squared-Euclidean ground
  cost, :math:`\varepsilon = 0.1`) between the reconstruction's point cloud
  :math:`\mu_j` and the snapshot :math:`\nu_j`, both normalised to unit mass.
  The point cloud is the mesh pushed *forward* through the transformation,
  carrying the mode values times the amplitude. The code calls this term
  ``w1``/``w2`` for historical reasons; it is a :math:`W_2`-type term and is
  **not** square-rooted;
* :math:`L^2_0`, :math:`W_0`: the values of the two terms at the start of
  the current optimisation stage, floored (see :ref:`normalisation`), so both
  terms start near 1;
* :math:`a_c^\top K_c a_c`: the RKHS norm of trajectory :math:`c`, without
  the bias (paper Formulation 3.8);
* :math:`E`, :math:`P`: an overshoot penalty and a mode-norm penalty. Both
  have weight 0 in the final runs.

The 2D objective (:meth:`W_POD.two_dim.losses.Losses._cost_sum`) is the same
without weights and extra penalties:
:math:`J = W/W_0 + L^2/L^2_0 + \lambda \sum \text{RKHS}`, with
:math:`\lambda = 0` in the final runs.

Regularised and unregularised runs
----------------------------------

Every final experiment has a *reg* and a *noreg* variant:

**noreg (Formulation 3.5).**
  Every kernel has length scale :math:`\theta = 10^{-6}`, so
  :math:`K \approx I`. The interpolant is then a lookup table: one free value
  per snapshot, with :math:`\lambda = 0`.

**reg (Formulation 3.8).**
  The kernel length scales are of the order of the snapshot spacing, so the
  trajectories are smooth in :math:`z`. In 1D the RKHS norm is also
  penalised. In 2D only the radial-scaling trajectories get a finite length
  scale (10) and there is no RKHS penalty.

Workflow
--------

#. Build one :class:`~W_POD.util.RBF_interpol` per kind of trajectory
   (kernel, :math:`\theta`, :math:`\sigma`, bias).
#. Build the modes and the ansatz.
#. Build the loss configuration: :class:`~W_POD.ansatz.AnsatzConfig` in 1D,
   :class:`~W_POD.two_dim.losses.Losses` and
   :class:`~W_POD.two_dim.losses.Losses_scaled` in 2D.
#. Run several optimisation *stages* from an initial guess. Each stage:

   a. renormalises the loss terms at its starting point;
   b. recomputes a diagonal **Hessian preconditioner** by probing
      (:mod:`W_POD.hess_diag_probed`, :mod:`W_POD.two_dim.hess_diag_probed_2d`);
   c. runs SciPy (SLSQP in 1D, L-BFGS-B in 2D) in the rescaled variables.

#. Write the parameter vector to HDF5 after each stage, or at the end in 2D.

The two tutorials, :doc:`tutorial_1d` and :doc:`tutorial_2d`, follow the
final driver scripts step by step.
