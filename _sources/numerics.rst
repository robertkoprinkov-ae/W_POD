Numerical methods
=================

This page describes what the code computes, including choices that are not
visible from the API alone. Hard-coded values are collected in
:doc:`constants`.

Transformations and interpolation
---------------------------------

**Pull-back for L2, push-forward for W.** The :math:`L^2` term needs the
reconstruction on the data mesh. Each mode is therefore evaluated at the
mesh points mapped *back* through the inverse transformation (for a shift,
:math:`x - \gamma(z)`) and interpolated there. The W term needs no
interpolation. It uses the mode's own nodes mapped *forward*
(:math:`x + \gamma(z)`), each carrying mass
:math:`\beta(z)\varphi(x)`.

**1D interpolation.** :class:`~W_POD.modes.TransportedMode` uses
:func:`~W_POD.interp.interp_decay`. Inside the mesh this is Catmull-Rom
interpolation (``interpax``). Outside, the edge value decays as
:math:`e^{-15\,d}`, where :math:`d` is the distance to the mesh edge, so the
reconstruction stays continuous and finite outside the domain.
:class:`~W_POD.modes.StretchedMode` uses
:func:`~W_POD.interp.interp_decay_knots`: two extra knots of value
:math:`10^{-6}` are placed 10 % of the domain width outside each end, with
``cubic2`` interpolation and extrapolation value 0. The ``*_exact``
methods, used only to generate synthetic data, use linear ``jnp.interp``
with 0 outside.

**2D transformation.** :class:`~W_POD.two_dim.ansatz.CoordinateTransformation`
maps a point :math:`p` with centre :math:`c` (centre of the bounding box of
the grid) and rigid shift :math:`s(z)` to

.. math::

   p' = c + s(z) + r(\vartheta; z)\,(p - c),
   \qquad \vartheta = \arg(p - c),

where :math:`r(\cdot; z)` is interpolated linearly and periodically between
the :math:`n_\vartheta` stations. Each station value is a kernel trajectory
and is used as a *factor* (``incr_per_pt = radial_shifts_pts``). A radial
scaling preserves the azimuth, so the inverse is exact:
:math:`p = c + (p' - c - s)/r(\arg(p' - c - s); z)`.

**2D interpolation.** :func:`~W_POD.two_dim.ansatz.interp2d_decay`
(Catmull-Rom inside the box, C0 decay :math:`e^{-15 d_x} e^{-15 d_y}`
outside, toward 0). Its clamp has a custom JVP, so the gradient at a query
exactly on the boundary is 1 rather than ½. The function is wrapped in
``jax.checkpoint`` to reduce memory.

.. note::

   Catmull-Rom has a kink in its gradient at the box edge (at :math:`r = 1`
   for the radial map), where the interpolant switches to the decaying
   extension.

The W term
----------

1D (:meth:`W_POD.ansatz.AnsatzConfig.calc_w1`):

* point cloud of the reconstruction: all modes' pushed-forward nodes
  concatenated, masses normalised to sum 1 (``W2Config.rescale``);
* snapshot: the data mesh, masses ``U_j / sum(U_j)``, with the sum under
  ``stop_gradient``;
* ``ott.solvers.linear.solve`` on a ``PointCloud`` (squared-Euclidean cost),
  :math:`\varepsilon = 0.1`, threshold ``1e-3``, at most ``1e5`` iterations,
  ``recenter_potentials=True``, ``tau_a``/``tau_b`` from
  :class:`~W_POD.ansatz.W2Config` (1 = balanced);
* the returned value is ``reg_ot_cost``, replaced by ``+inf`` when it is
  negative (a diverged Sinkhorn solve), so that SLSQP rejects the step
  instead of accepting a cost of ``-inf``.

2D (:meth:`W_POD.two_dim.losses.Losses.loss_w2_z`):

* masses ``a = |U_j| + pad`` and ``b = max(vals, 0) + pad``, each normalised
  to 1, with ``pad = 1e-6``. The pad keeps every mass positive;
* per snapshot (:meth:`~W_POD.two_dim.losses.Losses.w2_per_z_vmappable`), a
  cost that is not finite or is negative is replaced by ``1e10``, so that
  the L-BFGS-B line search backtracks;
* with ``DOWNSCALED_SIZE`` set (the final runs): both measures and both sets
  of node coordinates are averaged over :math:`f \times f` pixel blocks,
  ``f = H // DOWNSCALED_SIZE`` (20 for the 500×500 grid), and renormalised. The OT solve
  uses threshold ``1e-3`` and ``use_danskin=False``. Without downscaling the
  threshold is ``1e-2`` and OTT's default ``use_danskin`` applies.

Differentiating through Sinkhorn
--------------------------------

In OTT 0.6, ``use_danskin=None`` (the default) means **Danskin** whenever
implicit differentiation is configured, which it is by default. The
potentials are then treated as constants. That gives the exact gradient at a
converged solve, but second derivatives that miss the potentials'
dependence on the input.

========================================== ====================================
Where                                      Mode
========================================== ====================================
1D cost and gradient seen by SLSQP         Danskin (``use_danskin=None``)
1D Hessian probing                         implicit diff, ``use_danskin=False``,
                                           ``ImplicitDiff(nonsym_solver=...,
                                           ridge_kernel=1e-6)``, LU in fp32,
                                           ``GuardedNormalCG`` in fp64
2D cost, gradient and Hessian (downscaled) implicit diff, ``use_danskin=False``,
                                           OTT's default ``ImplicitDiff``
========================================== ====================================

:meth:`~W_POD.ansatz.AnsatzConfig.calc_w1` picks the linear solver of the
implicit-diff solves from the dtype at trace time:
:func:`~W_POD.ansatz.LUSolver` in fp32 and
:func:`~W_POD.guarded_solver.GuardedNormalCG` in fp64. It prints its choice
once per compilation (``calc_w1: dtype ... -> implicit-diff solver ...``).

.. _normalisation:

Loss normalisation
------------------

At the start of every stage:

* 1D (:meth:`~W_POD.ansatz.AnsatzConfig.update_scaling`):
  :math:`L^2_0 = \max(L^2(p_0),\, 10^{-4} S)` with
  :math:`S = \Delta x \sum_j \big(\sum_x U_j(x)^2\big)^{q}`
  (:meth:`~W_POD.ansatz.AnsatzConfig.rel_error_scaling_factor`; :math:`q`
  is ``frobenius_or_l2_squared``, 1 in all final runs), and
  :math:`W_0 = \max(W(p_0),\, 10^{-3})`.
* 2D (:meth:`~W_POD.two_dim.losses.Losses_scaled.reset_scaling`):
  :math:`L^2_0 = L^2(p_0)` and :math:`W_0 = W(p_0)`, with no floor.

Variable scaling (Hessian preconditioning)
------------------------------------------

The optimiser works in scaled variables :math:`y = p / s` with

.. math::

   s_i = \frac{1}{\sqrt{\max\big(|H_{ii}|,\ \rho \max_k |H_{kk}|\big)}},

where :math:`H` is the Hessian of the stage objective at the stage's
starting point. :math:`\rho` is ``1e-6`` in both 1D and 2D, so
:math:`s_{\max}/s_{\min} \le 10^3`. In 2D, :math:`s` is then divided by its
geometric mean.
Bounds become :math:`[\ell/s, u/s]` and linear constraints
:math:`A\,\mathrm{diag}(s)`. The scaling is recomputed for every stage.

The diagonal is computed by *probing*. Each Hessian-vector product
:math:`Hv` is forward-over-reverse (``jvp`` of ``grad``), with :math:`v` an
indicator or ±1 vector on a set of parameters that do not couple, so
:math:`(Hv)_i = H_{ii}` (or :math:`v_i H_{ii}`) can be read off directly.

1D (:func:`~W_POD.hess_diag_probed.calc_hessian_diag_band`):

* **Coefficients.** A trajectory is linear in its coefficients, so two
  coefficients couple only if their basis functions overlap on the
  snapshots. :func:`~W_POD.hess_diag_probed.coeff_colouring` builds that
  overlap graph from the kernel matrices (normalised overlap > ``1e-2``) and
  greedily colours it. One HVP is used per colour.
* **Mode values.** These are coloured ``i mod 41``, with a separate colour
  set per mode (:func:`~W_POD.hess_diag_probed.band_colouring`): 41 HVPs per
  mode. The interpolation stencil couples only neighbouring nodes. The W
  term adds a weak, flat far-field coupling, which bounds the accuracy.

2D (:func:`~W_POD.two_dim.hess_diag_probed_2d.calc_hessian_diag_2d`,
``downscale_factor=20``):

* **L2 part of the mode pixels.** Pixel :math:`(i, j)` of a transported mode
  is coloured :math:`(i \bmod 4, j \bmod 4)` (the Catmull-Rom 4×4 stencil),
  giving 16 colours. A fixed mode uses 1 colour. Within every (block,
  colour) group the probe weights sum to exactly zero. The block-averaged
  W term is invariant along such directions, so the readout is the exact L2
  diagonal.
* **W part of the mode pixels.** The W diagonal is constant within a
  block. It is estimated from one representative pixel per block (the one
  with the smallest L2 diagonal), with random ±1 signs over blocks of the
  same block-colour, :math:`k_b = 3`, so 9 HVPs per mode.
* **Coefficients.** These use the 1D colouring with overlap threshold
  ``1e-3`` (1D: ``1e-2``), and every bias column gets its own colour. The
  number of colours is printed (``CLIM colouring: threshold 0.001, ...``).

Optimisers
----------

* **1D**: ``scipy.optimize.minimize`` with the method from
  :class:`~W_POD.solver.OptimCfg` (``'SLSQP'`` in the final runs), the
  bounds, the linear constraints and ``maxiter``. The ``maxcor=50`` option
  is also passed, but SLSQP ignores it.
* **2D**: L-BFGS-B with bounds only, ``maxcor=50``.

.. warning::

   In SciPy 1.17.1, SLSQP's callback, and therefore the per-iteration
   ``Iter`` log lines, reports the full-step *trial* point, not the accepted
   iterate. Judge a run from the saved parameters, not from the log.

Positivity and other constraints
--------------------------------

1D, per mode (:meth:`~W_POD.modes.Mode.get_constraints` and
``get_param_bounds_arr``):

* mode values :math:`\ge 10^{-6}` (bounds);
* amplitude at every snapshot, as a linear map of the coefficients
  :math:`K a + b`, :math:`\ge 10^{-3}`. Optionally the shift is monotone in
  :math:`z` (``gamma_increasing``/``gamma_decreasing``, not used in the final
  runs);
* the amplitude is also passed through
  :func:`~W_POD.util.frankenstein_softplus`, which is ``max(1e-7, x)``.

2D (bounds only, no linear constraints):

* every mode value and amplitude coefficient :math:`\ge 10^{-6}`;
* every radial coefficient (including the bias) :math:`\ge 10^{-4}`;
* the amplitude is also clipped at 0 inside
  :meth:`~W_POD.two_dim.ansatz.Mode.get_mode`.

Floating point
--------------

The crossing-waves Exp 1 drivers run in fp32 (x64 is not enabled), so their
implicit differentiation uses the LU solver. All other drivers enable x64 and
use ``GuardedNormalCG``.
