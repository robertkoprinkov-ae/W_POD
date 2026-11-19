W_POD.two_dim.losses
====================

.. py:module:: W_POD.two_dim.losses

Loss terms (:class:`Losses`) and the optimisation driver with variable
scaling (:class:`Losses_scaled`).

Losses
------

.. py:class:: Losses(pts, dx, data_z, data_exact, ansatz, l2_or_l1=1, DOWNSCALED_SIZE=None, reg_lambda=0.)

   :param pts: grid nodes, ``(n_y * n_x, 2)``.
   :param dx: grid spacing (the same in x and y).
   :param data_z: snapshot parameters.
   :param data_exact: snapshots, ``(n_z, H, W)``.
   :param ansatz: a :class:`W_POD.two_dim.ansatz.Ansatz`.
   :param int l2_or_l1: exponent applied to each snapshot's squared error
      before summing (1 in the final runs).
   :param DOWNSCALED_SIZE: if set, compute the W term on
      ``DOWNSCALED_SIZE²`` block averages (25 in the final runs).
   :param float reg_lambda: weight of ``ansatz.get_reg`` in the objective
      (0 in the final runs).

   .. py:attribute:: bounds

      ``ansatz.get_bounds()`` (unscaled).

   .. py:method:: loss_l2(params, full=False)

      :math:`\sum_j \big(\Delta x^2 \sum (U_j - D_j)^2\big)^{q}` with
      :math:`q` = ``l2_or_l1``,
      mapped over the snapshots with ``jax.lax.map`` (batch 50). With
      ``full`` it returns the per-snapshot values.

   .. py:method:: loss_w2_z(params, z_i, data_exact_flat, downscaled_size=None)

      The OTT output for snapshot ``z_i``. The masses are
      ``|U| + pad`` and ``max(vals, 0) + pad`` with ``pad = 1e-6``, each
      normalised to 1. With ``downscaled_size``, masses and coordinates are
      block-averaged by ``H // downscaled_size`` and renormalised, and the
      solve uses threshold ``1e-3`` and ``use_danskin=False``. Without it,
      threshold ``1e-2`` and OTT's default ``use_danskin``. In both cases
      :math:`\varepsilon = 0.1` and at most ``1e5`` iterations.

   .. py:method:: loss_w2(params, downscaled_size=None, full=False)

      :math:`\sum_j` ``reg_ot_cost`` of :meth:`loss_w2_z` (vmapped).

   .. py:method:: get_l2_scaling_factor()

      :math:`\Delta x^2 \sum U^2` (for ``l2_or_l1 = 1``), so
      ``sqrt(loss_l2 / factor)`` is the relative Frobenius error.

   .. py:method:: _cost_sum(params, norm_l2, norm_w1, data_exact_flat, full=False)

      ``W / norm_w1 + L2 / norm_l2 + reg_lambda * ansatz.get_reg(params)``.
      With ``full`` it returns ``(l2, l2/norm_l2, w2, w2/norm_w1)`` per
      snapshot.

   .. py:method:: cost_sum(params, norm_l2, norm_w1, full=False)

      Jitted :meth:`_cost_sum` on the stored data (``full`` is ignored).

   .. py:method:: cost_sum_grad(params, norm_l2, norm_w1, full=False)

      Jitted gradient of :meth:`_cost_sum` with respect to ``params``
      (``full`` is ignored).

   .. py:method:: _loss_l2(params, data_exact_flat, full=False)

      :meth:`loss_l2` on explicitly passed data.

   .. py:method:: compute_l2_per_z_vmappable(z_i, params, data_exact_flat_row)

      :math:`\Delta x^2 \sum (U_j - D_j)^2` for snapshot ``z_i``.

   .. py:method:: _loss_w2(params, data_exact_flat, downscaled_size=None, full=False)

      :meth:`loss_w2` on explicitly passed data.

   .. py:method:: w2_per_z_vmappable(z_i, params, data_exact_flat, downscaled_size)

      ``reg_ot_cost`` of :meth:`loss_w2_z` for one snapshot (power 1). A
      cost that is not finite or is negative (a diverged Sinkhorn solve) is
      replaced by ``1e10``, so that the L-BFGS-B line search backtracks.

   .. py:method:: _downscale_array(arr, factor=2)

      Average non-overlapping ``factor × factor`` blocks of a stack of 2D
      arrays ``(N, H, W)``.

      :raises ValueError: ``H`` or ``W`` not divisible by ``factor``.

   .. py:attribute:: data_exact_flat

      Snapshots flattened to ``(n_z, H * W)``.

   .. py:attribute:: n_coord_frames

      Number of coordinate frames of the ansatz.

Losses_scaled
-------------

.. py:class:: Losses_scaled(losses, params_init, data_exact, scaling='gradient')

   Optimise :class:`Losses` in rescaled variables. The constructor already
   calls :meth:`reset_scaling` at ``params_init``.

   :param Losses losses:
   :param params_init: starting vector (unscaled).
   :param data_exact: snapshots, ``(n_z, H, W)``.
   :param str scaling: ``'hessian'`` (final runs), ``'gradient'`` or
      ``'identity'``.

   .. py:method:: reset_scaling(params)

      #. ``norm_l2 = loss_l2(params)`` and
         ``norm_w1 = loss_w2(params, DOWNSCALED_SIZE)`` (no floor);
      #. the scaling:

         * ``'hessian'``: :math:`h` from
           :func:`~W_POD.two_dim.hess_diag_probed_2d.calc_hessian_diag_2d`
           (``downscale_factor=20``, unfloored), then
           :math:`s = 1/\sqrt{\max(|h|, 10^{-6}\max|h|)}`, divided by its
           geometric mean;
         * ``'gradient'``: two scalars, the inverse gradient norms over the
           mode pixels and over everything else;
         * ``'identity'``: :math:`s = 1`;

      #. re-jit the scaled cost and gradient.

   .. py:method:: optimize(params_init, maxiter, debug=False)

      One stage. :meth:`reset_scaling` at ``params_init``, then
      ``scipy.optimize.minimize(method='L-BFGS-B', options={'maxiter': maxiter, 'maxcor': 50})``
      with the bounds mapped to the scaled variables (infinite bounds kept).
      Prints one line per iteration and the time per iteration.

      :returns: the unscaled result, plus the ``OptimizeResult`` if
         ``debug``.

   .. py:method:: convert_x(x)

      Scaled → unscaled: :math:`x \cdot s` for ``'hessian'``, otherwise
      the two gradient-norm factors applied to the mode pixels and to the
      rest.

   .. py:method:: convert_x_inv(x)

      Inverse of :meth:`convert_x` (unscaled → scaled).

   .. py:method:: cost(x)

      Objective at the scaled vector ``x`` (JAX).

   .. py:method:: cost_grad(x)

      Gradient of :meth:`cost` with respect to the scaled variables (JAX).

   .. py:method:: cost_np(x)

      :meth:`cost` for NumPy input and output, as passed to SciPy. Prints
      ``W2PATCH: guarded cost ...`` when the value is not finite or is at
      least ``1e6``, i.e. when the guard of :meth:`Losses.w2_per_z_vmappable`
      fired.

   .. py:method:: cost_grad_np(x)

      :meth:`cost_grad` for NumPy input and output, as passed to SciPy. A
      gradient with non-finite entries is replaced by zeros (and a
      ``W2PATCH: non-finite gradient ...`` line is printed), so that the
      L-BFGS-B line search backtracks instead of stopping.

   .. py:method:: _cost_graddable(x, data_exact_flat)

      ``losses._cost_sum(convert_x(x), norm_l2, norm_w1, data_exact_flat)``,
      with the normalisations and scaling of the current stage.

   .. py:method:: mask_modal_variables(x)

      ``x`` with everything except the mode pixels set to 0.

   .. py:method:: mask_remainder(x)

      ``x`` with the mode pixels set to 0.

   .. py:method:: _hess_diag(f, *args, key, argnums=0, n_samples=256, chunk=8)

      Hutchinson (Rademacher) estimate of the Hessian diagonal. Not
      called.

   .. py:attribute:: norm_l2

      L2 normalisation of the current stage, :math:`L^2(p_0)`.

   .. py:attribute:: norm_w1

      W normalisation of the current stage, :math:`W(p_0)`.

   .. py:attribute:: scaling_x

      Scaling vector :math:`s` (``'hessian'`` only).

   .. py:attribute:: n_iter

      Iteration counter, cumulative over stages.

   .. py:attribute:: hist

      Objective value per iteration, cumulative over stages.

   .. py:attribute:: hist_grad_modes

      Gradient norm over the mode pixels per iteration (scaled variables).

   .. py:attribute:: hist_grad_remainder

      Gradient norm over all other parameters per iteration.

   .. py:attribute:: hist_params_rescaling

      Parameter vector at each :meth:`reset_scaling`.

   .. py:attribute:: iter_rescaling

      Value of :attr:`n_iter` at each :meth:`reset_scaling`.
