W_POD.ansatz
============

.. py:module:: W_POD.ansatz

The ansatz (a list of modes), the configuration of one optimisation stage,
the objective, and the variable scaling.

Ansatz
------

.. py:class:: Ansatz(modes)

   A sum of modes with a block-structured parameter vector.

   :param list modes: :class:`~W_POD.modes.Mode` instances. All must share
      the same mesh.

   .. py:method:: n_param()

      Length of the full parameter vector.

   .. py:method:: n_coeff()

      Total number of trajectories over all modes.

   .. py:method:: get_param_pytree()

      List with one set of parameter names per mode. Checked against
      :class:`OptimPhaseGradConfig`.

   .. py:method:: get_coeff_pytree()

      List with one set of coefficient (trajectory) names per mode. Checked
      against :class:`OptimPhaseCoeffConfig`.

   .. py:method:: split_params_arr_by_mode(params_arr)

      List of per-mode blocks.

   .. py:method:: arr_to_dict(params_arr)

      Flat vector → list (one entry per mode) of dicts keyed by parameter
      name.

   .. py:method:: dict_to_arr(params_dict_per_mode)

      Inverse of :meth:`arr_to_dict`: concatenate the per-mode dicts into
      the flat vector.

   .. py:method:: get_bounds_arr()

      Concatenated ``(lb, ub)`` of all modes.

   .. py:method:: get_constraints()

      All modes' linear constraints, padded into the full vector.

      :returns: ``scipy.optimize.LinearConstraint`` with
         ``keep_feasible=False``.

   .. py:method:: get_values_grid(z, params_arr)

      Reconstruction on the data mesh: the sum of the modes'
      :meth:`~W_POD.modes.Mode.get_values_grid`.

   .. py:method:: get_values_nodes(z, params_arr)

      Concatenated point clouds ``(positions, masses)`` of all modes, for the
      W term.

   .. py:method:: get_values_grid_exact(sol_dict_per_mode)

      Reconstruction from trajectory values (synthetic data).

   .. py:method:: write_hdf5(params_arr, hdf5_group)

      Write ``params_arr`` and one group ``mode_<i>`` per mode.

Stage configuration
-------------------

.. py:class:: LossWeights(l2, w2, penalty_exceed, reg_norm_pod, normalize, frobenius_or_l2_squared=2.)

   Weights of the objective's terms.

   :param float l2: :math:`w_{L^2}`.
   :param float w2: :math:`w_W`. If ``0`` the W term is not computed.
   :param float penalty_exceed: weight of the overshoot penalty
      :meth:`AnsatzConfig.calc_exceed_training`.
   :param float reg_norm_pod: weight of
      :meth:`AnsatzConfig.calc_pod_norms_reg`.
   :param bool normalize: divide the L2 and W terms by their stage-start
      values.
   :param float frobenius_or_l2_squared: exponent :math:`q` applied to each
      snapshot's squared :math:`L^2` error before summing. All final runs
      pass ``1.``, giving :math:`\sum_j \Delta x\lVert D_j - U_j\rVert^2`.
      **The default 2 is not what the runs use.**

   .. py:method:: write_hdf5(grp)

      Store the weights as ``weight_l2``, ``weight_w2``,
      ``weight_penalty_exceed``, ``weight_reg_norm_pod``,
      ``weight_normalize`` and ``weight_frobenius_or_l2_squared`` in the
      HDF5 group ``grp``.

.. py:class:: W2Config(tau_a, tau_b, rescale)

   Settings of the OT term.

   :param float tau_a: OTT marginal relaxation of the reconstruction (1 = balanced).
   :param float tau_b: same for the snapshot.
   :param bool rescale: normalise the reconstruction's masses to sum 1.

.. py:class:: OptimPhaseGradConfig(ansatz, grad_cfg)

   Per mode, per parameter and per loss term: does the parameter receive
   gradient from that term?

   :param grad_cfg: list (one per mode) of
      ``{param_name: {'l2': bool, 'w1': bool}}``.
   :raises AssertionError: the keys do not match the ansatz.

   Implemented with ``jax.lax.stop_gradient`` in
   :meth:`AnsatzConfig.cut_grad`. A parameter excluded from both terms
   still moves through the regularisation and the constraints.

   .. py:attribute:: grad_cfg

      The configuration list passed to the constructor.

   .. py:method:: write_hdf5(grp)

      One boolean dataset ``<mode>_<param>_<term>`` per entry.

.. py:class:: OptimPhaseCoeffConfig(ansatz, coeff_cfg)

   RKHS-norm weights :math:`\lambda`.

   :param coeff_cfg: list (one per mode) of ``{coeff_name: float}``.
   :raises AssertionError: the keys do not match the ansatz's coefficient
      names.

   .. py:attribute:: coeff_cfg

      The configuration list passed to the constructor.

   .. py:method:: write_hdf5(grp)

      One dataset ``<mode>_<coeff>`` per weight.

AnsatzConfig
------------

.. py:class:: AnsatzConfig(data, weights_loss, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg, variable_scaling_method='identity')

   Everything that defines one optimisation stage's objective.

   :param data: data object (``x``, ``dx``, ``data_z``, ``data_exact``).
   :param LossWeights weights_loss:
   :param W2Config w2_config:
   :param Ansatz ansatz:
   :param OptimPhaseGradConfig ansatz_grad: must refer to the same ``ansatz``.
   :param OptimPhaseCoeffConfig ansatz_coeff_reg: must refer to the same ``ansatz``.
   :param str variable_scaling_method: ``'identity'`` or ``'hessian'``.

   .. note::

      The cost methods are jitted with ``self`` static. Do not change the
      attributes of a config after its first evaluation.

   .. py:attribute:: l2_standard

      Stage normalisation :math:`L^2_0` of the L2 term. ``None`` until
      :meth:`update_scaling` has run.

   .. py:attribute:: w2_standard

      Stage normalisation :math:`W_0` of the W term. ``None`` until
      :meth:`update_scaling` has run.

   .. py:attribute:: rescale_params

      The :class:`ParameterRescaling` of this config.

   .. rubric:: Loss terms

   .. py:method:: calc_l2(z, params_arr, sol_exact)

      :math:`\Delta x \sum_x (D(p;z) - U)^2` for one snapshot.

   .. py:method:: calc_l2_training(params_arr)

      :math:`\sum_j \ell_j^{\,q}` with :math:`\ell_j` = :meth:`calc_l2` of
      snapshot :math:`j`, vmapped over the snapshots.

   .. py:method:: calc_w1(z, params_arr, sol_exact, return_full=False, use_danskin=None)

      Entropic OT cost (``reg_ot_cost``) between the reconstruction's point
      cloud and the snapshot (see :doc:`../../numerics`), or ``+inf`` when
      that cost is negative (a diverged Sinkhorn solve).

      :param use_danskin: ``False`` forces implicit differentiation
         (``ridge_kernel=1e-6``) with a linear solver chosen from the dtype
         at trace time: :func:`LUSolver` in fp32,
         :func:`~W_POD.guarded_solver.GuardedNormalCG` in fp64. Anything else
         uses OTT's default, which is Danskin.
      :param bool return_full: also return the OTT output object.

   .. py:method:: calc_w1_training(params_result, use_danskin=None)

      Sum of :meth:`calc_w1` over the snapshots (power 1, no square root).

   .. py:method:: calc_w1_div(z, params_arr, sol_exact, return_full=False)

      Sinkhorn *divergence* alternative to :meth:`calc_w1`. Not used.

   .. py:method:: calc_reg(params_arr, verbose=False)

      :math:`\sum_c \lambda_c\, a_c^\top K_c a_c`. With ``verbose`` it also
      returns the per-mode breakdown.

   .. py:method:: calc_exceed(z, params_arr, sol_exact)

      Overshoot penalty of one snapshot,
      :math:`\Delta x \sum \max(D - U, 0)^2`.

   .. py:method:: calc_exceed_training(params_arr)

      Sum over the snapshots of :meth:`calc_exceed` **squared**, so the
      overshoot enters to the fourth power.

   .. py:method:: calc_pod_norms_reg(params_arr)

      :math:`\sum_i (\Delta x \lVert\varphi_i\rVert^2 - 1)^2`.

   .. py:method:: cut_grad(params_arr, loss_term_name)

      Copy of ``params_arr`` with ``stop_gradient`` on the blocks that
      :class:`OptimPhaseGradConfig` excludes from ``loss_term_name``
      (``'l2'`` or ``'w1'``).

   .. rubric:: Objective

   .. py:method:: _cost(params_arr, normalize_l2, normalize_w2, use_danskin=None)

      The stage objective in unscaled variables:

      .. code-block:: text

         w_l2 * L2(cut_grad(p,'l2')) / normalize_l2
       + w_w2 * W (cut_grad(p,'w1')) / normalize_w2
       + calc_reg(p) + w_exceed * exceed(p) + w_pod * pod_norm_reg(p)

      If ``weights_loss.w2 == 0`` the W term is skipped and L2 is divided by
      :attr:`l2_standard`.

   .. py:method:: _cost_scaled(params_arr, norm_l2, norm_w1, scaling_x)

      ``_cost(params_arr * scaling_x, norm_l2, norm_w1)`` (jitted).

   .. py:method:: cost_jit_scaled(params_arr)

      Objective in the **scaled** variables :math:`y = p / s` that the
      optimiser sees, using the current :attr:`l2_standard`,
      :attr:`w2_standard` and scaling.

   .. py:method:: cost_jit_scaled_grad(params_arr)

      Gradient of :meth:`cost_jit_scaled` with respect to the scaled
      variables.

   .. py:method:: rel_error_scaling_factor()

      :math:`S = \Delta x \sum_j (\sum_x U_j^2)^{q}`. With :math:`q = 1`
      this is the squared Frobenius norm of the data, so
      :math:`\sqrt{L^2/S}` is the relative error.

   .. py:method:: update_scaling(params_arr)

      Start-of-stage normalisation:
      :math:`L^2_0 = \max(L^2(p), 10^{-4} S)`,
      :math:`W_0 = \max(W(p), 10^{-3})`. Then calls
      :meth:`ParameterRescaling.reset_scaling`.

   .. py:method:: get_bounds_arr()

      The ansatz bounds mapped to the scaled variables, ``(lb/s, ub/s)``.

   .. py:method:: get_constraints()

      The ansatz linear constraints mapped to the scaled variables
      (:math:`A \to A\,\mathrm{diag}(s)`).

   .. py:method:: write_hdf5(params_arr, file)

      Ansatz blocks, plus a group ``ansatz_cfg`` with the loss weights and
      the ``grad`` and ``reg`` subgroups.

ParameterRescaling
------------------

.. py:class:: ParameterRescaling(ansatz_cfg, cost, calc_l2_training, calc_w1_training, method='identity')

   Diagonal change of variables :math:`p = s \odot y`.

   :param str method: ``'identity'`` (:math:`s = 1`) or ``'hessian'``.

   .. py:attribute:: scaling_x

      The vector :math:`s`.

   .. py:method:: reset_scaling(x0)

      With ``'hessian'``: compute the Hessian diagonal :math:`h` of
      ``ansatz_cfg._cost`` at ``x0`` with
      :func:`~W_POD.hess_diag_probed.calc_hessian_diag_band`, then set
      :math:`s = 1/\sqrt{\max(|h|, 10^{-6}\max|h|)}`, so
      :math:`s_{\max}/s_{\min} \le 10^3`. No geometric-mean normalisation
      is applied.

   .. py:method:: scaled_to_unscaled(x)

      :math:`p = x \cdot s`.

   .. py:method:: unscaled_to_scaled(x)

      :math:`y = x / s`.

   .. py:method:: unscaled_to_scaled_bounds(lb, ub)

      :math:`(lb/s,\ ub/s)`. Infinite bounds stay infinite.

   .. py:method:: unscaled_to_scaled_constraints(constraints)

      New ``LinearConstraint`` with :math:`A\,\mathrm{diag}(s)` and the
      same bounds.

   .. py:method:: _hess_diag_exact(cost, params, norm_l2, norm_w1)

      Exact diagonal, one HVP per parameter (``use_danskin=False``).
      Diagnostic, not called.

   .. py:method:: _hess_exact(cost, params, norm_l2, norm_w1)

      Full Hessian, row by row. Diagnostic, not called.

   .. py:method:: _hess_diag(f, *args, key, argnums=0, n_samples=256, chunk=8)

      Hutchinson (Rademacher) estimate of the diagonal. Not called.

.. py:function:: LUSolver(**kwargs)

   Factory returning ``lineax.LU()``. It accepts and ignores the
   ``rtol``/``atol`` that OTT passes. Used as ``nonsym_solver`` of the
   implicit differentiation in fp32.
