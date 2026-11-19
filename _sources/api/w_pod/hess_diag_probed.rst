W_POD.hess_diag_probed
======================

.. py:module:: W_POD.hess_diag_probed

Hessian diagonal by probing, for the ``'hessian'`` variable scaling.

A probe :math:`v` is the 0/1 indicator of one colour class. If no two
indices of the class couple, :math:`(Hv)_i = H_{ii}` for every :math:`i` in
the class. Probes supported only on the coefficients, or only on the mode
values, cannot pick up couplings between the two groups, so each group is
coloured separately.

Measured on the 1600-parameter Exp 1 Hessian (module docstring):

=============================== ======== ==========================
Setting                         HVPs     scaling error med / p90
=============================== ======== ==========================
``k_band=41`` (default)         ~297     0.38 % / 3.07 %
``k_band=99``                   ~413     0.19 % / 1.59 %
=============================== ======== ==========================

Entry points
------------

.. py:function:: calc_hessian_diag_band(cfg, params, *, k_band=41, batch_size=8)

   Probed diagonal of ``cfg._cost(p, L2(params), W(params), use_danskin=False)``
   at ``params``. The two normalisations are evaluated at ``params``, so the
   L2 and W terms are both ≈ 1 there.

   :param cfg: an :class:`~W_POD.ansatz.AnsatzConfig`. Only ``ansatz``,
      ``calc_l2_training``, ``calc_w1_training`` and ``_cost`` are used.
   :param params: parameter vector (unscaled).
   :param int k_band: modulus of the mode-value colouring.
   :param int batch_size: unused by the sequential probe loop.
   :returns: signed diagonal, ``numpy`` array of length ``n``.

.. py:function:: calc_hessian_diag_mc(cfg, params, *, n_samples=256, key=None, batch_size=8)

   As :func:`calc_hessian_diag_band`, but with Hutchinson estimation for the
   mode values. Less accurate at equal cost. Kept as a fallback.

   :returns: ``(diag, info)``.

Building blocks
---------------

.. py:function:: hessian_diag_band(hvp, n, pod_idx, coeff_idx, *, coeff_colours, pod_k=41, pod_blocks=1, pod_per_block=True, batch_size=8)

   Diagonal from an ``hvp(v) -> H v`` callable. Coefficients use
   ``coeff_colours``, mode values use :func:`band_colouring`.

   :returns: ``(diag, info)`` with HVP counts.

.. py:function:: hessian_diag_mc(hvp, n, pod_idx, coeff_idx, *, coeff_colours, n_samples=256, key=None, batch_size=8)

   Same, with Rademacher probes on the mode values.

.. py:function:: coeff_colouring(rbfs, tol=1e-2)

   Colour the coefficient indices from the kernels alone, with no Hessian
   needed. Each trajectory is linear in its coefficients,
   :math:`y = \Phi a` with :math:`\Phi` =
   :meth:`~W_POD.util.RBF_interpol.get_linear_constraint_matrix`. Two
   coefficients are linked when their normalised overlap
   :math:`|\Phi_i^\top \Phi_j| / (\lVert\Phi_i\rVert \lVert\Phi_j\rVert)`
   exceeds ``tol``, and the graph is coloured greedily.

   :param rbfs: list of :class:`~W_POD.util.RBF_interpol` in
      parameter-index order (a dict from ``Mode.get_RBF()`` is accepted).
   :returns: integer colour per coefficient.

.. py:function:: band_colouring(n, k, n_blocks=1, per_block=True)

   ``i mod k`` colouring of ``n`` indices split into ``n_blocks`` equal
   consecutive blocks (one per mode). With ``per_block`` every block gets
   its own ``k`` colours (cost ``n_blocks * k`` HVPs).

   :raises ValueError: ``n`` not divisible by ``n_blocks``.

.. py:function:: colour_from_matrix(S, tol=1e-2, scale=None)

   Colour an explicit coupling matrix (boolean, or numeric thresholded at
   ``tol`` relative to :math:`\min(|S_{ii}|, |S_{jj}|)`). Diagnostic
   helper.

.. py:function:: greedy_colour(adj)

   Largest-degree-first greedy colouring (Welsh–Powell) of a boolean
   adjacency matrix.

.. py:function:: ansatz_layout(ansatz, n)

   ``(pod_idx, coeff_idx, rbfs)`` for an :class:`~W_POD.ansatz.Ansatz`.
   It is derived by passing ``arange(n)`` through ``arr_to_dict``, so it
   follows the ansatz's own layout.

.. py:function:: _cfg_probe_inputs(cfg, params)

   Build the inputs of the probes for an
   :class:`~W_POD.ansatz.AnsatzConfig`: the HVP of
   ``cfg._cost(·, L2(params), W(params), use_danskin=False)`` at
   ``params``, the index layout and the coefficient colouring.

.. py:function:: _probe_coeffs(hvp, n, coeff_idx, coeff_colours, batch_size)

   :func:`_probe_diag` restricted to the coefficients.

.. py:function:: _mc_diag(hvp, n, idx, n_samples, key, batch_size=8)

   Hutchinson estimate on a subspace: Rademacher vectors that are ±1 on
   ``idx`` and 0 elsewhere, batched with ``jax.lax.map``.

.. py:function:: _probe_diag(hvp, n, idx, colours, batch_size=8)

   One jitted HVP per colour, run in a sequential Python loop with a
   ``tqdm`` bar. The batched ``jax.lax.map`` variant is commented out.
