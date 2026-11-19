W_POD.modes
===========

.. py:module:: W_POD.modes

One class per kind of mode. A mode owns its mesh and its kernel
interpolators. It converts its slice of the parameter vector to and from a
dict, and evaluates itself either on the data mesh (for the L2 term) or as a
transported point cloud (for the W term).

Mode (interface)
----------------

.. py:class:: Mode

   Abstract base class. Every method below raises ``NotImplementedError``
   except :meth:`n_coeff` and :meth:`n_pod`.

   .. py:method:: n_param()

      Length of this mode's block in the parameter vector.

   .. py:method:: n_coeff()

      Number of trajectories (``len(get_coeff_names())``).

   .. py:method:: n_pod()

      Number of mode-vector parameters (``len(get_pod_names())``).

   .. py:method:: get_mode_name()

      Class name as a string, used in plots.

   .. py:method:: get_param_names()

      Names of all parameter blocks.

   .. py:method:: get_coeff_names()

      Names of the blocks that are kernel coefficients (trajectories in
      :math:`z`). These are the ones regularised in the RKHS norm.

   .. py:method:: get_pod_names()

      Names of the blocks that hold the mode vector.

   .. py:method:: get_RBF()

      Dict from coefficient name to its :class:`~W_POD.util.RBF_interpol`.

   .. py:method:: get_param_bounds_arr()

      ``(lb, ub)`` arrays of length :meth:`n_param`.

   .. py:method:: get_constraints()

      ``(A, lb, ub)`` of linear constraints :math:`lb \le A p \le ub` on
      this mode's block.

   .. py:method:: calc_coeff(coeff_name, z, coeff, softplus=True, variances=False)

      Evaluate one trajectory at ``z``. Amplitudes go through
      :func:`~W_POD.util.frankenstein_softplus` when ``softplus`` is true.
      With ``variances`` the GP variance is returned as well.

   .. py:method:: get_values_grid(z, params_arr)

      The mode's contribution on the data mesh at parameter ``z`` (pulled
      back and interpolated). Used by the L2 term.

   .. py:method:: get_values_nodes(z, params_arr)

      ``(positions, masses)``: the mode's own nodes after the forward
      transformation, and the mode values times the amplitude. Used by the
      W term.

   .. py:method:: get_values_grid_exact(sol_dict)

      As :meth:`get_values_grid`, but from trajectory *values*
      (``'beta_value'``, …) instead of coefficients. Uses linear
      interpolation with zero outside the mesh. Used for synthetic data.

   .. py:method:: arr_to_dict(params_arr)

      This mode's block → dict keyed by :meth:`get_param_names`.

   .. py:method:: dict_to_arr(params_dict)

      Inverse of :meth:`arr_to_dict`. The output length must equal
      :meth:`n_param`.

   .. py:method:: write_hdf5(params_arr, hdf5_group)

      Write each named block as a dataset of ``hdf5_group``.

GlobalMode
----------

.. py:class:: GlobalMode(x, RBF_alpha)

   Stationary mode :math:`\alpha(z)\,\varphi(x)`.

   :param x: mesh, shape ``(N,)``.
   :param RBF_alpha: interpolator of :math:`\alpha`.

   * Parameters: ``pod_global`` (N), ``alpha_coeff``.
   * Bounds: ``pod_global`` :math:`\ge 10^{-6}`.
   * Constraints: :math:`10^{-6} \le \alpha(z_j) \le 100` at every snapshot.

TransportedMode
---------------

.. py:class:: TransportedMode(x, RBF_beta, RBF_gamma, gamma_increasing=False, gamma_decreasing=False)

   Shifted mode :math:`\beta(z)\,\varphi(x - \gamma(z))`. This is the mode
   used in all 1D final runs.

   :param x: mesh, shape ``(N,)``.
   :param RBF_beta: interpolator of the amplitude :math:`\beta`.
   :param RBF_gamma: interpolator of the shift :math:`\gamma`.
   :param bool gamma_increasing: constrain :math:`\gamma(z_{j+1}) \ge \gamma(z_j)`.
   :param bool gamma_decreasing: constrain :math:`\gamma(z_{j+1}) \le \gamma(z_j)`.
      At most one of the two may be set.

   * Parameters: ``pod_local`` (N), ``beta_coeff``, ``gamma_coeff``. The
     flat block is ordered ``pod_local``, ``beta_coeff``, ``gamma_coeff``.
   * :meth:`~Mode.get_values_grid`:
     ``frankenstein_softplus(beta(z)) * interp_decay(x - gamma(z), x, pod_local,
     method='catmull-rom', decay_rate=15)``.
   * :meth:`~Mode.get_values_nodes`: positions ``x + gamma(z)``, masses
     ``frankenstein_softplus(beta(z)) * pod_local``.
   * Bounds: ``pod_local`` :math:`\ge 10^{-6}`. The coefficients are free.
   * Constraints: :math:`\beta(z_j) \ge 10^{-3}` (linear in the
     coefficients), plus the optional monotonicity of :math:`\gamma`.

StretchedMode
-------------

.. py:class:: StretchedMode(x, RBF_beta, RBF_gamma_shift, RBF_gamma_stretch)

   Shifted and stretched mode. The forward map is

   .. math::

      x' = \gamma_s(z) + c + \gamma_r(z)\,(x - c), \qquad c = (\max x + \min x)/2,

   and the mode is evaluated on the mesh through the inverse map, using
   :func:`~W_POD.interp.interp_decay_knots` (``cubic2``, padding 0.1,
   extrapolation 0).

   * Parameters: ``pod_local`` (N), ``beta_coeff``, ``gamma_shift_coeff``,
     ``gamma_stretch_coeff``.
   * Bounds: ``pod_local`` :math:`\ge 10^{-6}`.
   * Constraints: :math:`0.5 \le \beta(z_j) \le 100`.

   Not used in the final runs.
