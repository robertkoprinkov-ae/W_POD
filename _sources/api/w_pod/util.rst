W_POD.util
==========

.. py:module:: W_POD.util

Kernel interpolation of the trajectories, the amplitude floor, and a
synthetic-data generator.

.. py:function:: frankenstein_softplus(x)

   Positivity map applied to every amplitude (:math:`\alpha`, :math:`\beta`)
   before it multiplies a mode.

   This is simply ``jnp.maximum(1e-7, x)``. An earlier version, the identity
   above 0.01 with a smooth positive floor below, is kept in a comment in
   the source. The name is historical.

   :param x: array of amplitude values.
   :returns: ``max(1e-7, x)`` elementwise.

RBF_interpol
------------

.. py:class:: RBF_interpol(theta, sigma, data_z, bias=False, nugget=1e-3, kernel='SE')

   Kernel interpolant on the snapshot parameters,

   .. math::

      y(z) = b + \sum_{j=1}^{n_z} a_j\, k(z, z_j).

   The object stores only the kernel and the matrix
   :math:`K_{ij} = k(z_i, z_j)`. The coefficients
   :math:`(a_1, \dots, a_{n_z}[, b])` live in the parameter vector and are
   passed to each method.

   :param float theta: length scale :math:`\theta`. ``1e-6`` makes
      :math:`K = I` (a lookup table: one free value per snapshot).
   :param float sigma: amplitude; the kernel is scaled by :math:`\sigma^2`.
   :param data_z: snapshot parameters, shape ``(n_z,)`` or ``(n_z, d)``.
   :param bool bias: append a constant term :math:`b` as the last
      coefficient. It is excluded from the RKHS norm.
   :param float nugget: default diagonal regularisation in
      :meth:`get_coeff`.
   :param str kernel: ``'SE'`` (squared exponential,
      :math:`\sigma^2 e^{-r^2/(2\theta^2)}`), ``'matern52'`` or
      ``'matern32'``. :math:`r` is the Euclidean distance in :math:`z`.
   :raises TypeError: unknown kernel name.

   .. py:attribute:: _K

      The :math:`n_z \times n_z` kernel matrix (no nugget).

   .. py:attribute:: theta

      Length scale :math:`\theta`.

   .. py:attribute:: sigma

      Kernel amplitude :math:`\sigma`.

   .. py:attribute:: data_z

      Snapshot parameters the interpolant is built on.

   .. py:attribute:: dim_z

      Dimension of :math:`z` (1 for a 1-d array).

   .. py:attribute:: bias

      Whether the last coefficient is a constant term.

   .. py:attribute:: nugget

      Default nugget of :meth:`get_coeff`.

   .. py:attribute:: kernel

      Kernel name (``'SE'``, ``'matern52'`` or ``'matern32'``).

   .. py:method:: n_param()

      :returns: :math:`n_z`, plus 1 if ``bias``.

   .. py:method:: predict(z, coeff, variances=False)

      Evaluate the interpolant.

      :param z: one parameter value or an array of them (shape ``(m,)`` for
         1-d :math:`z`, ``(m, d)`` otherwise).
      :param coeff: coefficient vector of length :meth:`n_param`.
      :param bool variances: also return the GP posterior variance
         :math:`k(z,z) - k_*^\top (K + 10^{-6} I)^{-1} k_*`.
      :returns: values (and variances), shaped like ``z`` without the last
         axis.

   .. py:method:: get_coeff(values, nugget=None)

      Coefficients that interpolate ``values`` at ``data_z``. Solves
      :math:`(K + \text{nugget}\, I)\, a = y - \bar y` and sets
      :math:`b = \bar y` when ``bias`` is on, or solves
      :math:`(K + \text{nugget}\, I)\, a = y` otherwise.

      :param values: values at the snapshots, shape ``(n_z,)``.
      :param nugget: overrides the instance default.
      :returns: coefficient vector of length :meth:`n_param`.

   .. py:method:: calc_RKHS_reg(coeff)

      RKHS norm :math:`a^\top K a` of the interpolant. The bias is
      dropped.

   .. py:method:: get_linear_constraint_matrix()

      Matrix :math:`\Phi` (shape ``(n_z, n_param)``) with
      :math:`\Phi\,\text{coeff} = y(z_j)`: :math:`K`, plus a column of ones
      for the bias. Used for positivity constraints at the snapshots and
      for the coefficient colouring of the Hessian probe.

   .. py:method:: get_monotonic_linear_constraints_matrix()

      Matrix (``(n_z-1, n_param)``) of successive differences
      :math:`y(z_{j+1}) - y(z_j)`, without the bias column. Used for
      monotone shifts.

   .. py:method:: rbf_kernel(z_1, z_2)

      Kernel between a (possibly vector) ``z_1`` and a single ``z_2``.

   .. py:method:: rbf_kernel_scalars(z_1, z_2)

      Kernel between two single points. Bound at construction to one of
      ``_rbf_kernel_scalars_SE``, ``_rbf_kernel_scalars_matern52`` or
      ``_rbf_kernel_scalars_matern32``.

   .. py:method:: _calc_K()

      Build :attr:`_K` with a double ``vmap`` of :meth:`rbf_kernel`.

DataCreation
------------

.. py:class:: DataCreation(x, ansatz, ansatz_values, data_z, RBF, noise=False)

   Generates synthetic snapshots from known trajectories, used by the unit
   tests and the early experiments. Exp 1 builds its data directly instead.

   :param x: 1D mesh.
   :param ansatz: a :class:`W_POD.ansatz.Ansatz`.
   :param ansatz_values: one dict per mode, mapping ``'<name>_value'``
      (e.g. ``'beta_value'``, ``'gamma_value'``) and the mode-vector name
      (e.g. ``'pod_local'``) to a callable of :math:`z`.
   :param data_z: snapshot parameters.
   :param RBF: dict from coefficient name to the
      :class:`RBF_interpol` used for it. Fills :attr:`coeffs_exact`.
   :param noise: unused.

   .. py:attribute:: data_exact

      Snapshots, shape ``(n_z, N)``, floored at ``1e-3``.

   .. py:attribute:: dx

      Mesh spacing ``x[1] - x[0]``.

   .. py:attribute:: coeffs_exact

      Exact interpolation coefficients per coefficient name.

   .. py:method:: get_solution(z)

      The exact snapshot at ``z``, built with
      :meth:`W_POD.ansatz.Ansatz.get_values_grid_exact`.

The data interface
------------------

Any object passed as ``data`` to :class:`~W_POD.ansatz.AnsatzConfig` and the
solvers needs only ``x`` (mesh), ``dx``, ``data_z`` and ``data_exact``
(``(n_z, N)``). The visualiser also calls ``get_solution(z)`` when no exact
snapshot is passed.

