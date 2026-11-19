W_POD.two_dim.ansatz
====================

.. py:module:: W_POD.two_dim.ansatz

The 2D ansatz is a hierarchy:

* :class:`Ansatz` is a sum over **coordinate frames**;
* a :class:`CoordinateFrame` is one coordinate transformation plus several
  modes that it transports together;
* a transformation is a :class:`CoordinateTransformation` (radial scaling
  plus rigid shift) or an :class:`IdentityCoordinateTransformation`;
* a :class:`Mode` is a pixel field times an amplitude trajectory.

Every level has the same parameter-vector methods (``n_param``,
``get_bounds``, ``split_params``, ``split_params_recursive``,
``get_params``, ``get_params_recursive``, ``get_reg``). They are listed
for each class below.

Interpolation
-------------

.. py:function:: interp2d_decay(xq, yq, x, y, f, k=1.0, asymptote=0.0, n_chunks=None)

   Interpolate the grid function ``f`` (shape ``(len(x), len(y))``) at
   ``(xq, yq)``. Inside the box this is exactly
   ``interpax.interp2d(..., method='catmull-rom')``. Outside, the value at
   the nearest boundary point decays toward ``asymptote`` as
   :math:`e^{-k s_x} e^{-k s_y}`, where :math:`s_x, s_y` are the distances
   to the box.

   The method (``'catmull-rom'``) and the boundary join (C0, value only) are
   fixed inside the function. The function is wrapped in
   ``jax.checkpoint``.

   :param xq: query x-coordinates.
   :param yq: query y-coordinates (same shape as ``xq``).
   :param x: grid x-coordinates (strictly increasing).
   :param y: grid y-coordinates (strictly increasing).
   :param f: values on the grid, shape ``(len(x), len(y))``.
   :param float k: decay rate outside the box (the code passes 15).
   :param float asymptote: value far outside the box.
   :param n_chunks: process the queries in this many sequential chunks
      (``jax.lax.map``) to cut peak memory.
   :returns: values, same shape as ``xq``.

.. py:function:: clamp(v, lo, hi)

   ``jnp.clip`` with a custom JVP whose gradient is 1 (not ½) for points
   exactly on the boundary.

.. py:function:: _core_c0(xq, yq, x, y, f, method, k, a)

   C0 decay: one ``interp2d`` pass at the clamped point, times
   :math:`e^{-k s_x} e^{-k s_y}`. Used by :func:`interp2d_decay`.

Coordinate transformations
--------------------------

.. py:class:: IdentityCoordinateTransformation()

   The identity map, with no parameters.

   .. py:method:: n_param()

      0.

   .. py:method:: get_bounds()

      ``(lb, ub)``, two empty arrays.

   .. py:method:: get_reg(params)

      0.

   .. py:method:: split_params(params)

      Empty dict.

   .. py:method:: split_params_recursive(params)

      Empty dict.

   .. py:method:: get_params(params_split)

      Empty array.

   .. py:method:: get_params_recursive(params_split)

      Empty array.

   .. py:method:: transform(pts, z, params)

      A copy of ``pts``.

   .. py:method:: inv_transform(pts, z, params)

      A copy of ``pts``.

   .. py:method:: plot_coeff(*args)

      Does nothing.

.. py:class:: CoordinateTransformation(n_discret_theta, rbf_radial_shifts, rbf_shift_x, rbf_shift_y, fix_center=False)

   Radial scaling about the centre of the grid, followed by a rigid shift:

   .. math::

      p' = c + s(z) + r(\vartheta; z)\,(p - c), \qquad
      \vartheta = \operatorname{atan2}(p_y - c_y, p_x - c_x) \bmod 2\pi.

   :math:`r(\cdot; z)` is interpolated linearly and periodically between
   ``n_discret_theta`` equally spaced stations
   :math:`\vartheta_k = 2\pi k / n`. The value of station :math:`k` is the
   trajectory ``rbf_radial_shifts.predict(z, radial_k_coeff)`` and is used
   directly as a scaling *factor*. The centre :math:`c` is the centre of the
   bounding box of the points passed in, i.e. of the full grid in every call
   made by the code.

   :param int n_discret_theta: number of azimuthal stations (24 in the
      final runs).
   :param rbf_radial_shifts: interpolator shared by all stations.
   :param rbf_shift_x: interpolator of the rigid shift :math:`s_x`.
   :param rbf_shift_y: interpolator of the rigid shift :math:`s_y`.
   :param bool fix_center: if true, :math:`s \equiv 0`. The shift
      coefficients stay in the parameter vector but have no effect.

   .. py:attribute:: theta_discret

      Station angles :math:`\vartheta_k`.

   .. py:attribute:: param_names

      ``['radial_0_coeff', …, 'radial_{n-1}_coeff', 'shift_x_coeff', 'shift_y_coeff']``,
      the block order in the parameter vector.

   .. py:method:: n_param()

      ``n_discret_theta * rbf_radial_shifts.n_param() + rbf_shift_x.n_param() + rbf_shift_y.n_param()``.

   .. py:method:: get_bounds()

      ``(lb, ub)``: every radial coefficient (including its bias)
      :math:`\ge 10^{-4}`; the shift coefficients are free.

   .. py:method:: get_reg(params)

      Unweighted sum of the RKHS norms of all stations and both shifts.
      The weight is ``Losses.reg_lambda``.

   .. py:method:: split_params(params)

      Dict from each name in :attr:`param_names` to its block.

   .. py:method:: split_params_recursive(params)

      Same as :meth:`split_params` (this is a leaf).

   .. py:method:: get_params(params_split)

      Inverse of :meth:`split_params`.

   .. py:method:: get_params_recursive(params_split)

      Same as :meth:`get_params`.

   .. py:method:: transform(pts, z, params)

      Forward map of the points ``pts`` (shape ``(m, 2)``), used for the W
      point cloud.

   .. py:method:: inv_transform(pts, z, params)

      Exact inverse, :math:`p = c + (p' - c - s)/r(\vartheta')`, with
      :math:`\vartheta'` the azimuth of :math:`p' - c - s`. A radial scaling
      preserves the azimuth. Used to pull the data grid back for the L2
      term.

   .. py:method:: get_shift_center(pts, z, params)

      :math:`c + s(z)`, the moving centre (plotted as a red cross).

   .. py:method:: plot_coeff(params, data_z, filename)

      Polar plots of :math:`r(\vartheta)` per snapshot and the shift
      trajectories. Intended for a handful of snapshots.

Modes
-----

.. py:class:: Mode(n_pts, amplitude_fixed, rbf_amplitudes=None)

   A pixel field :math:`\varphi` (``n_pts`` values, row-major
   :math:`n_y \times n_x`) with amplitude :math:`\max(0, a(z))`.

   :param int n_pts: number of pixels.
   :param bool amplitude_fixed: if true, no amplitude trajectory
      (:math:`a \equiv 1`).
   :param rbf_amplitudes: interpolator of :math:`a(z)`.

   .. py:attribute:: param_names

      ``['mode', 'amplitude_coeff']``, or ``['mode']`` if the amplitude is
      fixed.

   .. py:method:: n_param()

      ``n_pts``, plus ``rbf_amplitudes.n_param()`` unless the amplitude is
      fixed.

   .. py:method:: get_bounds()

      ``(lb, ub)`` with **every** entry :math:`\ge 10^{-6}`, mode values
      and amplitude coefficients alike.

   .. py:method:: split_params(params)

      Dict ``{'mode': ..., 'amplitude_coeff': ...}``.

   .. py:method:: split_params_recursive(params)

      Same as :meth:`split_params` (this is a leaf).

   .. py:method:: get_params(params_split)

      Inverse of :meth:`split_params`.

   .. py:method:: get_params_recursive(params_split)

      Same as :meth:`get_params`.

   .. py:method:: get_mode(z, params)

      :math:`\max(0, a(z))\,\varphi`, or :math:`\varphi` if the amplitude
      is fixed.

   .. py:method:: get_reg(params)

      RKHS norm of the amplitude trajectory. Fails if the amplitude is
      fixed (:doc:`../../caveats`).

   .. py:method:: plot_coeff(vis, params, ax)

      Draw the pixel field in ``ax[0]`` and the amplitude trajectory in
      ``ax[1]``.

Coordinate frames
-----------------

.. py:class:: CoordinateFrame(pts, x_crds, y_crds, data_z, coordinate_transformation, modes, fixed=False)

   A transformation shared by several modes.

   :param pts: grid nodes, shape ``(n_y * n_x, 2)``, in ``meshgrid`` order.
   :param x_crds: 1D x-coordinates (length ``n_x``).
   :param y_crds: 1D y-coordinates (length ``n_y``).
   :param data_z: snapshot parameters.
   :param coordinate_transformation: :class:`CoordinateTransformation` or
      :class:`IdentityCoordinateTransformation`.
   :param list modes: :class:`Mode` objects.
   :param bool fixed: treat the frame as stationary even if the
      transformation is not the identity.

   A frame is *transported* unless ``fixed`` is set or its transformation
   is the identity. Only transported frames are interpolated, and only
   their nodes move in the W term.

   .. py:attribute:: param_names

      ``['coordinate_transformation', 'mode_0', 'mode_1', …]``, the block
      order in the parameter vector.

   .. py:attribute:: n_x

      Number of grid points in x.

   .. py:attribute:: n_y

      Number of grid points in y.

   .. py:method:: n_param()

      Transformation parameters plus all modes' parameters.

   .. py:method:: get_bounds()

      ``scipy.optimize.Bounds`` of the frame's block.

   .. py:method:: get_reg(params)

      Transformation RKHS norms plus every mode's amplitude RKHS norm.

   .. py:method:: split_params(params)

      Dict from each name in :attr:`param_names` to its flat block.

   .. py:method:: split_params_recursive(params)

      Nested dict: ``{'coordinate_transformation': {...}, 'mode_k': {'mode': ..., 'amplitude_coeff': ...}}``.

   .. py:method:: get_params(params_split)

      Inverse of :meth:`split_params`.

   .. py:method:: get_params_recursive(params_split)

      Inverse of :meth:`split_params_recursive`.

   .. py:method:: calc_values(z, params_by_name)

      Sum of the frame's modes on its own nodes (untransformed).

   .. py:method:: get_value_grid_arbitrary(params, z)

      The frame's contribution on the data grid at parameter ``z``. For a
      transported frame this is :meth:`calc_shift_interpax`. Otherwise it
      is :meth:`calc_values`.

   .. py:method:: get_value_grid(params, z_i)

      :meth:`get_value_grid_arbitrary` at snapshot index ``z_i``.

   .. py:method:: get_value_nodes(params, z_i)

      ``(positions, values)`` for the W term: the forward-transformed nodes
      (or the fixed grid) and the summed mode values.

   .. py:method:: calc_shift_interpax(pts, z, params)

      Pull ``pts`` back with ``inv_transform`` and evaluate the summed
      modes there with :func:`interp2d_decay` (``k=15``). This is the path
      used by the final code.

   .. py:method:: calc_shift(pts, z, params)

      Alternative with a linear ``RegularGridInterpolator`` and zero fill
      outside. Not used.

   .. py:method:: plot_coeff(vis, params, filename_prefix)

      Write ``<prefix>_coord.png`` (transformation) and
      ``<prefix>_modes.png`` (modes and amplitudes).

Ansatz
------

.. py:class:: Ansatz(coordinate_frames, data_z)

   Sum of coordinate frames.

   :param list coordinate_frames: :class:`CoordinateFrame` objects.
   :param data_z: snapshot parameters.

   .. py:attribute:: param_names

      ``['coord_frame_0', 'coord_frame_1', …]``.

   .. py:method:: n_param()

      Sum over the frames.

   .. py:method:: get_bounds()

      ``scipy.optimize.Bounds`` of the full vector.

   .. py:method:: get_reg(params)

      Unweighted sum of all RKHS norms.

   .. py:method:: split_params(params)

      Dict from ``coord_frame_i`` to that frame's flat block.

   .. py:method:: split_params_recursive(params)

      Nested dict down to the leaves: ``{'coord_frame_i':
      {'coordinate_transformation': {...}, 'mode_k': {'mode': ...,
      'amplitude_coeff': ...}}}``.

   .. py:method:: get_params(params_split)

      Inverse of :meth:`split_params`.

   .. py:method:: get_params_recursive(params_split)

      Inverse of :meth:`split_params_recursive`.

   .. py:method:: get_value_grid_arbitrary(params, z)

      Reconstruction on the data grid at parameter ``z`` (flat, length
      ``n_y * n_x``): the sum over the frames.

   .. py:method:: get_value_grid(params, z_i)

      :meth:`get_value_grid_arbitrary` at snapshot index ``z_i``.

   .. py:method:: get_value_nodes(params, z_i)

      Concatenated point clouds of all frames, for the W term. With
      ``K`` frames the result has ``K * n_y * n_x`` points.
