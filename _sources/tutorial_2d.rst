Tutorial: 2D wildfire
=====================

This walks through ``examples/wildland_fire/formulation_B_4_2.py``: four
transported and two stationary modes, with a regularised radial map
(Formulation B). The six drivers differ only as follows:

========================== ===================================================
Variant                    Change
========================== ===================================================
``formulation_A_*``        radial kernel length scale ``1e-6`` instead of ``10``
``*_1_1``                  1 transported + 1 stationary mode
``*_8_4``                  8 transported + 4 stationary modes
========================== ===================================================

Each driver takes the seed of the random start as its only argument, e.g.
``python formulation_B_4_2.py 22``, and is run from a folder that contains
``Wildlandfire_2d/`` (input) and ``wildfire/`` (output).

Data
----

.. code-block:: python

   jax.config.update('jax_enable_x64', True)

   grid      = jnp.load('Wildlandfire_2d/1D_Grid_full_burela.npy')   # (2, 500)
   data_z    = jnp.load('Wildlandfire_2d/data_z_full_burela.npy')    # (250, 2)
   snapshots = jnp.load('Wildlandfire_2d/data_full_burela.npy')      # (5, 50, 2, 500, 500)
   snapshots = snapshots / snapshots.max()

   time_idx = jnp.linspace(0., snapshots.shape[1]-1, 50).round().astype(int)
   data_exact = snapshots[:, time_idx, 0].reshape((-1, 500, 500))   # (250, 500, 500)
   data_z = data_z.reshape((5, 50, 2))[:, time_idx].reshape((-1, 2))

The run fits 5 cases × 50 time steps of the first field, 250 snapshots in
all. :math:`z` is two-dimensional, and its second column is time. The
physical grid is replaced by the unit square, and the nodes are listed row
by row (``meshgrid`` order, :math:`n_y \times n_x`):

.. code-block:: python

   x_crds = jnp.linspace(0., 1., 500); y_crds = jnp.linspace(0., 1., 500)
   dx = x_crds[1] - x_crds[0]
   xx, yy = jnp.meshgrid(x_crds, y_crds)
   pts = jnp.concatenate((xx.reshape((-1, 1)), yy.reshape((-1, 1))), axis=1)

Ansatz
------

The 2D ansatz is a list of :class:`~W_POD.two_dim.ansatz.CoordinateFrame`
objects. Each frame has one coordinate transformation that all of its
:class:`~W_POD.two_dim.ansatz.Mode` objects share:

.. code-block:: python

   rbf_radial    = RBF_interpol(10.,  1., data_z, nugget=1e-6, bias=True)   # noreg: 1e-6
   rbf_shift_x   = RBF_interpol(1e-6, 1., data_z, nugget=1e-6, bias=False)
   rbf_shift_y   = RBF_interpol(1e-6, 1., data_z, nugget=1e-6, bias=False)
   rbf_amplitude = RBF_interpol(1e-6, 1., data_z, nugget=1e-6, bias=False)

   transported = CoordinateFrame(
       pts, x_crds, y_crds, data_z,
       CoordinateTransformation(24, rbf_radial, rbf_shift_x, rbf_shift_y, fix_center=True),
       [Mode(pts.shape[0], False, rbf_amplitude) for _ in range(4)])
   stationary = CoordinateFrame(
       pts, x_crds, y_crds, data_z, IdentityCoordinateTransformation(),
       [Mode(pts.shape[0], False, rbf_amplitude) for _ in range(2)], fixed=True)
   ansatz = Ansatz([transported, stationary], data_z)

* :class:`~W_POD.two_dim.ansatz.CoordinateTransformation` scales every point
  radially about the domain centre. The factor :math:`r(\vartheta; z)` is
  interpolated periodically between 24 azimuthal stations, and each station
  is a kernel trajectory in :math:`z`. With ``fix_center=True`` the rigid
  shift is fixed at zero and gets no gradient. Its length scale then only
  affects the cost of the Hessian probe, not the result.
* Each mode stores one value per pixel and an amplitude trajectory. With
  :math:`\theta = 10^{-6}` and no bias, the amplitude is a free value per
  snapshot.

Initial guess
-------------

* Every mode starts as the same narrow Gaussian at the centre
  (:math:`\theta = 0.05` in both directions).
* Every radial factor starts at 1 (identity transformation), the shifts
  at 0.
* Amplitudes start from GP draws (SE kernel, :math:`\theta = 200`,
  :math:`\sigma = 0.2`, seed from the command line). Mode 0 is replaced by a
  ramp from 0 to 1
  in time. The other transported modes get 1 + draw, and the stationary
  modes 1 − ramp + draw. All amplitudes are then clipped at 0 and normalised
  to sum to 1 per snapshot.
* The assembled vector is clipped into
  :meth:`~W_POD.two_dim.ansatz.Ansatz.get_bounds`, because the radial
  coefficients would otherwise start at 0 < 1e-4 and the mode tails below
  1e-6.

Nested dictionaries are flattened with ``get_params_recursive``:

.. code-block:: python

   x0 = ansatz.get_params({
       'coord_frame_0': transported.get_params_recursive(coord_frame_params_0),
       'coord_frame_1': stationary.get_params_recursive(coord_frame_params_1)})
   bounds = ansatz.get_bounds()
   x0 = jnp.clip(x0, bounds.lb, bounds.ub)

Losses and optimisation
-----------------------

.. code-block:: python

   losses = Losses(pts, dx, data_z, data_exact, ansatz, l2_or_l1=1, DOWNSCALED_SIZE=25)
   losses_scaled = Losses_scaled(losses, x0, data_exact, scaling='hessian')

   params = losses_scaled.optimize(x0, 20)
   params = losses_scaled.optimize(params, 200)
   params = losses_scaled.optimize(params, 600)
   params = losses_scaled.optimize(params, 1000)
   params = losses_scaled.optimize(params, 2000)

* ``DOWNSCALED_SIZE=25``: the W term is computed on 25×25 block averages
  (20×20 pixels per block) of both measures, which keeps the OT problem
  small.
* ``reg_lambda`` keeps its default 0, so the RKHS norm is not penalised.
  Regularisation comes only from the radial length scale.
* Each :meth:`~W_POD.two_dim.losses.Losses_scaled.optimize` call is one
  stage. It renormalises both loss terms, recomputes the probed Hessian
  diagonal at its starting point, and runs L-BFGS-B for at most
  ``maxiter`` iterations.

Output
------

.. code-block:: python

   with h5py.File('wildfire/seed_reg_4_2_s%d.hdf5' % seed, 'w') as f:   # noreg: seed_noreg_...
       f['params'] = params
       f['hist_grad_modes'] = jnp.array(losses_scaled.hist_grad_modes)
       f['hist_grad_remainder'] = jnp.array(losses_scaled.hist_grad_remainder)

``hist_grad_modes`` and ``hist_grad_remainder`` hold, per iteration, the
norm of the gradient (in scaled variables) over the mode pixels and over
everything else. The driver code after ``exit()`` is dead.

To evaluate a stored vector:

.. code-block:: python

   l2 = losses.loss_l2(params)                               # sum_j dx^2 ||D_j - U_j||^2
   rel = jnp.sqrt(l2 / losses.get_l2_scaling_factor())       # relative Frobenius error
   rec_17 = ansatz.get_value_grid(params, 17).reshape((500, 500))
   parts = ansatz.split_params_recursive(params)             # nested dict by frame / mode
