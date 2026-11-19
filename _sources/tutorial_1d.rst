Tutorial: 1D crossing waves
===========================

This walks through ``examples/crossing_waves/exp1/cw_exp1_formulation_B.py``,
the driver of the paper's Exp 1 with Formulation B (regularised
trajectories). ``cw_exp1_formulation_A.py`` differs only in the kernels and
the regularisation weights (see :ref:`tut1d-noreg`). The ``*_budget.py``
variants are identical except for 200 instead of 1000 iterations per stage.
Run them from a folder where ``data/`` and ``img/`` may be written.

Data
----

Two Gaussian pulses cross. Their amplitudes are :math:`|\sin(\pi z/10)|`
and :math:`|\cos(\pi z/10)|`:

.. code-block:: python

   x = jnp.linspace(0., 1., 400, endpoint=False)
   data_z = jnp.linspace(-10, 10., 200)
   scaling_x = 400

   def u(z):
       mu, sigma = 200, 4.
       delta_1 = 0.15*z**3 + 0.8*z + 1.5
       delta_2 = -18*z + 2
       return (jnp.abs(jnp.sin(jnp.pi*z/10.)) * jnp.exp(-((scaling_x*x - mu - delta_1)/sigma)**2)
             + jnp.abs(jnp.cos(jnp.pi*z/10.)) * jnp.exp(-((scaling_x*x - mu - delta_2)/sigma)**2))

   data_exact = jnp.stack([jnp.maximum(u(z), 1e-6) for z in data_z])
   data_exact = data_exact / data_exact.max()

The loss code only reads four attributes from the data object, so a small
holder class is enough:

.. code-block:: python

   class DataHolder:
       def __init__(self, x, data_z, data_exact):
           self.x, self.dx = x, x[1] - x[0]
           self.data_z, self.data_exact = data_z, data_exact

   data = DataHolder(x, data_z, data_exact)

Kernels and ansatz
------------------

One interpolator per kind of trajectory. Both modes share them; an
:class:`~W_POD.util.RBF_interpol` holds no coefficients, only the kernel
matrix.

.. code-block:: python

   RBF_beta  = RBF_interpol(0.2, 1.,  data_z, kernel='matern32', bias=True)  # amplitude
   RBF_gamma = RBF_interpol(3.0, 0.3, data_z, kernel='SE',       bias=True)  # shift

   ansatz = Ansatz([TransportedMode(x, RBF_beta, RBF_gamma),
                    TransportedMode(x, RBF_beta, RBF_gamma)])

Each :class:`~W_POD.modes.TransportedMode` contributes ``pod_local`` (400
values), ``beta_coeff`` (201) and ``gamma_coeff`` (201), so the parameter
vector has :math:`2 \times 802 = 1604` entries.

Loss configuration
------------------

.. code-block:: python

   cfg = AnsatzConfig(
       data,
       LossWeights(l2=1., w2=1., penalty_exceed=0., reg_norm_pod=0.,
                   normalize=True, frobenius_or_l2_squared=1.),
       W2Config(tau_a=1., tau_b=1., rescale=True),       # balanced OT
       ansatz,
       OptimPhaseGradConfig(ansatz, [
           {'beta_coeff': {'l2': True, 'w1': True},
            'gamma_coeff': {'l2': True, 'w1': True},
            'pod_local': {'l2': True, 'w1': True}}] * 2),
       OptimPhaseCoeffConfig(ansatz, [
           {'beta_coeff': 1e-4, 'gamma_coeff': 1e-5}] * 2),   # lambda per trajectory
       variable_scaling_method='hessian',
   )

* :class:`~W_POD.ansatz.OptimPhaseGradConfig` decides, per mode, parameter
  and loss term, whether gradients flow. Here every parameter sees both
  terms.
* :class:`~W_POD.ansatz.OptimPhaseCoeffConfig` holds the RKHS weights
  :math:`\lambda`.
* ``variable_scaling_method='hessian'`` makes every stage start with a
  probed Hessian diagonal (:doc:`numerics`).

Solver
------

Three SLSQP stages with the same configuration:

.. code-block:: python

   optim_cfg = [OptimCfg(True, True, True, 1000, 'SLSQP'),
                OptimCfg(True, True, True, 1000, 'SLSQP'),
                OptimCfg(True, True, True, 1000, 'SLSQP')]
   solver = IteratedSolver(data, [cfg, cfg, cfg], optim_cfg, n_iter=1)

The positional arguments of :class:`~W_POD.solver.OptimCfg` are
``stop_if_linesearch_fails`` (stored, unused), ``verbose``, ``log_w1`` and
``maxiter``, then the SciPy method.

Before anything else, the driver replaces ``W_POD.hess_diag_probed._probe_diag``
by a batched version: all Hessian-vector products of one probe go through
``jax.lax.map(hvp, V, batch_size=8)`` instead of one call per colour, and the
batch is halved on a GPU out-of-memory error. The probed diagonal is the
same; only the run time changes.

Initial guess
-------------

Ten trials share the starting amplitudes and shifts. They differ in the
width of the starting modes: both modes are a Gaussian centred at 0.5 with
width :math:`w = 0.01 \cdot 5^{u}`, :math:`u \sim U[-1, 1]` drawn per trial
(log-uniform between 0.002 and 0.05):

.. code-block:: python

   init_gamma_1 = -(1 + data_z + 0.01*data_z**3) / 399.
   init_gamma_2 = -(1 - data_z) / 399.

   def pod_init(i):
       k_mu, k_w = jax.random.split(jax.random.fold_in(jax.random.key(1044), i))
       w = 0.01 * 5. ** (2. * jax.random.uniform(k_w, (2,)) - 1.)
       return jnp.maximum(1e-3, jnp.exp(-((x - 0.5) / w[0])**2))   # same width for both modes

   pod_local_init = pod_init(i)
   init = ansatz.dict_to_arr([
       {'beta_coeff': RBF_beta.get_coeff(jnp.ones(200)),
        'gamma_coeff': RBF_gamma.get_coeff(init_gamma_1),
        'pod_local': pod_local_init},
       {'beta_coeff': RBF_beta.get_coeff(jnp.ones(200)),
        'gamma_coeff': RBF_gamma.get_coeff(init_gamma_2),
        'pod_local': pod_local_init}])

:meth:`~W_POD.util.RBF_interpol.get_coeff` turns values at the snapshots
into interpolation coefficients. It solves :math:`(K + 10^{-3} I)a = y -
\bar y` and stores the mean :math:`\bar y` as the bias.

Running and output
------------------

.. code-block:: python

   casename = 'cw1_formulation_B'
   solver.solve_scipy(init, directory_out='data/%s/every_%d/' % (casename, i))
   vis = Visualizer(data, solver)
   vis.plot_training(solver.params_result, filename='img/%s/training_%d.png' % (casename, i))
   vis.plot_pod_and_coeff(solver.params_result, filename='img/%s/pod_%d.png' % (casename, i))
   vis.plot_hist(filename='img/%s/hist_%d.png' % (casename, i))

The driver wraps ``solve_scipy`` in ``try``/``except``: a trial that raises
prints ``TRIAL CRASHED`` and the loop continues with the next one.

:meth:`~W_POD.solver.IteratedSolver.solve_scipy` writes
``params_0.hdf5`` (the initial vector only) and ``params_k.hdf5`` after
stage ``k``. The file layout is given in :ref:`hdf5-1d`. To read one back:

.. code-block:: python

   import h5py
   with h5py.File('data/cw1_formulation_B/every_0/params_3.hdf5') as f:
       p = jnp.array(f['params_arr'])
   per_mode = ansatz.arr_to_dict(p)            # [{'pod_local': ..., 'beta_coeff': ..., 'gamma_coeff': ...}, ...]
   rec = ansatz.get_values_grid(data_z[17], p) # reconstruction of snapshot 17
   rel = jnp.sqrt(cfg.calc_l2_training(p) / cfg.rel_error_scaling_factor())

``rel`` is the relative error in the ``rel:`` column of the log. With
``frobenius_or_l2_squared=1`` it equals
:math:`\lVert D - U\rVert_F / \lVert U\rVert_F`.

.. _tut1d-noreg:

Formulation A (no regularisation)
---------------------------------

``cw_exp1_formulation_A.py``:

.. code-block:: python

   RBF_beta  = RBF_interpol(1e-6, 1., data_z, kernel='matern32', bias=False)
   RBF_gamma = RBF_interpol(1e-6, 1., data_z, kernel='SE',       bias=False)
   # OptimPhaseCoeffConfig: beta_coeff = gamma_coeff = 0.

With :math:`\theta = 10^{-6}` the kernel matrix is the identity, so every
coefficient is one snapshot's value (divided by :math:`1 + 10^{-3}`, the
nugget), and the trajectories are not regularised. Without the bias, each
:class:`~W_POD.modes.TransportedMode` has :math:`400 + 200 + 200 = 800`
parameters, 1600 in total.
