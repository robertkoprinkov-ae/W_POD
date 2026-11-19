Known issues and pitfalls
=========================

Behaviour of the code that can surprise a user. None of these affect
the final runs as configured unless stated.

JIT and configuration objects
-----------------------------

* :meth:`W_POD.ansatz.AnsatzConfig.calc_w1`, ``calc_w1_training`` and
  ``_cost`` are jitted with ``self`` **static**. JAX hashes the config by
  identity and reads its attributes (loss weights, ``W2Config``, gradient
  masks, RKHS weights) at trace time. Changing an attribute of an existing
  :class:`~W_POD.ansatz.AnsatzConfig` after the first call has no effect.
  Build a new config instead.
* :class:`W_POD.two_dim.losses.Losses_scaled` re-jits its cost and gradient
  in every :meth:`~W_POD.two_dim.losses.Losses_scaled.reset_scaling`, so
  each stage starts with a recompilation.

Broken or unused code paths (1D)
--------------------------------

* :meth:`W_POD.solver.L2_W1_optimizer.calc_gradients_per_param` calls
  ``ansatz_cfg.cost_grad``, which does not exist. Since
  :meth:`W_POD.visualizer.Visualizer.plot_pod_and_coeff` calls it, always
  give the :class:`~W_POD.visualizer.Visualizer` an
  :class:`~W_POD.solver.IteratedSolver`, even for a single stage. That
  variant returns the gradient with respect to the *scaled* variables,
  evaluated at the vector it is given.
* :meth:`W_POD.solver.IteratedSolver.solve` calls ``solver.solve``, which
  does not exist. Use :meth:`~W_POD.solver.IteratedSolver.solve_scipy`.
* :meth:`W_POD.solver.IteratedSolver.filename` and
  :meth:`~W_POD.solver.IteratedSolver.description_pretty` refer to a
  missing ``solvers_cfg`` attribute.
  :meth:`W_POD.solver.L2_W1_optimizer.filename` indexes the
  :class:`~W_POD.ansatz.OptimPhaseGradConfig` object directly, which raises
  ``TypeError``.
* ``OptimCfg.stop_if_linesearch_fails`` is stored but never read.
* :meth:`~W_POD.ansatz.AnsatzConfig.calc_exceed_training` squares a value
  that is already squared. Its weight is 0 in all final runs.
* When ``LossWeights.w2 == 0``, ``_cost`` normalises the L2 term by the
  attribute ``l2_standard`` rather than by its ``normalize_l2`` argument.
* ``ParameterRescaling.reset_scaling`` builds ``jnp.diag(hess)``, an
  :math:`n \times n` matrix, only to print its extremes.
* ``maxcor=50`` is passed to SLSQP, which ignores it (SciPy may warn about
  an unknown option).

Broken or fragile code paths (2D)
---------------------------------

* :meth:`W_POD.two_dim.ansatz.Mode.get_reg` reads ``amplitude_coeff`` even
  when ``amplitude_fixed=True``, where it does not exist. This matters only
  with ``reg_lambda > 0``.
* :meth:`~W_POD.two_dim.losses.Losses_scaled.reset_scaling` calls the
  Hessian probe with ``downscale_factor=20``, which is correct only for
  ``H // DOWNSCALED_SIZE == 20`` (500×500 grid, ``DOWNSCALED_SIZE=25``).
  On another grid, change it to match, or the probe raises (grid not
  divisible) or silently uses the wrong blocks.
* The probe assumes the downscaled W term and an inactive ``max(vals, 0)``
  in :meth:`~W_POD.two_dim.losses.Losses.loss_w2_z`, i.e. all node values
  positive. The bounds guarantee the latter for the mode values and the
  amplitude coefficients.
* :meth:`W_POD.two_dim.visu.Visualizer.plot_recon` assumes exactly two
  coordinate frames.
* :meth:`W_POD.two_dim.losses.Losses.cost_sum` and ``cost_sum_grad`` accept
  ``full`` but ignore it.
* In the per-iteration log of
  :meth:`~W_POD.two_dim.losses.Losses_scaled.optimize`, the
  ``frobenius/l2`` field prints :math:`L^2 \cdot L^2_0` and has no useful
  meaning. The ``w2`` field and the final number (relative Frobenius error)
  are correct.

Interpreting logs and results
-----------------------------

* SLSQP's callback (SciPy 1.17.1) receives the full-step **trial** point,
  not the accepted iterate. A 1D ``Iter`` line can therefore show a loss
  that the run never accepted. Judge runs from the saved ``params_*.hdf5``.
* The ``rel`` value in the 1D log is
  :math:`\sqrt{L^2/S}`, the relative Frobenius error only when
  ``frobenius_or_l2_squared = 1``.
* Runs are not bit-reproducible, on GPU or on CPU, so reruns of the same
  job drift apart after a while. The CPU adv-diff reruns from a fresh
  install, for example, give the same first stage bit for bit and the same
  final reconstruction errors to within rounding.
* ``params_0.hdf5`` written by
  :meth:`~W_POD.solver.IteratedSolver.solve_scipy` contains only
  ``params_arr``. Later files also contain the per-mode groups and the
  configuration (:ref:`hdf5-1d`).
* With ``fix_center=True`` the shift coefficients of the 2D transformation
  stay in the parameter vector with zero gradient and zero curvature. Their
  Hessian scaling is set by the floor.
* The Catmull-Rom extension has a gradient kink at the box edge. For the
  radial map this happens at :math:`r = 1`.

Reproducibility
---------------

* Results depend on the package versions, in particular of ``jax``,
  ``ott-jax``, ``lineax`` and ``scipy``. Install from ``requirements_cpu.txt``
  or ``requirements_gpu.txt`` to match the paper's runs (:doc:`versions`).
* Output file names depend only on the case name (and, for the wildland fire
  drivers, the seed). Running the same driver again in the same folder
  overwrites the previous ``data/``, ``img/`` or ``wildfire/`` output. Copy
  results you want to keep.
