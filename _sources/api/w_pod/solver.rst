W_POD.solver
============

.. py:module:: W_POD.solver

Drive SciPy over one or several stages.

.. py:class:: OptimCfg(stop_if_linesearch_fails, verbose, log_w1, maxiter, scipy_optim_method, callback_prefix='')

   Options of one stage.

   :param bool stop_if_linesearch_fails: stored, not used.
   :param bool verbose: print an ``Iter`` line per callback.
   :param bool log_w1: also evaluate and log the W term in the callback.
      This costs an extra OT solve per iteration.
   :param int maxiter: SciPy ``maxiter``.
   :param str scipy_optim_method: e.g. ``'SLSQP'`` (final runs),
      ``'L-BFGS-B'`` (bounds only), ``'trust-constr'``.
   :param str callback_prefix: prefix of the log lines.

.. py:class:: L2_W1_optimizer(data, ansatz_cfg, optim_cfg)

   One stage.

   .. py:method:: solve_scipy(initial)

      #. :meth:`~W_POD.ansatz.AnsatzConfig.update_scaling` at ``initial``
         (loss normalisation and Hessian scaling);
      #. ``scipy.optimize.minimize`` on
         :meth:`~W_POD.ansatz.AnsatzConfig.cost_jit_scaled` with its
         gradient, scaled bounds, scaled linear constraints and
         ``options={'maxiter': maxiter, 'maxcor': 50}``;
      #. stores :attr:`result` (the ``OptimizeResult``) and
         :attr:`params_result` (unscaled).

      The callback logs per iteration::

         Iter k; error: <objective>; l2: <L2> (norm: <L2_0>; normalized: ...);
         w1: <W> (norm: <W_0>; normalized: ...;) reg: <reg>; rel: <sqrt(L2/S)>

      and prints the per-mode regularisation every 50 iterations.

      .. warning:: With SLSQP the callback sees the trial point
         (:doc:`../../caveats`).

   .. py:attribute:: result

      SciPy ``OptimizeResult`` of the stage (in scaled variables).

   .. py:attribute:: params_init

      Initial vector of the stage.

   .. py:attribute:: _log_l2

      Weighted L2 term per callback.

   .. py:attribute:: _log_w1

      Weighted W term per callback (only if ``log_w1``).

   .. py:attribute:: _log_tot

      Objective per callback.

   .. py:attribute:: params_result

      Final parameter vector of the stage (unscaled).

   .. py:method:: calc_gradients_per_param(params_arr)

      Broken: calls the nonexistent ``ansatz_cfg.cost_grad``. Use the
      :class:`IteratedSolver` version.

   .. py:method:: filename()

      Broken: indexes :class:`~W_POD.ansatz.OptimPhaseGradConfig` directly.

   .. py:method:: write_hdf5(params=None, filename='data/output/params.hdf5')

      Writes the stage's loss history (group ``loss``: ``l2``, ``w1``,
      ``tot``, ``norm_l2``, ``norm_w2``), the parameters and the
      configuration (:ref:`hdf5-1d`).

   .. py:method:: description_pretty()

      Human-readable list of which parameters see which term.

   .. py:method:: renormalize_losses(params_arr)

      Alias of :meth:`~W_POD.ansatz.AnsatzConfig.update_scaling`.

.. py:class:: IteratedSolver(data, ansatz_cfgs, optim_cfgs, n_iter)

   Run a list of stages, ``n_iter`` times over. Each stage starts from the
   previous stage's result.

   :param list ansatz_cfgs: one :class:`~W_POD.ansatz.AnsatzConfig` per
      stage, all with the same ansatz. The same object may be repeated.
   :param list optim_cfgs: one :class:`OptimCfg` per stage.
   :param int n_iter: number of passes over the list.

   .. py:method:: solve_scipy(initial=None, directory_out=None)

      Run all stages. With ``directory_out`` (which must end with ``/``) it
      writes ``params_0.hdf5`` (the initial vector, ``params_arr`` only) and,
      after stage ``k``, ``params_k.hdf5`` through
      :meth:`L2_W1_optimizer.write_hdf5`. It also prints SciPy's
      ``success``, ``message``, ``status`` and ``nit`` per stage.

   .. py:method:: solve(initial=None)

      Broken: calls ``L2_W1_optimizer.solve``, which does not exist.

   .. py:method:: filename()

      Broken: refers to a missing ``solvers_cfg`` attribute.

   .. py:method:: description_pretty()

      Broken: refers to a missing ``solvers_cfg`` attribute.

   .. py:attribute:: solvers

      One :class:`L2_W1_optimizer` per stage.

   .. py:attribute:: params_result

      Final parameter vector (unscaled), i.e. the last stage's result.

   .. py:attribute:: params_hist

      List of the parameter vectors after each stage.

   .. py:method:: calc_gradients_per_param(params_arr)

      Gradient of each stage's scaled objective, split per mode and
      parameter. Used by the visualiser.

   .. py:method:: write_hdf5_every(directory='data/output/params_every/')

      Re-write every stage's vector after the run.

.. _hdf5-1d:

HDF5 layout (``params_k.hdf5``, k ≥ 1)
--------------------------------------

.. code-block:: text

   /                      attrs: description
   /params_arr            full parameter vector (unscaled)
   /mode_<i>/pod_local    (or pod_global)
   /mode_<i>/beta_coeff   (alpha_coeff / gamma_coeff / gamma_shift_coeff ...)
   /loss/{l2,w1,tot}      per-callback history of the stage (weighted)
   /loss/norm_l2, norm_w2 stage normalisations L2_0, W_0
   /ansatz_cfg/weight_*   LossWeights
   /ansatz_cfg/grad/<mode>_<param>_<term>   OptimPhaseGradConfig
   /ansatz_cfg/reg/<mode>_<coeff>           OptimPhaseCoeffConfig
