W_POD.visualizer
================

.. py:module:: W_POD.visualizer

Matplotlib plots of 1D results. Every method writes a PNG (``dpi=300``)
and closes the figure.

.. py:class:: Visualizer(data, solver)

   :param data: the data object.
   :param solver: an :class:`~W_POD.solver.IteratedSolver` (recommended) or
      an :class:`~W_POD.solver.L2_W1_optimizer`. The L2/W values in the
      plots use the first stage's configuration.

   .. py:method:: plot_training(params_arr, samples=None, filename='img/training.png')

      Reconstruction against data for at most 10 evenly spaced training
      snapshots. Each panel shows the total, the individual modes, and the
      snapshot's L2 and W values with the OT convergence flag.

   .. py:method:: plot_test(params_arr, test_z, test_data, samples=True, filename='img/test.png')

      Same for held-out snapshots.

   .. py:method:: compare_exact(z, params_arr, sol_exact=None, ax=None, samples=True, legend=True, filename='img/compare.png')

      One panel. If ``samples`` is set, 100 gray reconstructions are added:
      each trajectory value at ``z`` is drawn independently from a normal
      with the interpolant as mean and the GP posterior variance, and the
      reconstruction is built with ``get_values_grid_exact`` (linear
      interpolation).

   .. py:method:: plot_pod_and_coeff(params_arr, filename='img/pod_gamma.png')

      Top row: mode vectors. Bottom row: every trajectory on
      :math:`z` ± 10 % of the data range, without the amplitude floor.

      Requires an :class:`~W_POD.solver.IteratedSolver`, because it calls
      :meth:`~W_POD.solver.IteratedSolver.calc_gradients_per_param`.

   .. py:method:: plot_hist(filename='img/hist.png')

      Total loss, L2 and (total − L2) over the callback history of the
      solver.

   .. py:method:: _plot_pod(pod_name, params_dict, gradients_per_param_dict, ax)

      One mode vector, plus for each stage the vector moved a small step
      against that stage's gradient (step scaled to at most 10 % of the
      mode's maximum).

   .. py:method:: _plot_coeff(z_discret, calc_values, coeff, label, ax, gradients_coeff=None, plot_variances=False)

      One trajectory on ``z_discret`` with its values at the snapshots.
      Optionally adds ±2 standard deviations and gradient steps.

   .. py:method:: _plot_samples(z, params_dict_per_mode, ax, n_samples=100)

      The gray sample reconstructions of :meth:`compare_exact`.

   .. py:method:: plot_OT_training(pod_global=None, pod_local=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, filename='img/OT_training.png')

      Legacy plot of the OT transport plans. Broken: it calls helpers
      (``_fill_none``, ``get_prediction``, ``calc_w_1``) that no longer
      exist.

   .. py:method:: plot_OT_local(mat, pod_local, beta_value, exact, ax)

      Legacy helper of :meth:`plot_OT_training`.

   .. py:method:: plot_every(directory='img/plot_every/')

      :meth:`plot_training` and :meth:`plot_pod_and_coeff` for the initial
      vector and after every stage of an ``IteratedSolver``.
