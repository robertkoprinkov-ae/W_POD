W_POD.two_dim.visu
==================

.. py:module:: W_POD.two_dim.visu

.. py:class:: Visualizer(pts, x_crds, y_crds, data_z, data_exact, ansatz)

   ``pcolormesh`` plots of 2D reconstructions.

   .. py:method:: plot_data(data, ax=None)

      Plot one flat field (length ``n_y * n_x``) on the grid.

   .. py:method:: plot_recon(params, filename_prefix, losses)

      One panel per snapshot, titled with the relative L2 and W values,
      with the moving centre as a red cross. Then the coefficient plots of
      frames 0 and 1. Assumes exactly two frames and few snapshots.
      Writes ``<prefix>_recon.png``.

   .. py:method:: plot_recon_2d(params, filename_prefix, losses, n_temp, n_time)

      Reconstructions on an ``n_temp × n_time`` grid of panels (for
      :math:`z` = (case, time)). Writes ``<prefix>_recon.png``.

      :raises AssertionError: ``n_z != n_temp * n_time``.
