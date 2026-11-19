W_POD.two_dim.hess_diag_probed_2d
=================================

.. py:module:: W_POD.two_dim.hess_diag_probed_2d

2D counterpart of :mod:`W_POD.hess_diag_probed`. The parameters split into
three families, each probed separately (see :doc:`../../numerics`).

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Family
     - Probes
     - HVPs
   * - mode pixels, L2 part
     - zero-sum weights per (block, colour)
     - 16 per transported mode, 1 per fixed mode
   * - mode pixels, W part
     - one representative pixel per block, random ±1
     - ``w_block_k²`` per mode (9)
   * - coefficients
     - kernel-overlap colouring, every bias isolated
     - number of colours

Assumptions: the W term is downscaled (``DOWNSCALED_SIZE`` set); every node
value is positive, so ``max(vals, 0)`` is inactive; the grid is row-major
:math:`n_y \times n_x`.

Entry point
-----------

.. py:function:: calc_hessian_diag_2d(f, *args, ansatz, downscale_factor, argnums=0, w_block_k=3, seed=0, floor=True, verbose=False)

   Probed diagonal of ``f(*args)`` with respect to the flat vector
   ``args[argnums]``. The other arguments are passed through ``jit`` as
   inputs, not baked in as constants. HVPs are forward-over-reverse.

   :param f: scalar function, e.g. ``Losses._cost_sum``.
   :param ansatz: the :class:`W_POD.two_dim.ansatz.Ansatz`.
   :param int downscale_factor: pixels per block side,
      ``H // DOWNSCALED_SIZE``.
   :param int argnums: index of the parameter vector (an int only).
   :param int w_block_k: block-colouring modulus of the W part (0 skips
      W; a value ≥ blocks per side is exact).
   :param int seed: seed of the ±1 signs.
   :param bool floor: return :func:`floor_diag` of the result. The final
      code passes ``False`` and floors in
      :meth:`~W_POD.two_dim.losses.Losses_scaled.reset_scaling` instead.
   :returns: diagonal, ``numpy`` array.
   :raises TypeError: ``argnums`` is not an int, or no arguments are given.

   Example (as called in :mod:`W_POD.two_dim.losses`)::

      calc_hessian_diag_2d(losses._cost_sum, params, norm_l2, norm_w1,
                           losses.data_exact_flat, ansatz=ansatz,
                           argnums=0, downscale_factor=20, floor=False)

Building blocks
---------------

.. py:function:: hessian_diag_2d(hvp, n, ansatz, downscale_factor, *, w_block_k=3, seed=0, coeff_tol=1e-3, verbose=False)

   Unfloored diagonal from an ``hvp`` callable. Prints the coefficient
   colouring threshold and the number of colours
   (``CLIM colouring: threshold ..., ... coefficient colours``).
   :func:`calc_hessian_diag_2d` does not pass ``coeff_tol``, so the final
   runs use ``1e-3``.

   :returns: ``(diag, info)``. ``info`` contains ``diag_l2``, ``diag_w``
      and the HVP counts.

.. py:function:: ansatz_layout_2d(ansatz, n)

   ``(pods, coeffs, rbfs)``: per mode a dict with the pixel index array,
   whether it is transported, and its grid shape. Also the coefficient
   index blocks and their interpolators.

   :raises ValueError: the blocks do not partition ``0..n-1``.

.. py:function:: l2_probes(n_y, n_x, factor, transported)

   Zero-sum probes for one mode: colour :math:`(i \bmod 4, j \bmod 4)` for
   a transported mode, 1 colour for a fixed one. Within every (block,
   colour) group the weights are ±1, closed by (1, −½, −½) for odd groups,
   and sum to exactly zero.

.. py:function:: w_probes(n_y, n_x, factor, k_b, l2_diag_local, rng)

   Representative-pixel probes (the pixel with the smallest L2 diagonal in
   each block). Blocks are coloured :math:`(b_i \bmod k_b, b_j \bmod k_b)`.

.. py:function:: coeff_probes(rbfs, coeffs, tol=1e-2)

   :func:`W_POD.hess_diag_probed.coeff_colouring`, with every bias column
   moved to its own colour.

.. py:function:: zero_sum_weights(g)

   ``g`` power-of-two weights summing to exactly zero.

   :raises ValueError: ``g < 2``.

.. py:function:: _pixel_grid(n_y, n_x, factor)

   Row, column and block index of every pixel.

   :raises ValueError: the grid is not divisible by ``factor``.

.. py:function:: _probe(hvp, n, gidx, w)

   :math:`(Hv)` restricted to ``gidx``, for :math:`v` equal to ``w`` on
   ``gidx`` and 0 elsewhere.

.. py:function:: floor_diag(diag, rel=1e-8)

   :math:`\max(|d|, \text{rel}\cdot\max|d|)`.
