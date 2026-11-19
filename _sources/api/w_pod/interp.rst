W_POD.interp
============

.. py:module:: W_POD.interp

1D interpolation of a mode at transformed query points, with a controlled
continuation outside the mesh.

.. py:function:: interp_decay(old_x, coords, values, method='cubic2', decay_rate=15.)

   Interpolate inside the mesh. Outside it, continue with the edge value
   decaying exponentially.

   .. math::

      f(q) = \begin{cases}
        \varphi(x_{\min})\, e^{-\kappa |q - x_{\min}|} & q < x_{\min} \\
        \texttt{interpax.interp1d}(q) & x_{\min} \le q \le x_{\max} \\
        \varphi(x_{\max})\, e^{-\kappa |q - x_{\max}|} & q > x_{\max}
      \end{cases}

   :param old_x: query points (pulled-back mesh).
   :param coords: mesh nodes.
   :param values: mode values at the nodes.
   :param str method: ``interpax`` method.
      :class:`~W_POD.modes.TransportedMode` passes ``'catmull-rom'``.
   :param float decay_rate: :math:`\kappa`.
   :returns: interpolated values, shape of ``old_x``.

.. py:function:: interp_decay_knots(old_x, coords, values, method='catmull-rom', padding_percentage=0.1, extrap=1e-6)

   Interpolate on the mesh extended by one knot on each side, at
   ``padding_percentage`` × (domain width) beyond the ends and with value
   ``1e-6``. Beyond those knots the result is ``extrap``.
   :class:`~W_POD.modes.StretchedMode` calls it with ``method='cubic2'``
   and ``extrap=0``.
