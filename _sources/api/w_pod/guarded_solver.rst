W_POD.guarded_solver
====================

.. py:module:: W_POD.guarded_solver

A conjugate-gradient solver for OTT's implicit differentiation that returns
zero, instead of NaN, when the right-hand side has underflowed.

:meth:`W_POD.ansatz.AnsatzConfig.calc_w1` uses it as ``nonsym_solver`` of
the implicit differentiation in fp64 (:func:`~W_POD.ansatz.LUSolver` in
fp32). The crossing-waves Exp 1 runs are fp32 and do not use it; the other
1D experiments do.

Background
----------

In a normal-equations CG solve, :math:`\gamma = \lVert A^\top b\rVert^2`
becomes subnormal once :math:`\lVert A^\top b\rVert < \sqrt{\text{tiny}}`
(≈ 1e-19 in fp32, ≈ 1e-154 in fp64), and CG then writes NaN into its step.
The guard runs inside ``CG.compute``, below lineax's ``linear_solve_p``
primitive, so JAX never differentiates through it. Lineax passes the same
solver object to the tangent solves its JVP creates, so the guard covers
those too.

Only OTT's *non-symmetric* branch has a hook (``nonsym_solver``). With
``symmetric=True`` OTT hard-codes ``lx.CG``.

API
---

.. py:function:: GuardedNormalCG(**kwargs)

   Factory returning ``lineax.Normal(GuardedCG(**kwargs))``. Pass it as
   ``nonsym_solver``:

   .. code-block:: python

      from ott.solvers.linear import implicit_differentiation as imp
      ImplicitDiff(solver_kwargs={'nonsym_solver': GuardedNormalCG, 'ridge_kernel': 1e-6})

.. py:class:: GuardedCG

   Subclass of ``lineax.CG``. After the usual solve, the solution is
   replaced by zeros, and the result code by *successful*, when
   :math:`\max|A^\top b|` is not finite or is below :func:`floor_for`.

.. py:function:: floor_for(dtype)

   Threshold below which the RHS counts as underflowed:
   :data:`FLOOR` if set, else :math:`\sqrt{\text{finfo(dtype).tiny}} \times 10^4`.

.. py:data:: FLOOR

   ``None`` (derive from dtype) or a fixed float.

.. py:data:: RECORD

   If ``True``, log ``(|A^T b|, floored)`` for every solve into
   :data:`SOLVES` via a host callback.

.. py:data:: SOLVES

   List of ``(|A^T b|, floored)`` tuples filled when :data:`RECORD` is set.

.. py:data:: DEBUG

   If ``True``, print trace-time and run-time diagnostics to stderr.

.. py:function:: report()

   Summarise :data:`SOLVES`: the number floored, the distribution of
   :math:`|A^\top b|`, and a warning if many solves come within 6 orders of
   magnitude of the floor.

.. py:function:: reset()

   Clear :data:`SOLVES`.
