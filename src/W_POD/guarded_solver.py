# The below code is Claude written and human directed and verified

# WHY DOES THIS CODE EXIST? OTT ImplicitDiff, which is needed to calculate the Hessian, ran into NaN sometimes. 
# These NaNs were preventable, and this is the fix.

# The problem was that ImplicitDiff requires the solution of systems with very small matrix $A$ and rhs $b$ 
# (presumably linked to the presence of near-zero masses in the discrete measures).

# The problem is that CG does the following iterations (from Wikipedia):
# r_0 = b - Ax_0
# P_0 = r_0
# repeat for k=0, ..., n_iter :
#   a_k = (r_k.T @ r_k) / (p_k.T @ A @ p_k)
#   ...

# If A @ p_k underflows, a_k becomes NaN and the solver crashes. Lineax knows this is a danger, and there are
# checks that ensure it does not underflow. Unfortunately, there were situations on the GPU were the number 
# was big enough to pass the check, but underflowed when A @ p_k was calculated (p_k subnormal).

# This code wraps around lineax and returns 0 when there is such an underflow.

"""Floor the RHS of lineax solves via OTT's own extension point.

Alternative to `snippet_fix.py`. Both guard at the SAME layer -- inside
`CG.compute`, below the `linear_solve_p` primitive, where JAX never traces for
AD -- so they are numerically identical. They differ in how they get there:

                      snippet_fix.py              guarded_solver.py
  mechanism           monkeypatch lx.CG.compute   subclass + solver_kwargs
  scope               every lineax CG in the      only solves from the
                      process                     ImplicitDiff you configure
  symmetric branch    covered                     NOT covered (see below)
  lineax upgrades     fragile                     supported extension point
  visibility          hidden global state         explicit in your config
  what it thresholds  |A^T b|                     |A^T b|  (same)

WHY THIS REACHES THE TANGENT SOLVES
-----------------------------------
`solve_lineax` builds `solver = solver_type(**kwargs)` (lineax_implicit.py:96)
and hands that object to `lx.linear_solve`. lineax then threads THAT SAME
OBJECT into every tangent bind its JVP rule creates (`_solve.py:245` passes
`solver` through). So a guarded solver injected here propagates to the solves
OTT never constructs itself -- which is where the failure actually lives:

    role        lineax frame     solves   failing
    primal      _solve.py:782      2000        0
    primal/jvp  _solve.py:149      1800        0
    TANGENT     _solve.py:244      1800        1     <- the only failure

Verified on a 7x5 OTT problem: the guarded `compute` was reached by all 14
tangent solves, and the resulting Hessian was bit-identical to the unguarded
one (max rel diff 0.00e+00).

LIMITATION
----------
Only the NON-symmetric branch has a hook. `solve_lineax:88-89` hardcodes
`lx.CG(**kwargs)` when `symmetric=True`, with no `sym_solver` argument. With
OTT's default `precondition_fun` (eps*log) the system is non-symmetric, so this
applies; if you ever set `precondition_fun=identity` with `symmetric=True` you
fall off this hook and need `snippet_fix.py` instead.

USAGE
-----
    from guarded_solver import GuardedNormalCG
    from ott.solvers.linear import implicit_differentiation as implicit_lib

    imp = implicit_lib.ImplicitDiff(
        solver_kwargs={"nonsym_solver": GuardedNormalCG})
    solver = sinkhorn.Sinkhorn(implicit_diff=imp, use_danskin=False)

To see what it is doing, set `guarded_solver.RECORD = True` before the run and
call `report()` afterwards.
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from lineax._solution import RESULTS

# `gamma = ||A^T b||^2` goes subnormal once `||A^T b|| < sqrt(finfo.tiny)`.
# That cliff is PRECISION DEPENDENT and the gap is enormous:
#
#     fp32:  tiny=1.18e-38   sqrt(tiny)=1.08e-19   -> floor ~1e-15
#     fp64:  tiny=2.23e-308  sqrt(tiny)=1.49e-154  -> floor ~1e-150
#
# A hardcoded 1e-150 is not even REPRESENTABLE in fp32 -- it rounds to 0.0, so
# the test `m > FLOOR` silently degenerates to `m > 0.0` and catches nothing.
# Leave FLOOR as None to derive it from the RHS dtype at trace time (dtype is
# static, so this costs nothing); set a float only to override deliberately.
FLOOR = None
_MARGIN = 1e4          # sit this far above the subnormal cliff


def floor_for(dtype) -> float:
    """The RHS magnitude below which `gamma` underflows, for this precision."""
    if FLOOR is not None:
        return float(FLOOR)
    return float(np.sqrt(np.finfo(dtype).tiny) * _MARGIN)

RECORD = False        # set True to log every solve; costs a host callback each
SOLVES = []           # (|A^T b|, floored) per compute call

# Diagnostics that survive a cluster. Everything goes to STDERR with flush=True:
# `jax.debug.print` writes to STDOUT, which on a batch scheduler lands in a
# different file from the traceback you are reading.
DEBUG = False
_N_TRACE = [0]        # times compute() was TRACED (Python-level, per trace)


def _err(msg):
    print(msg, file=sys.stderr, flush=True)


def _run_cb(v, u, nf, xn):
    """Never let the diagnostic itself be the thing that raises. A failing host
    callback prints `jax.pure_callback failed` and swallows its own output,
    which is indistinguishable from `compute` never having run."""
    try:
        _err(f"[guarded_solver] RUN |A^T b|={float(v):.4e} "
             f"floored={not bool(u)} |x_raw|={float(xn):.4e} "
             f"x_nonfinite={int(nf)}")
    except Exception as e:                      # noqa: BLE001
        try:
            _err(f"[guarded_solver] RUN <callback error {type(e).__name__}: {e}>")
        except Exception:
            pass


class GuardedCG(lx.CG):
    """CG that returns zeros when the RHS has underflowed out of the
    arithmetic, instead of letting `cg.py:176` write NaN into `alpha`.

    Safe here precisely because `linear_solve_p` carries its own JVP and
    transpose rules, so JAX never differentiates through `compute` -- it runs
    on concrete values only. The same branch one layer up (in a
    `filter_primitive_bind` wrapper) trips bilinear_transpose.

    Wrapped in `lx.Normal`, `vector` here is `A^T b`, which is exactly the
    quantity whose square becomes CG's `gamma`.
    """

    def compute(self, state, vector, options):
        if DEBUG:
            # TRACE-TIME: plain Python, runs while the jaxpr is being built.
            # If this never appears, GuardedCG is not in the solver at all --
            # check the import name and that solver_kwargs actually reached
            # ImplicitDiff. It is immune to caching of the compiled executable
            # only in a fresh process, so run one.
            _N_TRACE[0] += 1
            _err(f"[guarded_solver] TRACE #{_N_TRACE[0]}  compute() "
                 f"shape={jnp.shape(jtu.tree_leaves(vector)[0])} "
                 f"dtype={jtu.tree_leaves(vector)[0].dtype} "
                 f"floor={floor_for(jtu.tree_leaves(vector)[0].dtype):.3e}")
        solution, result, stats = super().compute(state, vector, options)
        leaves = jtu.tree_leaves(vector)
        if not leaves:
            return solution, result, stats
        m = jnp.max(jnp.stack([jnp.max(jnp.abs(l)) for l in leaves]))
        usable = jnp.isfinite(m) & (m > floor_for(leaves[0].dtype))
        if DEBUG:
            # RUN-TIME: a host callback, so it fires when the executable runs.
            # TRACE printed but this did not => you are on a cached executable
            # compiled before DEBUG was set; clear JAX_COMPILATION_CACHE_DIR
            # and call jax.clear_caches().
            # NOTE: ordered=True is NOT available here -- the callback sits
            # inside OTT's custom_vjp and JAX raises
            # `KeyError: OrderedDebugEffect`. So this output is ASYNCHRONOUS:
            # on GPU it can be lost entirely if the process dies on the
            # EquinoxRuntimeError before the host queue drains, which looks
            # identical to "compute was never called". The TRACE line above is
            # synchronous Python and cannot be lost -- trust that one.
            # `solution` here is what CG returned BEFORE we zero it, so
            # |x_raw| answers the question the floor cannot: was the discarded
            # solution actually small?  A small |b| does NOT imply a small |x|
            # -- ||x|| <= ||b||/sigma_min, and sigma_min is the gauge.
            #   floored + x_nonfinite>0        -> nothing lost, CG had no answer
            #   floored + |x_raw| small        -> zeroing is harmless
            #   floored + |x_raw| LARGE finite -> the guard is discarding a real
            #                                     contribution; FLOOR is too high
            _raw = jnp.stack(jtu.tree_leaves(solution))
            jax.debug.callback(
                _run_cb, m, usable, jnp.sum(~jnp.isfinite(_raw)),
                jnp.linalg.norm(jnp.nan_to_num(_raw, posinf=0.0, neginf=0.0)))
        if RECORD:
            jax.debug.callback(
                lambda v, u: SOLVES.append((float(v), not bool(u))), m, usable)
        solution = jtu.tree_map(
            lambda l: jnp.where(usable, l, jnp.zeros_like(l)), solution)
        # reset the code too: CG may already have reported `singular` for a
        # reason other than non-finite output (e.g. hitting max_steps), and
        # `_linear_solve_impl:112` raises on any non-successful result.
        result = RESULTS.where(usable, result, RESULTS.successful)
        return solution, result, stats


def GuardedNormalCG(**kwargs):
    """`solve_lineax` calls `solver_type(**kwargs)`, so a factory is a valid
    `nonsym_solver`. `lx.NormalCG` itself cannot be subclassed -- it is a
    deprecated shim -- hence `lx.Normal(GuardedCG(...))`."""
    return lx.Normal(GuardedCG(**kwargs))


def reset():
    SOLVES.clear()


def report():
    if not SOLVES:
        print("no solves recorded -- set guarded_solver.RECORD = True before "
              "the run (and call jax.clear_caches() if it was already traced)")
        return
    v = np.array([s[0] for s in SOLVES])
    n_floored = sum(s[1] for s in SOLVES)
    fl = floor_for(np.float64) if FLOOR is None else float(FLOOR)
    print(f"solves={len(SOLVES)}  floored={n_floored}  "
          f"(floor={fl:.2e} for fp64, {floor_for(np.float32):.2e} for fp32)")
    print(f"  |A^T b|  min={v.min():.3e}  p1={np.percentile(v, 1):.3e}  "
          f"med={np.median(v):.3e}  max={v.max():.3e}")
    if n_floored:
        f = np.array([s[0] for s in SOLVES if s[1]])
        print(f"  floored |A^T b|: {f.min():.3e} .. {f.max():.3e}")
    near = int(((v > fl) & (v < fl * 1e6)).sum())
    if near:
        print(f"  !! {near} solves are within 6 orders of FLOOR -- the "
              "threshold is closer to live data than it should be")
