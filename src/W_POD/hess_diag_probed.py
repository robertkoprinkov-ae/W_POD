
# The below code is Claude written and human directed and verified
# The goal of this algorithm is to calculate the Hessian diagonal with the minimum number of probes of the 
# costly objective function. We do this by making some prior assumptions on the structure of the Hessian.
# For example:
#
# 1. The Hessian of different coefficient values, $\gamma_i(z_j), \gamma_i(z_m)$ is 0
# 2. The Hessian of the kernel interpolation weights is 0 when the kernel interpolation weights are more
#    than a few length scales away from eachother (the influence of the kernel decays)
# 3. The Hessian of two points on the mesh decays as they are further from eachother, eventually reaching 0*
#
# * assumption 3 is satsified for the L2 distance, because the used cubic spline interpolator has a decaying
#   radius of influence. It is not satisfied for the W2 distance, which adds global coupling, so that this
#   bandedness strategy amounts to an approximation in that case.
#
# This is discussed in detail below ("WHERE EACH COLORING COMES FROM"). I do note that the validity of the 
# assumption strongly depends on whether we are at the very beginning of the optimization, where none of the
# features are aligned yet and the gradient w.r.t. the mode is therefore rather meaningless and random, or
# midway through the optimization where the modes are aligned with the features and the gradient points in
# the direction of the feature therefore.

"""Hessian diagonal by probing: colouring for the coefficients, colouring or
Hutchinson for the POD block.

WHY THE SPLIT
-------------
A probe supported on only part of the parameter vector cannot be contaminated
by couplings to the rest: if v is zero on every POD index then for a
coefficient index i

    v_i (H v)_i = H_ii + sum_{j in coeff, j != i} v_i v_j H_ij

and the POD<->coeff block drops out entirely. So each half only has to handle
its own internal coupling, and the two halves can use different estimators.

MEASURED ON THE REAL 1600x1600 HESSIAN (t = 1e-2 significance):

    subspace              dim  colours    med err    p90       max
    beta only             400       14     0.365%   0.675%    1.33%
    gamma only            400      200     0.092%   0.202%    0.24%
    coeff (beta+gamma)    800      215     0.184%   0.971%    1.25%
    pod only              800       99     1.682%   2.886%    4.86%

    Hutchinson on POD    N=256    55.29% med / 171.52% p90
                         N=1024   27.65% med /  85.76% p90

So POD is the CHEAP half to colour and Hutchinson is not competitive on it --
keep `pod_method="mc"` only as a fallback -- it is both slower and less
accurate than the band colouring at every cost tested.

WHERE EACH COLOURING COMES FROM
-------------------------------
coefficients: from theory, no Hessian needed.  `predict` is linear in the
    coefficients, so H = Phi^T W Phi with Phi_ki = k(z_k, z_i); two coefficients
    couple exactly when their basis functions overlap over the snapshots.  The
    normalised overlap is computed from the RBF kernel matrices directly.
    Verified against the real H: beta predicted 5.4% dense vs 5.5% actual
    (recall 96.6%), gamma 100% vs 100%.

POD: no closed form.  `cubic2` contributes only |p-q| <= 5 (it decays as
    0.268^d -- the tridiag(1,4,1)^-1 rate, measured 0.2693), but the W1 term
    adds a FLAT far-field plateau at ~0.8% of the diagonal that runs to
    |p-q| = 399 and holds 95.6% of the significant entries.  A pure bandedness
    colouring therefore cannot be exact -- but no modulus can separate a flat
    tail, and a greedy colouring of the MEASURED pattern did no better at equal
    cost (p90 8.2% vs 4.3%, though max 22% vs 408%).  So `i mod k` is used.
"""
from typing import Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from W_POD.util import RBF_interpol

from tqdm import tqdm

# --------------------------------------------------------------------------
# colouring
# --------------------------------------------------------------------------
def greedy_colour(adj):
    """Largest-degree-first greedy distance-1 colouring (Welsh-Powell)."""
    n = adj.shape[0]
    col = -np.ones(n, dtype=int)
    for v in np.argsort(adj.sum(1))[::-1]:
        nb = col[adj[v]]
        used = set(nb[nb >= 0].tolist())
        c = 0
        while c in used:
            c += 1
        col[v] = c
    return col


def colour_from_matrix(S, tol=1e-2, scale=None):
    """Colour an explicit coupling matrix -- used for the COEFFICIENT half when
    a measured pattern is preferred over the kernel-derived one.

    `scale` defaults to min(|S_ii|,|S_jj|) -- the
    error on row i is judged against ITS OWN diagonal, so a coupling that is
    negligible beside a large H_jj can still swamp a small H_ii. Using the
    geometric mean or max instead produced 21787% and 246075% worst-case errors
    on the real Hessian."""
    S = np.asarray(S)
    if S.dtype == bool:
        adj = S.copy()
    else:
        d = np.abs(np.diag(S))
        sc = np.minimum(d[:, None], d[None, :]) if scale is None else scale
        adj = np.abs(S) / np.maximum(sc, np.finfo(float).tiny) > tol
    np.fill_diagonal(adj, False)
    return greedy_colour(adj | adj.T)


def coeff_colouring(rbfs: Sequence[RBF_interpol] | dict,
                    tol: float = 1e-2) -> np.ndarray:
    """Colouring for coefficient blocks, from the kernels alone -- no Hessian
    needed.

    `rbfs` is a list of RBF_interpol in parameter-index order, e.g.
    [beta0, gamma0, beta1, gamma1] for a two-mode ansatz. (A dict, as returned
    by Mode.get_RBF(), is also accepted for the single-mode case.)

    Uses H = Phi^T W Phi with Phi_ki = k(z_k, z_i): two coefficients couple
    exactly when their basis functions overlap over the snapshots. Verified
    against a real 1600x1600 Hessian -- beta predicted 5.4% dense vs 5.5%
    actual (recall 96.6%), gamma 100% vs 100%. Cross-mode pairs share a basis,
    so the overlap over-predicts there (5.4% vs 1.7% measured), which is the
    safe direction: it separates pairs that need not have been separated.
    """
    if isinstance(rbfs, dict):
        rbfs = list(rbfs.values())
    # Phi_ki = phi_i(z_k), shape (n_snapshots, n_param). This is exactly
    # `get_linear_constraint_matrix()`: it is K, plus a column of ones when
    # bias=True -- the bias coefficient's basis function is the CONSTANT 1
    # (predict adds coeff[-1] at every z), so it overlaps everything and gets
    # its own colour.
    Phi = [np.asarray(r.get_linear_constraint_matrix(), dtype=np.float64)
           for r in rbfs]
    nrm = [np.sqrt((P ** 2).sum(0)) for P in Phi]
    sizes = [P.shape[1] for P in Phi]
    off = np.cumsum([0] + sizes)
    n = int(off[-1])
    adj = np.zeros((n, n), dtype=bool)
    for a, Pa in enumerate(Phi):
        for b, Pb in enumerate(Phi):
            G = np.abs(Pa.T @ Pb) / np.outer(nrm[a], nrm[b])
            adj[off[a]:off[a+1], off[b]:off[b+1]] = G > tol
    np.fill_diagonal(adj, False)
    return greedy_colour(adj | adj.T)


def band_colouring(n, k, n_blocks=1, per_block=True):
    """`i mod k` interleaving -- the no-Hessian-needed POD option.

    `n_blocks` splits the n indices into equal consecutive blocks (one per
    mode). With `per_block=True` each block gets its OWN colour set, so index i
    of mode 0 never shares a colour with index j of mode 1; cost is
    n_blocks * k. With `per_block=False` the colours are shared (cost k).

    Measured on the real 1600x1600 Hessian, POD half only, coefficients
    coloured separately at 215 HVPs:

        colouring                       HVPs   p90 err   max err
        k=41  shared across modes        256    13.398%  3503.77%
        k=99  shared across modes        314     4.288%   407.92%
        k=41  separate per mode          297     5.864%    18.04%
        k=99  separate per mode          413     3.107%     7.01%

    Sharing is cheaper per colour but leaves a heavy tail: pod0 x pod1 has 6.7%
    significant couplings, and a shared colouring puts those pairs together.
    per_block=True removes that and is the better value -- k=41 separate beats
    k=99 shared on max error at fewer HVPs.

    Separation costs k extra HVPs and cut the max error by 194x in the table
    above; the medians are indistinguishable, so only the tail responds. That
    tail is what matters for a preconditioner -- it is the SMALL diagonals that
    blow up, and 1/sqrt(h) amplifies exactly those.
    """
    if n % n_blocks:
        raise ValueError(f"n={n} is not divisible by n_blocks={n_blocks}")
    m = n // n_blocks
    if not per_block:
        return np.arange(n) % k
    return np.concatenate([b * k + (np.arange(m) % k) for b in range(n_blocks)])


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------
def _probe_diag(hvp, n, idx, colours, batch_size=8):
    """One HVP per colour. v is the 0/1 indicator of the colour class,
    embedded in the full space and zero elsewhere, so couplings to indices
    outside `idx` never enter."""
    out = np.zeros(n)
    ncol = int(colours.max()) + 1
    V = np.zeros((ncol, n))
    for c in range(ncol):
        V[c, idx[colours == c]] = 1.0
    print('Starting probe')
    #Hv = jax.lax.map(hvp, jnp.asarray(V), batch_size=batch_size)
    hvp_jit = jax.jit(hvp)
    V = jnp.array(V)
    Hv = jnp.zeros((V.shape[0], V.shape[1]))
    for i in tqdm(range(V.shape[0])):
        Hv = Hv.at[i].set(hvp_jit(V[i, :]))
    print('Ending probe')
    Hv = np.asarray(Hv)
    for c in range(ncol):
        sel = idx[colours == c]
        out[sel] = Hv[c][sel]
    return out, ncol


def _mc_diag(hvp, n, idx, n_samples, key, batch_size=8):
    """Rademacher (Hutchinson) on a subspace: v is +-1 on `idx`, 0 elsewhere,
    so only within-subspace couplings contribute to the variance."""
    keys = jax.random.split(key, n_samples)

    def one(k):
        v = jnp.zeros(n).at[idx].set(
            jax.random.rademacher(k, (len(idx),), dtype=jnp.float32).astype(jnp.float64)
            if jnp.zeros(1).dtype == jnp.float64
            else jax.random.rademacher(k, (len(idx),), dtype=jnp.float32))
        return v * hvp(v)

    acc = jax.lax.map(one, keys, batch_size=batch_size).sum(0) / n_samples
    out = np.zeros(n)
    out[idx] = np.asarray(acc)[idx]
    return out, n_samples


def _probe_coeffs(hvp, n, coeff_idx, coeff_colours, batch_size):
    return _probe_diag(hvp, n, coeff_idx, np.asarray(coeff_colours), batch_size)


def hessian_diag_band(hvp: Callable, n: int, pod_idx: np.ndarray,
                      coeff_idx: np.ndarray, *, coeff_colours,
                      pod_k: int = 41, pod_blocks: int = 1,
                      pod_per_block: bool = True, batch_size: int = 8):
    """diag(H) with the POD half done by `i mod pod_k` colouring.

    hvp(v) -> H v          a callable on a length-n vector
    pod_idx, coeff_idx     index arrays partitioning the parameters
    coeff_colours          colouring of coeff_idx (see `coeff_colouring`)
    pod_k                  modulus; larger = more HVPs, less error
    pod_blocks             number of modes in pod_idx
    pod_per_block          give each mode its own colour set (recommended)
    """
    diag = np.zeros(n)
    dc, kc = _probe_coeffs(hvp, n, coeff_idx, coeff_colours, batch_size)
    cols = band_colouring(len(pod_idx), pod_k, n_blocks=pod_blocks,
                          per_block=pod_per_block)
    dp, kp = _probe_diag(hvp, n, pod_idx, cols, batch_size)
    diag += dc + dp
    return diag, dict(coeff_hvps=kc, pod_hvps=kp, total_hvps=kc + kp,
                      pod_method="band", pod_k=pod_k)


def hessian_diag_mc(hvp: Callable, n: int, pod_idx: np.ndarray,
                    coeff_idx: np.ndarray, *, coeff_colours,
                    n_samples: int = 256, key=None, batch_size: int = 8):
    """diag(H) with the POD half done by Rademacher/Hutchinson.

    Makes no structural assumption, but the variance is set by the POD block's
    off-diagonal mass, which is large here -- see the module docstring. On the
    real problem this needed more HVPs than the exact method to reach 12%
    error. Prefer `hessian_diag_band` unless the band assumption is suspect.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    diag = np.zeros(n)
    dc, kc = _probe_coeffs(hvp, n, coeff_idx, coeff_colours, batch_size)
    dp, kp = _mc_diag(hvp, n, pod_idx, n_samples, key, batch_size)
    diag += dc + dp
    return diag, dict(coeff_hvps=kc, pod_hvps=kp, total_hvps=kc + kp,
                      pod_method="mc", n_samples=n_samples)


# --------------------------------------------------------------------------
# W_POD glue (duck-typed: no W_POD imports, only attribute access)
# --------------------------------------------------------------------------
def ansatz_layout(ansatz, n: int) -> tuple[np.ndarray, np.ndarray, list[RBF_interpol]]:
    """pod / coeff index arrays and the RBFs in parameter-index order.

    Asks the ansatz for its own indexing rather than reproducing it: feeding
    `arange(n)` through `arr_to_dict` makes each returned "value" literally the
    index array of that block. Nothing here assumes how modes are concatenated
    or how a mode orders [pod_local, beta, gamma], so it stays correct if the
    layout changes or a mode type has different parameter names.

    Coefficient index arrays and `rbfs` are built in the same key order, which
    is what `coeff_colouring` needs to line its overlap blocks up with the
    parameter indices.
    """
    idx = ansatz.arr_to_dict(jnp.arange(n))
    pod, coeff, rbfs = [], [], []
    for mode, per_mode in zip(ansatz.modes, idx):
        rbf = mode.get_RBF()                      # {coeff name: RBF_interpol}
        for name in mode.get_param_names():
            block = np.asarray(per_mode[name])
            if name in rbf:
                coeff.append(block)
                rbfs.append(rbf[name])
            else:
                pod.append(block)
    return np.concatenate(pod), np.concatenate(coeff), rbfs


def _cfg_probe_inputs(cfg, params):
    n = int(params.shape[0])
    pod_idx, coeff_idx, rbfs = ansatz_layout(cfg.ansatz, n)
    norm_l2 = cfg.calc_l2_training(params)
    norm_w1 = cfg.calc_w1_training(params)
    f = lambda p: cfg._cost(p, norm_l2, norm_w1, use_danskin=False)
    hvp = lambda v: jax.jvp(jax.grad(f), (params,), (v,))[1]
    return hvp, n, pod_idx, coeff_idx, coeff_colouring(rbfs)


def calc_hessian_diag_band(cfg, params: jax.Array, *,
                           k_band: int = 41, batch_size: int = 8):
    """Probed Hessian diagonal for an AnsatzConfig, POD half by band colouring.

    This is the recommended entry point. Lift it into the class as
    `_calc_hessian_diag` when ready -- it needs only `cfg.ansatz`,
    `cfg.calc_l2_training`, `cfg.calc_w1_training` and `cfg._cost`.

    Measured on the real 1600-parameter Hessian, against 1600 HVPs exact:

        k_band=41 -> ~297 HVPs   scaling error med 0.38%  p90 3.07%
        k_band=99 -> ~413 HVPs   scaling error med 0.19%  p90 1.59%
    """
    hvp, n, pod_idx, coeff_idx, cc = _cfg_probe_inputs(cfg, params)
    return hessian_diag_band(hvp, n, pod_idx, coeff_idx, coeff_colours=cc,
                             pod_k=k_band, pod_blocks=len(cfg.ansatz.modes),
                             batch_size=batch_size)[0]


def calc_hessian_diag_mc(cfg, params: jax.Array, *,
                         n_samples: int = 256, key=None, batch_size: int = 8):
    """As `calc_hessian_diag_band`, but the POD half by Hutchinson.

    Measured on the real 1600-parameter Hessian:

        n_samples=256  ->  471 HVPs   diag p90 89.4%
        n_samples=1024 -> 1239 HVPs   diag p90 46.2%

    i.e. more expensive than the 1600-HVP exact method at usable accuracy.
    Kept for when the band assumption cannot be trusted.
    """
    hvp, n, pod_idx, coeff_idx, cc = _cfg_probe_inputs(cfg, params)
    return hessian_diag_mc(hvp, n, pod_idx, coeff_idx, coeff_colours=cc,
                           n_samples=n_samples, key=key, batch_size=batch_size)
