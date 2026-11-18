"""Hessian diagonal by probing for the 2D wildfire ansatz.

2D counterpart of `W_POD/hess_diag_probed.py`. The parameters split into
three kinds of index, each probed with its own family of probe vectors. A
probe is zero outside its own family, so couplings to the other families
never enter the readout (same argument as in 1D).

POD PIXELS, L2 PART -- exact, "zero-sum" probes
-----------------------------------------------
`catmull-rom` in `interpax.interp2d` is exactly local: a query depends on a
4x4 block of nodes (3x3 at a corner, 1 node on a node). So through the L2
term two pixels of a TRANSPORTED mode couple iff |di| <= 3 and |dj| <= 3,
and colouring pixel (i, j) by (i mod 4, j mod 4) separates them exactly:
16 colours per mode. A FIXED mode is not interpolated at all, so its L2
Hessian is diagonal in pixels: 1 colour.

The W term breaks this. With `DOWNSCALED_SIZE` it sees the pixels only
through the block sums of the downscaling, so two pixels of the same block
couple exactly as strongly as a pixel couples to itself, and the blocks
couple through a dense matrix. But W is then also exactly INVARIANT along
any direction that sums to zero inside every block. So within every
(block, colour) group the probe weights are chosen to sum to zero (+-1
alternating, and 1, -1/2, -1/2 to close an odd group; every weight is a
power of two, so the sum is exactly zero in floating point too). H_W v = 0,
and v_p^-1 (H v)_p is exactly the L2 diagonal -- W's contamination AND W's
own diagonal both drop out.

Measured (32x32 grid, 8x8-pixel blocks, total cost, use_danskin default):
max |H_W v| = 0.0 over all zero-sum probes, and the L2 diagonal read off the
probes equals the exact one to 0.0.

POD PIXELS, W PART -- one representative pixel per block
---------------------------------------------------------
The W diagonal is the same for every pixel of a block (it is a function of
the block sums only). Measured on a 100x100 grid with 25x25 blocks (the
production block layout): (max-min)/|mean| within a block = 1.4e-16 ..
4.2e-16. So there is one unknown per (mode, block). Each probe puts one
representative pixel per block, with random +-1 signs, in the blocks of one
block-colour (bi mod k_b, bj mod k_b); the representative's readout minus
its (exact) L2 diagonal is W's diagonal for the whole block. The
representative is the block's pixel with the SMALLEST L2 diagonal, so the
subtraction cancels as little as possible.

Contamination comes from the other same-coloured blocks through the dense
block coupling. The random signs matter: with all +1 the normalisation
`b / b.sum()` gives every block coupling the same sign and the sum is a bias
as large as the diagonal. Measured, relative error of W's diagonal alone:

    probes / mode         transported              fixed
                        med     p90    max      med     p90    max
    k_b=1, all +1     50.8%  144.8% 371.0%     9.1%  118.7% 302.9%
    k_b=1, random     2.66%   9.64%  44.6%    0.16%   1.29%  15.1%
    k_b=3 (9)         0.85%   2.41%  30.6%    0.11%   0.71%  10.8%
    k_b=5 (25)        0.39%   1.81%  12.4%    0.07%   0.45%   4.7%
    k_b=8 (64)        0.25%   1.03%   5.8%    0.04%   0.24%   2.7%

(k_b >= blocks per side is exact.) The large maxima sit on blocks whose W
diagonal is ~10x smaller than the rest. Relative to the TOTAL diagonal the
error is smaller again, since the L2 part is exact. On that toy the W
diagonal was NEGATIVE in every block, for both modes -- with the default
`use_danskin`, so possibly a Danskin artefact (see future.md 11e/11f).

COEFFICIENTS
------------
`coeff_colouring` from the 1D module, unchanged: with theta = 1e-6 every RBF
is a lookup table (K = I), each coefficient touches one snapshot, and
coefficients of different snapshots never couple. One change: every BIAS
column gets its own colour. The pairwise overlap test sees a bias column
overlap each lookup coefficient by only 1/sqrt(n_z), which drops below the
1e-2 threshold once n_z > 1e4 -- the bias would then share a colour with all
n_z coefficients and its diagonal would be off by ~100%.

ASSUMPTIONS
-----------
- The downscaled W (`Losses(..., DOWNSCALED_SIZE=...)`); without
  downscaling W couples all pixels densely and nothing here applies.
- The `max(vals, 0)` in `loss_w2_z` is inactive (all node values > 0, which
  the bounds guarantee); otherwise W is not a function of block sums only.
- The frame's grid is row-major (n_y, n_x) in node order, which is how
  `pts` is built from `meshgrid(x_crds, y_crds)` and how `loss_w2_z`
  reshapes the values.
"""
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from W_POD.hess_diag_probed import coeff_colouring

from src_wildfire.ansatz import IdentityCoordinateTransformation

from tqdm import tqdm


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
def ansatz_layout_2d(ansatz, n: int):
    """POD blocks, coefficient blocks and their RBFs, in parameter order.

    Feeds `arange(n)` through `split_params_recursive`, so every leaf is the
    index array of that parameter -- the ansatz's own indexing, not a copy.

    Returns
      pods    list of dicts: idx (n_y*n_x,), transported (bool), n_y, n_x, name
      coeffs  list of index arrays, one per RBF-coefficient block
      rbfs    the RBF_interpol for each entry of `coeffs`, same order
    """
    idx = ansatz.split_params_recursive(jnp.arange(n))
    pods, coeffs, rbfs = [], [], []
    for cf_i, cf in enumerate(ansatz.coordinate_frames):
        cf_idx = idx['coord_frame_%d' % cf_i]
        ct = cf.coordinate_transformation
        # same test as CoordinateFrame.get_value_nodes / get_value_grid_arbitrary
        transported = not (cf.fixed or type(ct) == IdentityCoordinateTransformation)
        for name, block in cf_idx['coordinate_transformation'].items():
            if name.startswith('radial_'):
                rbf = ct.rbf_radial_shifts
            elif name == 'shift_x_coeff':
                rbf = ct.rbf_shift_x
            elif name == 'shift_y_coeff':
                rbf = ct.rbf_shift_y
            else:
                raise KeyError('unknown coordinate-transformation parameter %s' % name)
            coeffs.append(np.asarray(block))
            rbfs.append(rbf)
        for mode_i, mode in enumerate(cf.modes):
            m_idx = cf_idx['mode_%d' % mode_i]
            pods.append(dict(idx=np.asarray(m_idx['mode']), transported=transported,
                             n_y=cf.n_y, n_x=cf.n_x,
                             name='coord_frame_%d/mode_%d' % (cf_i, mode_i)))
            if 'amplitude_coeff' in m_idx:
                coeffs.append(np.asarray(m_idx['amplitude_coeff']))
                rbfs.append(mode.rbf_amplitudes)

    seen = np.concatenate([p['idx'] for p in pods] + coeffs)
    if seen.shape[0] != n or not np.array_equal(np.sort(seen), np.arange(n)):
        raise ValueError('layout does not partition the %d parameters' % n)
    return pods, coeffs, rbfs


# --------------------------------------------------------------------------
# probe construction
# --------------------------------------------------------------------------
def zero_sum_weights(g: int) -> np.ndarray:
    """g nonzero weights, all powers of two, summing to exactly zero."""
    if g < 2:
        raise ValueError('a zero-sum group needs at least 2 pixels, got %d' % g)
    if g % 2 == 0:
        return np.tile([1., -1.], g // 2)
    return np.concatenate([np.tile([1., -1.], (g - 3) // 2), [1., -.5, -.5]])


def _pixel_grid(n_y, n_x, factor):
    if n_y % factor or n_x % factor:
        raise ValueError('grid %dx%d is not divisible by the downscale factor %d'
                         % (n_y, n_x, factor))
    row, col = np.divmod(np.arange(n_y * n_x), n_x)
    n_bx = n_x // factor
    blk = (row // factor) * n_bx + (col // factor)
    return row, col, blk


def l2_probes(n_y, n_x, factor, transported):
    """Zero-sum probes for one mode, as (local pixel indices, weights) pairs.

    Colour (i mod 4, j mod 4) for a transported mode, a single colour for a
    fixed one. Within every (block, colour) group the weights sum to zero.
    """
    row, col, blk = _pixel_grid(n_y, n_x, factor)
    k = 4 if transported else 1
    colour = (row % k) * k + (col % k)
    probes = []
    for c in range(k * k):
        sel = np.where(colour == c)[0]
        pix, w = [], []
        for b in np.unique(blk[sel]):
            g = sel[blk[sel] == b]
            pix.append(g)
            w.append(zero_sum_weights(g.shape[0]))
        probes.append((np.concatenate(pix), np.concatenate(w)))
    return probes


def w_probes(n_y, n_x, factor, k_b, l2_diag_local, rng):
    """Representative-pixel probes for one mode's W diagonal.

    Returns (probes, rep, blk): probes as (local pixel indices, +-1 signs);
    rep[b] is the representative pixel of block b; blk maps pixel -> block.
    """
    _, _, blk = _pixel_grid(n_y, n_x, factor)
    n_bx = n_x // factor
    n_b = blk.max() + 1
    order = np.argsort(blk, kind='stable')
    per_block = order.reshape(n_b, factor * factor)          # pixels of each block
    rep = per_block[np.arange(n_b), np.argmin(l2_diag_local[per_block], axis=1)]
    bi, bj = np.divmod(np.arange(n_b), n_bx)
    bcol = (bi % k_b) * k_b + (bj % k_b)
    sign = rng.choice([-1., 1.], n_b)
    probes = [(rep[bcol == c], sign[bcol == c]) for c in range(k_b * k_b)
              if (bcol == c).any()]            # k_b > blocks per side leaves colours empty
    return probes, rep, blk


def coeff_probes(rbfs, coeffs, tol=1e-2):
    """Colour classes of the coefficients, bias columns isolated."""
    colours = coeff_colouring(rbfs, tol=tol)
    off = np.cumsum([0] + [c.shape[0] for c in coeffs])
    bias_pos = [off[i + 1] - 1 for i, r in enumerate(rbfs) if r.bias]
    colours = colours.copy()
    colours[bias_pos] = colours.max() + 1 + np.arange(len(bias_pos))
    colours = np.unique(colours, return_inverse=True)[1]
    flat = np.concatenate(coeffs)
    return [(flat[colours == c], np.ones(int((colours == c).sum())))
            for c in range(colours.max() + 1)]


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------
def _probe(hvp, n, gidx, w):
    """(H v)[gidx] for v = w on gidx, zero elsewhere."""
    v = np.zeros(n)
    v[gidx] = w
    return np.asarray(hvp(jnp.asarray(v)))[gidx]


def hessian_diag_2d(hvp: Callable, n: int, ansatz, downscale_factor: int, *,
                    w_block_k: int = 3, seed: int = 0, coeff_tol: float = 1e-3,
                    verbose: bool = False):
    """Unfloored diag(H) from an hvp(v) -> H v callable.

    w_block_k   block colouring modulus for the W part: k_b**2 probes per
                mode; 0 skips W; >= blocks per side is exact.
    Returns (diag, info). info['diag_l2'] and info['diag_w'] hold the two POD
    parts separately (zero on coefficients).
    """
    pods, coeffs, rbfs = ansatz_layout_2d(ansatz, n)
    rng = np.random.default_rng(seed)
    diag_l2 = np.zeros(n)
    diag_w = np.zeros(n)
    diag_c = np.zeros(n)
    n_l2 = n_w = 0

    for pod in pods:
        gi, n_y, n_x = pod['idx'], pod['n_y'], pod['n_x']
        probes = l2_probes(n_y, n_x, downscale_factor, pod['transported'])
        for loc, w in (tqdm(probes, desc='L2 ' + pod['name']) if verbose else probes):
            diag_l2[gi[loc]] = _probe(hvp, n, gi[loc], w) / w
        n_l2 += len(probes)

        if w_block_k > 0:
            probes, rep, blk = w_probes(n_y, n_x, downscale_factor, w_block_k,
                                        diag_l2[gi], rng)
            w_block = np.zeros(rep.shape[0])
            for loc, s in (tqdm(probes, desc='W  ' + pod['name']) if verbose else probes):
                hv = _probe(hvp, n, gi[loc], s)
                w_block[blk[loc]] = s * hv - diag_l2[gi[loc]]
            diag_w[gi] = w_block[blk]
            n_w += len(probes)

    probes = coeff_probes(rbfs, coeffs, tol=coeff_tol)
    print('CLIM colouring: threshold %g, %d coefficient colours' % (coeff_tol, len(probes)), flush=True)
    for gidx, w in (tqdm(probes, desc='coefficients') if verbose else probes):
        diag_c[gidx] = _probe(hvp, n, gidx, w)

    diag = diag_l2 + diag_w + diag_c
    return diag, dict(diag_l2=diag_l2, diag_w=diag_w, l2_hvps=n_l2, w_hvps=n_w,
                      coeff_hvps=len(probes), total_hvps=n_l2 + n_w + len(probes),
                      w_block_k=w_block_k)


def floor_diag(diag, rel=1e-8):
    """|diag| floored at rel * max|diag| -- the same rule as
    ParameterRescaling.reset_scaling in W_POD/ansatz.py, so zero-curvature
    parameters (e.g. the shift coefficients under fix_center=True) get a
    finite scaling."""
    a = np.abs(diag)
    return np.maximum(a, rel * a.max())


def calc_hessian_diag_2d(f: Callable, *args, ansatz, downscale_factor: int,
                         argnums: int = 0, w_block_k: int = 3, seed: int = 0,
                         floor: bool = True, verbose: bool = False):
    """Probed Hessian diagonal of `f(*args)` with respect to `args[argnums]`.

    Same calling convention as the Hutchinson `Losses_scaled._hess_diag`
    (losses.py), so the two are interchangeable, e.g.

        calc_hessian_diag_2d(losses._cost_sum, params, norm_l2, norm_w1,
                             losses.data_exact_flat,
                             ansatz=ansatz, downscale_factor=20)

    f                  scalar function; differentiated w.r.t. args[argnums]
    args               all positional arguments of f; args[argnums] is the
                       flat parameter vector the ansatz lays out. The others
                       are held fixed and passed through jit as INPUTS (not
                       baked-in constants), so large arrays such as
                       data_exact_flat do not end up inside the compiled
                       program. They must be arrays / pytrees of arrays; bind
                       anything static (strings, python flags) with
                       functools.partial first.
    ansatz, downscale_factor  keyword-only; downscale_factor = pixels per
                       block side, i.e. H // DOWNSCALED_SIZE
    argnums            an int -- the probing layout is that of ONE flat
                       parameter vector (unlike _hess_diag, no tuples)
    w_block_k          see `hessian_diag_2d`; 3 -> 9 probes per mode
    floor              apply `floor_diag` (returns |diag|, floored); False
                       returns the signed diagonal

    HVPs are forward-over-reverse (jvp of grad), the combination that works
    through OTT's implicit differentiation (future.md 11f).
    """
    if not isinstance(argnums, int):
        raise TypeError('argnums must be an int (one flat parameter vector), got %r' % (argnums,))
    if not args:
        raise TypeError('calc_hessian_diag_2d needs at least the parameter vector in *args')
    args = list(args)
    params = jnp.asarray(args[argnums])
    n = int(params.shape[0])

    def f_of_params(p, rest):
        full = list(rest)
        full[argnums] = p
        return f(*full)

    hvp_p = jax.jit(lambda p, rest, v: jax.jvp(lambda q: jax.grad(f_of_params)(q, rest),
                                               (p,), (v,))[1])
    hvp = lambda v: hvp_p(params, args, v)
    diag, info = hessian_diag_2d(hvp, n, ansatz, downscale_factor,
                                 w_block_k=w_block_k, seed=seed, verbose=verbose)
    if verbose:
        print('Hessian diagonal: %d HVPs (L2 %d, W %d, coefficients %d)'
              % (info['total_hvps'], info['l2_hvps'], info['w_hvps'], info['coeff_hvps']))
    return floor_diag(diag) if floor else diag
