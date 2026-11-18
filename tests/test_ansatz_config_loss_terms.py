"""Math-tier tests for AnsatzConfig's loss terms (group P in unit_tests.md).

Implemented one piece at a time, per explicit instruction; this file
currently covers `calc_l2` / `calc_l2_training`, `calc_reg`, `cut_grad`,
`calc_w1` / `calc_w1_training`, `cost_jit`'s interaction with
`update_scaling`, and `calc_pod_norms_reg` -- all of group P's items are now
implemented, see unit_tests.md group P. Also includes one DELIBERATELY,
PERMANENTLY FAILING test, `test_P_calc_w1_training_hessian_matches_fd` (see
its own comment block and `future.md` section 11): `jax.hessian
(calc_w1_training)` does not compute the true Hessian, most likely because
OTT's Sinkhorn solve uses implicit/Danskin differentiation that is only
valid to first order.

Uses a small, single-mode `DataCreation` fixture (same style as group H's
``tests/test_datacreation_plumbing.py``, kept self-contained here rather than
imported, matching this test suite's existing convention of no cross-file
test imports).

`frobenius_or_l2_squared=1.` is passed explicitly, matching real usage (see
`future.md`'s "Legacy **power in the L2 term." entry: this parameter is
always passed `1.` in practice; `LossWeights`'s own default, `2.`, is the
legacy/unused value). With `frobenius_or_l2_squared=1.`,
`calc_l2_training(params) == sum_z calc_l2(z, params, data_exact_z)` exactly
-- no extra squaring.

`calc_l2`/`calc_l2_training` don't depend on `weights_loss.{l2,w2,
penalty_exceed,reg_norm_pod,normalize}`, `w2_config`, `ansatz_grad`, or
`ansatz_coeff_reg` at all -- but `AnsatzConfig.__init__` requires valid
instances of all of them regardless (it asserts `ansatz_grad.ansatz ==
ansatz` and `ansatz_coeff_reg.ansatz == ansatz`), so minimal-but-valid ones
are still built.

Both formulas are pure arithmetic over `ansatz.get_values_grid` (already
validated independently by group N) and `DataCreation.data_exact`/`dx` --
no RBF-fitting noise, no interpolation-scheme choice being tested here, just
whether `AnsatzConfig` wires dx/reconstruction/sum-of-squares together
correctly.

Tolerance history worth recording: an initial 5-seed sample gave fp32
absolute diffs of `5.96e-8` (calc_l2) / `2.384e-7` (calc_l2_training), which
looked like it would support a small fixed atol -- but sweeping 200 random
parameter draws found a worse case (`2.75e-6`) that broke an atol picked
from just 5 seeds. The reason: the residual `rec - sol_exact` (and hence
calc_l2's own magnitude) scales with `params_arr`'s random draw, and fp32
absolute roundoff scales with it too -- this is ordinary floating-point
error proportional to the VALUE's magnitude, not a fixed noise floor, so a
fixed atol is the wrong tool no matter how carefully it's calibrated against
a small sample. Checking the RELATIVE error instead confirms this: worst
relative diff over 200 seeds was `1.61e-7` (calc_l2) / `5.24e-7`
(calc_l2_training) -- both consistently near fp32 machine epsilon
(`~1.19e-7`), regardless of the draw's magnitude. `rtol=2e-6` (~3.8x-12x
headroom over the 200-seed worst cases) with a tiny `atol=1e-9` floor (for
the near-zero-expected-value edge case) is used for both, verified stable
across 500 seeds. fp64 is exactly `0.0` at every seed tried -- pure
machine-epsilon roundoff, trivially inside this tolerance too. Detection
power: comparing against a deliberately mismatched (shifted-by-one-snapshot)
`data_exact` gives `diff=0.0553`, `rel=4.4%` -- far above either bound.
"""

import numpy as np
import jax
import jax.numpy as jnp

from W_POD.modes import GlobalMode, TransportedMode, StretchedMode
from W_POD.util import RBF_interpol, DataCreation
from W_POD.ansatz import (
    Ansatz, AnsatzConfig, LossWeights, W2Config,
    OptimPhaseGradConfig, OptimPhaseCoeffConfig,
)

_ATOL = 1e-9
_RTOL = 2e-6

_X = jnp.linspace(0.0, 1.0, 21)
_DATA_Z = jnp.array([0.2, 0.6, 1.0])


def _bump(center, width):
    return jnp.exp(-0.5 * ((_X - center) / width) ** 2)


def _build_ansatz_config():
    """A small, single-TransportedMode AnsatzConfig with a known closed-form
    ground truth (beta(z), gamma(z) as explicit functions of z) -- enough to
    build a real DataCreation, but no particular relationship to the
    arbitrary params_arr the loss-term tests evaluate at."""
    rbf = {
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "gamma_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
    }
    ansatz = Ansatz([TransportedMode(_X, rbf["beta_coeff"], rbf["gamma_coeff"])])
    values = [{
        "pod_local": lambda z: _bump(0.5, 0.1),
        "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
        "gamma_value": lambda z: 0.1 * jnp.sin(z),
    }]
    data = DataCreation(_X, ansatz, values, _DATA_Z, rbf)

    grad_cfg = [{name: {"l2": True, "w1": True} for name in ansatz.modes[0].get_param_names()}]
    coeff_cfg = [{name: 0.1 for name in ansatz.modes[0].get_coeff_names()}]
    ansatz_grad = OptimPhaseGradConfig(ansatz, grad_cfg)
    ansatz_coeff_reg = OptimPhaseCoeffConfig(ansatz, coeff_cfg)

    # l2/w2/penalty_exceed/reg_norm_pod/normalize/w2_config are all unused by
    # calc_l2/calc_l2_training -- values here are arbitrary-but-valid.
    weights = LossWeights(l2=1., w2=0., penalty_exceed=0., reg_norm_pod=0.,
                          normalize=False, frobenius_or_l2_squared=1.)
    w2_config = W2Config(tau_a=1., tau_b=1., rescale=True)

    return AnsatzConfig(data, weights, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg), ansatz, data


def _independent_l2(ansatz, dx, z, params_arr, sol_exact):
    rec = np.asarray(ansatz.get_values_grid(z, params_arr))
    return float(dx) * float(np.sum((rec - np.asarray(sol_exact)) ** 2))


def test_P_calc_l2_matches_independent_recompute():
    """calc_l2(z, params, sol_exact) == dx * sum((reconstruction - sol_exact)^2),
    recomputed independently in plain numpy (not by calling calc_l2's own
    formula back at itself)."""
    cfg, ansatz, data = _build_ansatz_config()
    rng = np.random.default_rng(0)
    params_arr = jnp.asarray(rng.standard_normal(ansatz.n_param()))

    z_idx = 1  # z=0.6, arbitrary -- off the mesh/data boundary, no special role
    z = data.data_z[z_idx]
    sol_exact = data.data_exact[z_idx]

    result = float(cfg.calc_l2(z, params_arr, sol_exact))
    expected = _independent_l2(ansatz, data.dx, z, params_arr, sol_exact)

    np.testing.assert_allclose(result, expected, atol=_ATOL, rtol=_RTOL)


def test_P_calc_l2_training_matches_independent_recompute():
    """calc_l2_training(params) == sum over all training z of calc_l2(z,
    params, data_exact_z), with frobenius_or_l2_squared=1. (real usage, see
    module docstring) -- recomputed independently, summing plain-numpy
    calc_l2 values across every snapshot."""
    cfg, ansatz, data = _build_ansatz_config()
    rng = np.random.default_rng(0)
    params_arr = jnp.asarray(rng.standard_normal(ansatz.n_param()))

    result = float(cfg.calc_l2_training(params_arr))
    expected = sum(
        _independent_l2(ansatz, data.dx, data.data_z[zi], params_arr, data.data_exact[zi])
        for zi in range(data.data_z.shape[0])
    )

    np.testing.assert_allclose(result, expected, atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# calc_reg -- aggregate RKHS regularization across modes/coefficients
# ---------------------------------------------------------------------------
#
# calc_reg(params) = sum over (mode_i, coeff_name) of
#     coeff_cfg[mode_i][coeff_name] * RBF.calc_RKHS_reg(coeff)
#
# calc_RKHS_reg itself is already independently verified by group G's item
# 39 (coeff_nobias.T @ _K @ coeff_nobias, by direct recompute). This test is
# NOT about re-deriving that formula -- it's about whether AnsatzConfig
# wires the per-mode/per-coefficient LOOP together correctly: right
# mode_i/coeff_name indexing, right lambda_reg picked up for each term, and
# a correct sum across all of them. A single-mode fixture (as used for
# calc_l2 above) couldn't exercise this -- there'd be nothing to mix up.
#
# Fixture: GlobalMode (1 coeff: alpha_coeff) + TransportedMode (2 coeffs:
# beta_coeff, gamma_coeff) -- 3 (mode_i, coeff_name) terms total. Each given
# a DIFFERENT lambda_reg (0.3 / 0.7 / 1.5, deliberately distinct, not equal
# or round) specifically so that a bug swapping which lambda applies to
# which coefficient -- or which mode -- changes the aggregate rather than
# coincidentally cancelling out.
#
# Measured (200 random parameter draws): fp64 exactly `0.0` every time --
# machine epsilon. fp32 worst relative diff `9.55e-8`, consistent with
# ordinary fp32 roundoff (machine epsilon `~1.19e-7`) -- same lesson as
# calc_l2 applies: relative, not absolute, tolerance, sampled across many
# seeds rather than a handful. `rtol=2e-6` (~20x headroom) / `atol=1e-9`
# floor, reusing the same constants as calc_l2 since both are the same
# character of error; verified stable across 500 seeds. Detection power: a
# deliberately swapped lambda_reg between beta_coeff and gamma_coeff (0.7<->1.5)
# gives `diff=5.377`, `rel=33.9%` -- far above either bound.

def _build_multi_mode_ansatz_config():
    """GlobalMode + TransportedMode, each with a distinct lambda_reg per
    coefficient -- enough (mode_i, coeff_name) terms for calc_reg's
    aggregation loop to have something to get wrong."""
    rbf = {
        "alpha_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
        "gamma_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
    }
    ansatz = Ansatz([
        GlobalMode(_X, rbf["alpha_coeff"]),
        TransportedMode(_X, rbf["beta_coeff"], rbf["gamma_coeff"]),
    ])
    values = [
        {"pod_global": lambda z: _bump(0.3, 0.2),
         "alpha_value": lambda z: 1.0 + 0.5 * jnp.sin(z)},
        {"pod_local": lambda z: _bump(0.5, 0.1),
         "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
         "gamma_value": lambda z: 0.15 * jnp.sin(z)},
    ]
    data = DataCreation(_X, ansatz, values, _DATA_Z, rbf)

    coeff_cfg = [{"alpha_coeff": 0.3}, {"beta_coeff": 0.7, "gamma_coeff": 1.5}]
    grad_cfg = [
        {name: {"l2": True, "w1": True} for name in ansatz.modes[0].get_param_names()},
        {name: {"l2": True, "w1": True} for name in ansatz.modes[1].get_param_names()},
    ]
    ansatz_grad = OptimPhaseGradConfig(ansatz, grad_cfg)
    ansatz_coeff_reg = OptimPhaseCoeffConfig(ansatz, coeff_cfg)

    weights = LossWeights(l2=1., w2=0., penalty_exceed=0., reg_norm_pod=0.,
                          normalize=False, frobenius_or_l2_squared=1.)
    w2_config = W2Config(tau_a=1., tau_b=1., rescale=True)

    cfg = AnsatzConfig(data, weights, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg)
    return cfg, ansatz, coeff_cfg


def _independent_reg(ansatz, coeff_cfg, params_arr):
    params_dict_per_mode = ansatz.arr_to_dict(params_arr)
    total = 0.0
    for mode_i, mode in enumerate(ansatz.modes):
        RBFs = mode.get_RBF()
        for coeff_name in mode.get_coeff_names():
            lam = coeff_cfg[mode_i][coeff_name]
            total += lam * float(RBFs[coeff_name].calc_RKHS_reg(params_dict_per_mode[mode_i][coeff_name]))
    return total


def test_P_calc_reg_matches_independent_recompute():
    """calc_reg(params) == sum over every (mode, coefficient) pair of
    lambda_reg * RBF.calc_RKHS_reg(coeff), recomputed independently by
    looping the same structure in plain Python/numpy rather than calling
    calc_reg's own aggregation back at itself."""
    cfg, ansatz, coeff_cfg = _build_multi_mode_ansatz_config()
    rng = np.random.default_rng(0)
    params_arr = jnp.asarray(rng.standard_normal(ansatz.n_param()))

    result = float(cfg.calc_reg(params_arr))
    expected = _independent_reg(ansatz, coeff_cfg, params_arr)

    np.testing.assert_allclose(result, expected, atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# cut_grad -- per-(mode, param) gradient masking by loss term
# ---------------------------------------------------------------------------
#
# cut_grad(params, loss_term_name) returns params_arr UNCHANGED in VALUE
# (every block, masked or not, passes through as the identity) but with
# jax.lax.stop_gradient applied per-(mode_i, param_name) block wherever
# ansatz_grad.grad_cfg[mode_i][param_name][loss_term_name] is False. This is
# pure control flow (no arithmetic to accumulate roundoff in), so the
# expected behaviour is checked EXACTLY (assert_array_equal / exact 0.0,
# not a calibrated tolerance) -- same philosophy as N4/N11's "no
# interpolation, no tolerance to hide behind".
#
# Fixture: GlobalMode (pod_global, alpha_coeff) + TransportedMode
# (pod_local, beta_coeff, gamma_coeff) -- 5 (mode, param) blocks, giving
# cut_grad's per-mode/per-param loop real structure to get wrong (a
# single-mode fixture, or one where every block has the same mask, couldn't
# distinguish "masks correctly per block" from "masks everything the same
# way"). The mask pattern is DELIBERATELY DIFFERENT between 'l2' and 'w1'
# (not just each other's complement) specifically so a bug that reads the
# wrong loss_term_name key (e.g. always uses 'l2', or swaps the two) would
# show up as a wrong-shaped mask rather than coincidentally matching.
#
# Probe: f(p) = sum(cut_grad(p, term)**2). For an unmasked (True) block,
# cut_grad passes p through unchanged, so d/dp_i of p_i^2 = 2*p_i, exactly
# the same as if cut_grad weren't there at all. For a masked (False) block,
# stop_gradient makes that block a constant w.r.t. autodiff, so its
# gradient contribution is exactly 0, regardless of the actual value of
# p_i (including nonzero p_i) -- distinguishing "gradient blocked" from
# "coincidentally already zero".
#
# Measured (fp64 and fp32, both terms): value returned by cut_grad exactly
# equals the input params_arr (assert_array_equal, both terms); gradient
# exactly equals 2*params_arr on unmasked blocks and exactly 0.0 on masked
# blocks, max|diff| = 0.0 in every case -- bit-for-bit, no roundoff at
# either precision (expected: stop_gradient/pass-through involve no
# floating-point arithmetic of their own). Detection power: swapping which
# blocks are masked between 'l2' and 'w1' (i.e. checking 'w1''s mask
# against 'l2''s expected gradient) gives a large nonzero max|diff| at
# every masked-block index that differs between the two terms, confirming
# the two terms' masks are genuinely different and independently checked
# (not simply asserting a tautology).

def _build_cut_grad_ansatz_config():
    """GlobalMode + TransportedMode, 5 (mode, param) blocks total, with a
    grad_cfg mask that differs between 'l2' and 'w1' term-by-term."""
    rbf = {
        "alpha_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "gamma_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
    }
    ansatz = Ansatz([
        GlobalMode(_X, rbf["alpha_coeff"]),
        TransportedMode(_X, rbf["beta_coeff"], rbf["gamma_coeff"]),
    ])
    values = [
        {"pod_global": lambda z: _bump(0.3, 0.2),
         "alpha_value": lambda z: 1.0 + 0.5 * jnp.sin(z)},
        {"pod_local": lambda z: _bump(0.5, 0.1),
         "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
         "gamma_value": lambda z: 0.1 * jnp.sin(z)},
    ]
    data = DataCreation(_X, ansatz, values, _DATA_Z, rbf)

    grad_cfg = [
        {"pod_global": {"l2": True, "w1": False},
         "alpha_coeff": {"l2": True, "w1": True}},
        {"pod_local": {"l2": False, "w1": True},
         "beta_coeff": {"l2": True, "w1": False},
         "gamma_coeff": {"l2": False, "w1": True}},
    ]
    coeff_cfg = [{"alpha_coeff": 0.1}, {"beta_coeff": 0.1, "gamma_coeff": 0.1}]
    ansatz_grad = OptimPhaseGradConfig(ansatz, grad_cfg)
    ansatz_coeff_reg = OptimPhaseCoeffConfig(ansatz, coeff_cfg)

    weights = LossWeights(l2=1., w2=0., penalty_exceed=0., reg_norm_pod=0.,
                          normalize=False, frobenius_or_l2_squared=1.)
    w2_config = W2Config(tau_a=1., tau_b=1., rescale=True)

    cfg = AnsatzConfig(data, weights, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg)
    return cfg, ansatz, grad_cfg


def _mask_array(ansatz, params_arr, grad_cfg, term):
    """The full-length boolean mask (True == gradient flows) for one loss
    term, in params_arr order -- built independently from grad_cfg, not
    from cut_grad itself."""
    params_dict_per_mode = ansatz.arr_to_dict(params_arr)
    mask_dict = []
    for mode_i, mode in enumerate(ansatz.modes):
        mask_dict.append({})
        for name in mode.get_param_names():
            flag = grad_cfg[mode_i][name][term]
            mask_dict[-1][name] = jnp.full_like(params_dict_per_mode[mode_i][name], flag, dtype=bool)
    mask_arr = ansatz.dict_to_arr([
        {name: block.astype(params_arr.dtype) for name, block in md.items()} for md in mask_dict
    ])
    return mask_arr.astype(bool)


def test_P_cut_grad_passes_value_through_unchanged():
    """cut_grad's FORWARD value is the identity on every block, masked or
    not -- only the gradient is affected."""
    cfg, ansatz, grad_cfg = _build_cut_grad_ansatz_config()
    rng = np.random.default_rng(0)
    params_arr = jnp.asarray(rng.standard_normal(ansatz.n_param()))

    for term in ["l2", "w1"]:
        result = cfg.cut_grad(params_arr, term)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(params_arr))


def test_P_cut_grad_masks_gradient_exactly_per_term():
    """d/dp[sum(cut_grad(p, term)**2)] == 2*p on unmasked blocks and == 0
    exactly on masked blocks -- checked independently for 'l2' and 'w1',
    whose masks deliberately differ block-by-block (see module comment)."""
    cfg, ansatz, grad_cfg = _build_cut_grad_ansatz_config()
    rng = np.random.default_rng(0)
    params_arr = jnp.asarray(rng.standard_normal(ansatz.n_param()))

    for term in ["l2", "w1"]:
        def probe(p, term=term):
            return jnp.sum(cfg.cut_grad(p, term) ** 2)

        grad = jax.grad(probe)(params_arr)
        mask = _mask_array(ansatz, params_arr, grad_cfg, term)
        expected = jnp.where(mask, 2.0 * params_arr, 0.0)

        np.testing.assert_array_equal(np.asarray(grad), np.asarray(expected))
        # masked-out entries are exactly zero regardless of the (nonzero) param value
        assert bool(jnp.all(grad[~mask] == 0.0))


# ---------------------------------------------------------------------------
# calc_w1 -- entropic OT cost between the predicted point cloud and the true
# solution, sanity only (not full OT correctness -- that's OTT's own job)
# ---------------------------------------------------------------------------
#
# calc_w1(z, params, sol_exact) builds a PointCloud geometry between the
# predicted point cloud (get_values_nodes(z, params): positions x_pred,
# masses y_pred) and the FIXED reference mesh (self.data.x, masses
# sol_exact), both rescaled to sum to 1, and returns the entropic OT cost
# from an ott Sinkhorn solve. This is not re-deriving OT correctness (OTT
# has its own test suite for that) -- it's a sanity check that AnsatzConfig
# wires the pieces together sensibly: does the cost stay small when the
# predicted point cloud is close to the true solution, and grow as they're
# moved apart?
#
# Three params_arr per mode, deliberately increasingly different from the
# true solution: (1) DataCreation's own fitted "true" coefficients (see
# group Q -- same near-exact-fit idea); (2) a moderate shift
# (gamma/gamma_shift target = 0.5, beta pinned to a round 1.0 rather than
# the true trajectory); (3) a large shift (target = 3.0, same beta). No
# claim of exact linearity or a specific OT value -- just monotonic
# ordering, which is what "sanity" means here.
#
# Measured (fp64; fp32 agrees to ~6 sig figs -- e.g. Transported far case
# `8.682671` (fp64) vs `8.682671` (fp32) -- algorithmic Sinkhorn convergence
# behaviour under a fixed threshold/max_iterations, not roundoff):
# Transported `0.018265` (near) < `0.214989` (moderate) < `8.682671` (far);
# Stretched `0.020260` (near) < `0.242229` (moderate) < `8.852692` (far).
# Comfortably separated (near/moderate differ by ~10x, moderate/far by
# ~35x-40x) -- not a borderline ordering that could flip from run-to-run
# jitter. Detection power is inherent to the assertion itself: a calc_w1
# that didn't actually depend on the predicted point cloud's position (e.g.
# a broken cost function, or one that collapsed to a constant) would fail
# the strict-ordering assertion directly, without needing a separate
# negative control.

def _build_single_mode_ansatz_config(mode, values, rbf):
    """Like _build_multi_mode_ansatz_config, but for a single, caller-built
    mode -- shared by the calc_w1 case-builders below."""
    ansatz = Ansatz([mode])
    data = DataCreation(_X, ansatz, values, _DATA_Z, rbf)

    grad_cfg = [{name: {"l2": True, "w1": True} for name in ansatz.modes[0].get_param_names()}]
    coeff_cfg = [{name: 0.1 for name in ansatz.modes[0].get_coeff_names()}]
    ansatz_grad = OptimPhaseGradConfig(ansatz, grad_cfg)
    ansatz_coeff_reg = OptimPhaseCoeffConfig(ansatz, coeff_cfg)

    weights = LossWeights(l2=1., w2=1., penalty_exceed=0., reg_norm_pod=0.,
                          normalize=False, frobenius_or_l2_squared=1.)
    w2_config = W2Config(tau_a=1., tau_b=1., rescale=True)

    cfg = AnsatzConfig(data, weights, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg)
    return cfg, ansatz, data


def _w1_transported_cases():
    """(near, moderate, far) params_arr for TransportedMode, all at the
    same z, all compared against the same true sol_exact."""
    rbf = {
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
        "gamma_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
    }
    mode = TransportedMode(_X, rbf["beta_coeff"], rbf["gamma_coeff"])
    values = [{
        "pod_local": lambda z: _bump(0.5, 0.1),
        "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
        "gamma_value": lambda z: 0.1 * jnp.sin(z),
    }]
    cfg, ansatz, data = _build_single_mode_ansatz_config(mode, values, rbf)

    z_idx = 1
    z = data.data_z[z_idx]
    sol_exact = data.data_exact[z_idx]
    pod_local = _bump(0.5, 0.1)

    near_params = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": data.coeffs_exact["beta_coeff"],
        "gamma_coeff": data.coeffs_exact["gamma_coeff"],
    }])
    beta_round = rbf["beta_coeff"].get_coeff(jnp.ones(3))
    moderate_params = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": beta_round,
        "gamma_coeff": rbf["gamma_coeff"].get_coeff(0.5 * jnp.ones(3)),
    }])
    far_params = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": beta_round,
        "gamma_coeff": rbf["gamma_coeff"].get_coeff(3.0 * jnp.ones(3)),
    }])
    return cfg, z, sol_exact, near_params, moderate_params, far_params


def _w1_stretched_cases():
    """Same idea for StretchedMode, varying gamma_shift instead of gamma."""
    rbf = {
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
        "gamma_shift_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
        "gamma_stretch_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
    }
    mode = StretchedMode(_X, rbf["beta_coeff"], rbf["gamma_shift_coeff"], rbf["gamma_stretch_coeff"])
    values = [{
        "pod_local": lambda z: _bump(0.5, 0.1),
        "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
        "gamma_shift_value": lambda z: 0.05 * jnp.sin(z),
        "gamma_stretch_value": lambda z: 1.0 + 0.1 * jnp.sin(z),
    }]
    cfg, ansatz, data = _build_single_mode_ansatz_config(mode, values, rbf)

    z_idx = 1
    z = data.data_z[z_idx]
    sol_exact = data.data_exact[z_idx]
    pod_local = _bump(0.5, 0.1)

    near_params = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": data.coeffs_exact["beta_coeff"],
        "gamma_shift_coeff": data.coeffs_exact["gamma_shift_coeff"],
        "gamma_stretch_coeff": data.coeffs_exact["gamma_stretch_coeff"],
    }])
    beta_round = rbf["beta_coeff"].get_coeff(jnp.ones(3))
    stretch_round = rbf["gamma_stretch_coeff"].get_coeff(jnp.ones(3))
    moderate_params = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": beta_round,
        "gamma_shift_coeff": rbf["gamma_shift_coeff"].get_coeff(0.5 * jnp.ones(3)),
        "gamma_stretch_coeff": stretch_round,
    }])
    far_params = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": beta_round,
        "gamma_shift_coeff": rbf["gamma_shift_coeff"].get_coeff(3.0 * jnp.ones(3)),
        "gamma_stretch_coeff": stretch_round,
    }])
    return cfg, z, sol_exact, near_params, moderate_params, far_params


def test_P_calc_w1_transported_increases_as_point_clouds_separate():
    """calc_w1 should be small for a near-true reconstruction and increase
    as the predicted point cloud is moved further from the true solution --
    sanity only, not full OT correctness (that's OTT's own job)."""
    cfg, z, sol_exact, near_params, moderate_params, far_params = _w1_transported_cases()

    w1_near = float(cfg.calc_w1(z, near_params, sol_exact))
    w1_moderate = float(cfg.calc_w1(z, moderate_params, sol_exact))
    w1_far = float(cfg.calc_w1(z, far_params, sol_exact))

    assert w1_near < w1_moderate < w1_far, (w1_near, w1_moderate, w1_far)


def test_P_calc_w1_stretched_increases_as_point_clouds_separate():
    """Same check for StretchedMode."""
    cfg, z, sol_exact, near_params, moderate_params, far_params = _w1_stretched_cases()

    w1_near = float(cfg.calc_w1(z, near_params, sol_exact))
    w1_moderate = float(cfg.calc_w1(z, moderate_params, sol_exact))
    w1_far = float(cfg.calc_w1(z, far_params, sol_exact))

    assert w1_near < w1_moderate < w1_far, (w1_near, w1_moderate, w1_far)


# ---------------------------------------------------------------------------
# calc_w1_training -- aggregate OT cost across all training snapshots
# ---------------------------------------------------------------------------
#
# calc_w1_training(params) = sum over every training z of calc_w1(z, params,
# data_exact_z) (jax.vmap'd internally, power=1., no extra squaring -- unlike
# calc_l2_training, there's no frobenius_or_l2_squared knob here). calc_w1
# itself is already sanity-checked above (near/far ordering); this is NOT
# about re-checking that -- it's about whether AnsatzConfig's aggregation
# (vmap + sum across the 3 training z's) matches calling calc_w1 per z in a
# plain Python loop and summing by hand.
#
# Measured (fp64, all three near/moderate/far params reused from the calc_w1
# cases above): diff against the independent per-z loop is `<1.2e-16` in
# every case (`3.47e-17`, `1.11e-16`, `0.0`) -- machine epsilon, i.e. vmap's
# batched Sinkhorn solve and a sequential per-z loop agree essentially
# exactly. fp32: worst relative diff `9.78e-7` (Transported) / `1.44e-7`
# (Stretched) -- consistent with ordinary fp32 roundoff from a different
# summation/reduction order between the batched (vmap) and sequential
# paths, not a real discrepancy. `rtol=1e-5` (~10x headroom over the fp32
# worst case) / `atol=1e-9` floor, reused across both modes' three cases.
#
# Detection power: a plausible aggregation bug (using the mean instead of
# the sum across snapshots) gives `diff=0.437` -- far above tolerance;
# dropping one snapshot from the sum entirely gives `diff=0.249`, also far
# above. (A cyclic permutation of which data_exact pairs with which z was
# tried first and rejected as a negative control -- for this 3-snapshot
# fixture the total happened to come out nearly unchanged under that
# specific permutation, which would have been a misleading, non-informative
# "detection power" claim; the mean/dropped-snapshot controls above are
# genuine.)

_W1_TRAINING_ATOL = 1e-9
_W1_TRAINING_RTOL = 1e-5


def _independent_w1_training(cfg, params_arr):
    data = cfg.data
    return sum(
        float(cfg.calc_w1(data.data_z[zi], params_arr, data.data_exact[zi]))
        for zi in range(data.data_z.shape[0])
    )


def test_P_calc_w1_training_transported_matches_independent_sum():
    """calc_w1_training(params) == sum over every training z of calc_w1(z,
    params, data_exact_z), recomputed independently via a plain Python loop
    rather than calling calc_w1_training's own vmap'd formula back at
    itself. Checked at all three (near/moderate/far) params."""
    cfg, z, sol_exact, near_params, moderate_params, far_params = _w1_transported_cases()

    for params in [near_params, moderate_params, far_params]:
        result = float(cfg.calc_w1_training(params))
        expected = _independent_w1_training(cfg, params)
        np.testing.assert_allclose(result, expected, atol=_W1_TRAINING_ATOL, rtol=_W1_TRAINING_RTOL)


def test_P_calc_w1_training_stretched_matches_independent_sum():
    """Same check for StretchedMode."""
    cfg, z, sol_exact, near_params, moderate_params, far_params = _w1_stretched_cases()

    for params in [near_params, moderate_params, far_params]:
        result = float(cfg.calc_w1_training(params))
        expected = _independent_w1_training(cfg, params)
        np.testing.assert_allclose(result, expected, atol=_W1_TRAINING_ATOL, rtol=_W1_TRAINING_RTOL)


# ---------------------------------------------------------------------------
# cost_jit / update_scaling -- normalization constants must not be baked
# into the jit trace
# ---------------------------------------------------------------------------
#
# `update_scaling(params_arr)` sets `self.l2_standard = max(calc_l2_training
# (params_arr), _l2_standard_min)` and `self.w2_standard = max(calc_w1_
# training(params_arr), _w2_standard_min)`. `cost_jit(params_arr)` calls the
# jitted `_cost(params_arr, self.l2_standard, self.w2_standard)`
# (`@jax.jit(static_argnames=['self'])`). In the UNREGULARIZED case (every
# coefficient's lambda_reg == 0, so calc_reg == 0; reg_norm_pod ==
# penalty_exceed == 0, so calc_pod_norms_reg/calc_exceed_training don't
# contribute either; weights_loss.normalize=True, l2=w2=1.) and with the
# floor not binding, `_cost` reduces to exactly
# `l2_loss/l2_standard + w2_loss/w2_standard`, where `l2_loss`/`w2_loss` are
# (per cut_grad's forward-value-is-identity property, see above)
# value-identical to the same `calc_l2_training`/`calc_w1_training` used by
# `update_scaling` to set `l2_standard`/`w2_standard` -- so this should come
# out to EXACTLY `1. + 1. == 2.` (IEEE 754 x/x == 1.0 exactly for finite
# nonzero x), not merely "close to 2".
#
# The real risk being tested: `self` is a STATIC jit argument to `_cost`.
# If a future refactor read `self.l2_standard`/`self.w2_standard` directly
# inside `_cost` (instead of `cost_jit` passing them in as ordinary, traced
# arguments, as the code does today), those values would get baked into the
# compiled XLA graph at first-trace time and NOT refresh on a later call
# with the same `self` object, even after `update_scaling` mutates them --
# because JAX's jit cache keys on `self`'s (identity-based) hash, which
# doesn't change when its attributes are mutated. This test calls
# `update_scaling` twice, with two DELIBERATELY DIFFERENT params (so their
# calc_l2_training/calc_w1_training values differ by ~2700x/~470x -- see
# measured values below), and checks `cost_jit == 2.` after EACH call --
# a stale/baked-in first-call scaling would make the second `cost_jit` check
# fail loudly (not coincidentally still equal 2.).
#
# Measured (both fp32 and fp64, bit-for-bit identical to the printed
# precision): params_1 (DataCreation's own near-exact fitted coefficients)
# gives calc_l2_training=3.977e-4, calc_w1_training=5.526e-2, both above
# their floors (_l2_standard_min=1.071e-4, _w2_standard_min=1e-3 -- floor
# does not bind); params_2 (beta pinned to a round trajectory, gamma target
# =3.0, same as the calc_w1 "far" case) gives calc_l2_training=1.071,
# calc_w1_training=26.10 -- ~2700x/~470x larger than params_1's, floor
# again does not bind. `cost_jit(params_1)` after `update_scaling(params_1)`
# == `2.0` exactly; `cost_jit(params_2)` after `update_scaling(params_2)`
# == `2.0` exactly, at both precisions -- confirming the normalization
# constants are correctly refreshed, not baked into the first trace.
# Detection power: calling `cost_jit(params_1)` again AFTER
# `update_scaling(params_2)` (i.e. deliberately using the WRONG/stale
# scaling) gives `0.00249`, nowhere near `2.` -- confirming this fixture's
# params_1/params_2 pair is sensitive enough that a real staleness bug
# would be caught, not masked by coincidence.

def _build_unregularized_transported_config():
    """Single TransportedMode, every lambda_reg == 0 (unregularized),
    reg_norm_pod == penalty_exceed == 0, normalize=True, l2=w2=1. -- the
    minimal setup for which cost_jit(params) == 2. exactly right after
    update_scaling(params) (floor not binding)."""
    rbf = {
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
        "gamma_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
    }
    mode = TransportedMode(_X, rbf["beta_coeff"], rbf["gamma_coeff"])
    values = [{
        "pod_local": lambda z: _bump(0.5, 0.1),
        "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
        "gamma_value": lambda z: 0.1 * jnp.sin(z),
    }]
    ansatz = Ansatz([mode])
    data = DataCreation(_X, ansatz, values, _DATA_Z, rbf)

    grad_cfg = [{name: {"l2": True, "w1": True} for name in mode.get_param_names()}]
    coeff_cfg = [{name: 0.0 for name in mode.get_coeff_names()}]  # unregularized
    ansatz_grad = OptimPhaseGradConfig(ansatz, grad_cfg)
    ansatz_coeff_reg = OptimPhaseCoeffConfig(ansatz, coeff_cfg)

    weights = LossWeights(l2=1., w2=1., penalty_exceed=0., reg_norm_pod=0.,
                          normalize=True, frobenius_or_l2_squared=1.)
    w2_config = W2Config(tau_a=1., tau_b=1., rescale=True)

    cfg = AnsatzConfig(data, weights, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg)

    pod_local = _bump(0.5, 0.1)
    params_1 = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": data.coeffs_exact["beta_coeff"],
        "gamma_coeff": data.coeffs_exact["gamma_coeff"],
    }])
    beta_round = rbf["beta_coeff"].get_coeff(jnp.ones(3))
    params_2 = ansatz.dict_to_arr([{
        "pod_local": pod_local,
        "beta_coeff": beta_round,
        "gamma_coeff": rbf["gamma_coeff"].get_coeff(3.0 * jnp.ones(3)),
    }])
    return cfg, params_1, params_2


def test_P_cost_jit_equals_two_after_update_scaling_for_each_new_params():
    """In the unregularized case, cost_jit(params) == 2. right after
    update_scaling(params) -- checked for TWO DIFFERENT params (params_1,
    then params_2), each via its own update_scaling call, to confirm
    l2_standard/w2_standard are not stale/baked into cost_jit's jit trace
    from the first call (see module comment for why this is a real risk:
    `_cost` is jitted with `self` static)."""
    cfg, params_1, params_2 = _build_unregularized_transported_config()

    assert float(cfg.calc_l2_training(params_1)) > float(cfg._l2_standard_min)
    assert float(cfg.calc_w1_training(params_1)) > float(cfg._w2_standard_min)
    assert float(cfg.calc_l2_training(params_2)) > float(cfg._l2_standard_min)
    assert float(cfg.calc_w1_training(params_2)) > float(cfg._w2_standard_min)

    cfg.update_scaling(params_1)
    assert float(cfg.cost_jit(params_1)) == 2.0

    cfg.update_scaling(params_2)
    assert float(cfg.cost_jit(params_2)) == 2.0


# ---------------------------------------------------------------------------
# calc_pod_norms_reg -- penalizes each POD mode's discrete L2 norm deviating
# from 1
# ---------------------------------------------------------------------------
#
# calc_pod_norms_reg(params) = sum over (mode_i, pod_name) of
#     (dx * sum(pod_local_or_global^2) - 1.)^2
#
# Unweighted in practice (weights_loss.reg_norm_pod == 0. everywhere it's
# used), but the formula itself was untested. Reuses
# `_build_multi_mode_ansatz_config` (GlobalMode's `pod_global` +
# TransportedMode's `pod_local`, 2 `(mode_i, pod_name)` terms total) already
# defined above for the `calc_reg` tests -- same rationale: a single-mode
# fixture couldn't exercise the per-mode aggregation loop, since there'd be
# nothing to sum together or drop by mistake.
#
# Measured (5000 random parameter draws, both fp32 and fp64): worst absolute
# diff `1.907e-6` (fp32) / machine-epsilon at fp64; worst RELATIVE diff
# `3.812e-5` (fp32), occurring at a draw where the true value happens to be
# tiny (`~8.65e-5`, i.e. `dx*sum(pod^2)` landed close to `1.` by chance) --
# same cancellation pattern documented in `future.md` section 9: relative
# error blows up near a near-zero expected value, absolute error blows up at
# larger-magnitude draws, so both bounds are needed together (`atol=1e-5`,
# ~5.2x headroom over the worst absolute case; `rtol=2e-4`, ~5.2x headroom
# over the worst relative case), not just one or the other. Verified 0
# failures out of 5000 seeds at both precisions with these bounds. Detection
# power: dropping one mode's term from the sum entirely gives `diff=0.0584`;
# forgetting the `-1.` centering (i.e. `(dx*sum(pod^2))^2` instead of
# `(dx*sum(pod^2)-1.)^2`) gives `diff=0.886` -- both far above tolerance.

_POD_NORMS_REG_ATOL = 1e-5
_POD_NORMS_REG_RTOL = 2e-4


def _independent_pod_norms_reg(ansatz, dx, params_arr):
    params_dict_per_mode = ansatz.arr_to_dict(params_arr)
    total = 0.0
    for mode_i, mode in enumerate(ansatz.modes):
        for pod_name in mode.get_pod_names():
            pod = np.asarray(params_dict_per_mode[mode_i][pod_name])
            total += (float(dx) * float(np.sum(pod ** 2)) - 1.) ** 2
    return total


def test_P_calc_pod_norms_reg_matches_independent_recompute():
    """calc_pod_norms_reg(params) == sum over every (mode, pod-shape
    parameter) of (dx*sum(pod^2) - 1)^2, recomputed independently in plain
    numpy rather than calling calc_pod_norms_reg's own aggregation back at
    itself."""
    cfg, ansatz, coeff_cfg = _build_multi_mode_ansatz_config()
    rng = np.random.default_rng(0)
    params_arr = jnp.asarray(rng.standard_normal(ansatz.n_param()))

    result = float(cfg.calc_pod_norms_reg(params_arr))
    expected = _independent_pod_norms_reg(ansatz, cfg.data.dx, params_arr)

    np.testing.assert_allclose(result, expected, atol=_POD_NORMS_REG_ATOL, rtol=_POD_NORMS_REG_RTOL)


# ---------------------------------------------------------------------------
# calc_w1_training's Hessian -- DELIBERATELY, PERMANENTLY FAILING
# ---------------------------------------------------------------------------
#
# Investigated on request ("design and calibrate a Hessian test for
# calc_w1_training"). Finding, not a coverage gap: `jax.hessian
# (calc_w1_training)` does NOT compute the true second derivative.
#
# The GRADIENT is fine -- `jax.grad(calc_w1_training)` matches central FD to
# 4-5 significant figures at ordinary step sizes (`h~0.01-0.1`; e.g. index 10
# of the `near_params` fixture below: autodiff `-0.0047791`, FD
# `-0.0047795` to `-0.0047807` across that range). The HESSIAN does not:
# picking three entries in the `gamma_coeff` block (local indices 1, 2, 3;
# found via `_flat_param_index` below rather than hardcoded, since the flat
# array's block order comes from dict insertion order, not
# `get_param_names()`), central FD is a rock-solid PLATEAU across a full
# decade of step sizes (`h=0.03` to `0.15`, fp32 AND fp64, agreeing to 5+
# significant figures at every step -- this is not FD noise, it has
# genuinely converged), yet disagrees with `jax.hessian` by a consistent
# ~1.49x factor at all three entries (fp64: auto `2.7598699` vs FD
# `4.1091697`; auto `2.1542508` vs FD `3.2091943`; auto `4.0264791` vs FD
# `6.0` -- ratios `1.4889`/`1.4897`/`1.4901`, i.e. the SAME systematic factor
# to within 0.1%, not three unrelated numbers).
#
# Root cause (hypothesis, not further verified against OTT's internals):
# `calc_w1`'s underlying `linear.solve` call uses OTT's Sinkhorn solver with
# its DEFAULT `implicit_diff=ImplicitDiff(...)` (confirmed via
# `inspect.signature(ott.solvers.linear.sinkhorn.Sinkhorn.__init__)`) plus a
# Danskin's-theorem-style envelope trick. Both give a PROVABLY CORRECT first
# derivative by treating the converged dual potentials as locally constant
# w.r.t. the outer parameters -- but that's only valid to first order; the
# terms it drops (the potentials' own sensitivity to the parameters) reappear
# and matter for the SECOND derivative. Nesting `jax.hessian` differentiates
# through that first-order-only backward rule, which would produce exactly
# this signature: correct gradient, systematically wrong Hessian.
#
# Separately (not exercised by this test, to keep its failure clean rather
# than a NaN crash, but worth recording): perturbing `pod_local[0]`
# downward -- a near-zero mass node, `~3.7e-6` in this fixture, one of the
# Gaussian bump's tails -- gives `calc_w1_training == NaN` at EVERY step size
# tried (`h=1e-1` down to `1e-6`), even though the upward perturbation is
# fine. Likely a negative predicted mass reaching Sinkhorn's log-domain
# solve; a real fragility of `calc_w1`/`calc_w1_training` independent of the
# Hessian question, and worth a look if this test suite ever exercises
# `AnsatzConfig` near a POD mode with near-zero amplitude at some node.
#
# Both findings are also written up in `future.md`. This test is left
# failing ON PURPOSE, matching group N's `test_N8_..._corner_gradient_
# matches_fd` convention: NOT marked `xfail` (an `xfail(strict=True)` would
# report as an expected/passing result in test-run summaries), so it stays a
# standing, impossible-to-miss red result in every run until either the
# design decision changes (e.g. switching `calc_w1`'s Sinkhorn solve to
# unrolled/explicit differentiation for correct higher-order derivatives --
# at a real performance cost) or `AnsatzConfig` is confirmed to never need
# second derivatives of the W_1 term in practice.

_W1_HESS_FD_H = 0.1
# NOT a small step -- see comment above. Sinkhorn's threshold=0.001
# convergence criterion creates an algorithmic noise floor several orders of
# magnitude above floating-point roundoff; the tiny steps (1e-4 to 1e-7)
# used elsewhere in this suite's FD checks give garbage here (central FD
# diverges wildly as h shrinks below ~0.02), while h in [0.03, 0.15] gives a
# converged, h-independent plateau (see measurements above).


def _flat_param_index(ansatz, params_arr, mode_i, param_name, local_idx):
    """The flat params_arr index corresponding to (mode_i, param_name,
    local_idx) -- computed via a one-hot dict round-tripped through
    dict_to_arr, rather than hardcoded, since the flat array's block order
    follows dict insertion order (not `get_param_names()`)."""
    zero_dict = [
        {name: jnp.zeros_like(block) for name, block in mode_dict.items()}
        for mode_dict in ansatz.arr_to_dict(params_arr)
    ]
    zero_dict[mode_i][param_name] = zero_dict[mode_i][param_name].at[local_idx].set(1.0)
    onehot = ansatz.dict_to_arr(zero_dict)
    return int(jnp.argmax(jnp.abs(onehot)))


def _central_fd_hessian_diag_entry(f, params, i, h):
    e = jnp.zeros_like(params).at[i].set(1.0)
    f0 = float(f(params))
    fp = float(f(params + h * e))
    fm = float(f(params - h * e))
    return (fp - 2. * f0 + fm) / h ** 2


def test_P_calc_w1_training_hessian_matches_fd():
    """Deliberately, permanently failing (see comment above): jax.hessian
    (calc_w1_training) disagrees with a converged finite-difference estimate
    by a stable ~1.49x systematic factor -- not roundoff, not FD noise
    (verified h-independent across a full decade of step sizes, both
    precisions). Left failing on purpose, NOT xfail, as a standing reminder
    that OTT's implicit/Danskin Sinkhorn differentiation is validated for
    first derivatives only."""
    cfg, z, sol_exact, near_params, moderate_params, far_params = _w1_transported_cases()
    f = lambda p: cfg.calc_w1_training(p)

    for local_idx in [1, 2, 3]:  # gamma_coeff block, avoiding pod_local's near-zero-mass nodes
        idx = _flat_param_index(cfg.ansatz, near_params, 0, "gamma_coeff", local_idx)
        hess_diag_auto = float(jax.hessian(f)(near_params)[idx, idx])
        hess_diag_fd = _central_fd_hessian_diag_entry(f, near_params, idx, _W1_HESS_FD_H)
        np.testing.assert_allclose(hess_diag_auto, hess_diag_fd, rtol=1e-2)

