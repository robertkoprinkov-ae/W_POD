"""Math-tier self-consistency test (group Q in unit_tests.md).

Builds a `DataCreation` from known closed-form coefficient trajectories
(matching group H/P's fixture style, kept self-contained here per this
suite's no-cross-file-test-imports convention), then evaluates
`AnsatzConfig.calc_l2_training` at the "true" parameter vector -- the same
coefficients `DataCreation` itself fit via `RBF.get_coeff()` (stored as
`DataCreation.coeffs_exact`) combined with the same closed-form `pod_local`
array used to generate the ground truth.

This is a genuinely different check from group P's `calc_l2`/
`calc_l2_training` tests (which verify the LOSS FORMULA against an
independently-recomputed value, for an ARBITRARY params_arr) and from group
N7 (which checks a single mode's `get_values_grid` against
`get_values_grid_exact` at one z). Group Q checks the FULL forward-model
pipeline end-to-end, aggregated across every training snapshot, in ONE
number: does `DataCreation`'s own ground-truth generation
(`get_values_grid_exact`, exact scalars + linear interp) agree with the
`Ansatz`'s reconstruction (`get_values_grid`, RBF-predicted coefficients +
catmull-rom) closely enough that the aggregate loss is small? This is
cheaper and more targeted than a full `test_integration_unregularized.py`
run for catching a broken forward model (e.g. a sign-convention mismatch
between the two reconstruction paths, or the two paths disagreeing about
which mode owns which coefficient).

"Should be ~0", not EXACTLY 0 -- but NOT for the reason first assumed here.
An earlier draft of this file used a coarse, 21-point mesh and attributed
the nonzero residual to `RBF.get_coeff()`'s nugget/ridge-regression effect
(ordinary fitting, not the `bias=True`+constant-target exact-coefficient
trick used to isolate pure geometry in groups N1/N5/N7). That explanation
was WRONG, caught by deliberately shrinking the nugget to `1e-9` (in
`DataCreation`) and checking whether the residual shrank proportionally --
it didn't: `RBF.predict()` matched its target to `~1e-9`-`1e-11` (confirming
the nugget change worked), yet `calc_l2_training` at the true params stayed
at `~3.98e-4`, unchanged. Isolating the cause directly (`get_values_grid`
using the near-exact RBF coefficients vs. `get_values_grid_exact` fed the
TRUE closed-form scalars, bypassing the RBF entirely) reproduced that exact
`~3.98e-4` residual with ZERO RBF involvement -- proving the entire residual
was the INTERPOLATION-SCHEME difference (`get_values_grid`'s catmull-rom vs.
`get_values_grid_exact`'s linear `jnp.interp`), amplified by the 21-point
mesh's coarse spacing (`dx=0.05`) and `gamma` not landing on an exact mesh
multiple (unlike N1's deliberately mesh-aligned fixture) -- consistent with
N5's own `O(h^2)` linear-interpolation truncation bound, just not obviously
so until measured directly.

Refining the mesh to 101 points (`dx=0.01`, matching N1/N5/N10's own
convention) confirms this: `calc_l2_training` at the true params dropped
from `~3.98e-4` to `7.2108e-7` (Transported) / `5.5390e-7` (Stretched) -- a
~550x-680x reduction, matching the `O(h^4)` aggregate-loss scaling predicted
from N5's `O(h^2)` per-point bound (`5^4=625` predicted for a 5x finer mesh,
vs. `~550`-`680` measured) almost exactly. The residual genuinely is now
dominated by the (much smaller, at this resolution) interpolation-scheme
gap, not nugget/RBF noise -- `future.md` section 6 already establishes RBF
noise as the LARGER term only when both are present at typical nugget
values (`~1e-3`); this file's fixture, deliberately using a near-zero
nugget, isolates the OTHER term instead.

Measured (both fp64; fp32 agrees to ~5-6 sig figs -- e.g. `7.2108e-7` vs
`7.2108e-7` for Transported -- confirming this is algorithmic, not
roundoff, even at this much smaller magnitude): Transported
`calc_l2_training`=`7.2108e-7`, Stretched=`5.5390e-7`. For scale, an
arbitrary (wrong) `params_arr` gives `1.2042` (Transported) -- ~1.67 MILLION
times larger. `atol=1e-6` (~1.4x-1.8x headroom over both measured values --
deliberately tight; if this fixture's residual ever drifts upward for a
legitimate reason, this is the first tolerance that would need revisiting).
No `rtol`: both this check and the gradient check below are plain `<`/`>`
threshold comparisons against a fixed value, not
`np.testing.assert_allclose`, so there is no relative-tolerance component.
Detection power: a deliberately sign-flipped `gamma_shift_coeff`
(Stretched) gives `calc_l2_training=0.1612` -- ~161000x above atol,
confirming this check would catch a genuine forward-model bug (e.g. a
sign-convention mismatch between `get_values_grid` and
`get_values_grid_exact`), not just rubber-stamp whatever the two paths
happen to produce.

**Gradient extension.** `calc_l2_training` (with `frobenius_or_l2_squared=1.`,
see group P) is a sum of SQUARED residual norms, i.e. "L2 squared", not the
L2 norm itself. At the true params it's not just small in VALUE -- its
gradient is also small, i.e. the true params sit close to a stationary point
of this loss too (not merely of the RBF's own separate internal fitting
objective): a squared loss `u(p)^2` has gradient `2 u(p) u'(p)`, which
vanishes smoothly as `u(p) -> 0` regardless of `u'(p)`'s behaviour (unlike
the plain L2 norm, whose gradient `u(p) u'(p) / |u(p)|` stays large as
`u(p) -> 0` unless `u'(p)` also vanishes -- the classic reason squared
residuals, not norms, are used as optimization objectives). Measured:
`||grad(calc_l2_training)||` at true params is `8.6270e-4` (Transported) /
`1.3299e-3` (Stretched) -- ~767x/10250x smaller than the same gradient's
norm at an arbitrary (wrong) params_arr (`0.6623` / `13.631`), and fp32
matches fp64 to ~5-6 sig figs (algorithmic, not roundoff). `atol=5e-3`
(~3.8x-5.8x headroom over both measured values); no `rtol`, same reason as
above.
"""

import numpy as np
import jax
import jax.numpy as jnp

from W_POD.modes import TransportedMode, StretchedMode
from W_POD.util import RBF_interpol, DataCreation
from W_POD.ansatz import (
    Ansatz, AnsatzConfig, LossWeights, W2Config,
    OptimPhaseGradConfig, OptimPhaseCoeffConfig,
)

_ATOL = 1e-6

_GRAD_NEAR_ZERO_ATOL = 5e-3    # L2^2 gradient norm should be well below this

_X = jnp.linspace(0.0, 1.0, 101)   # matches N1/N5/N10's convention -- see module docstring
_DATA_Z = jnp.array([0.2, 0.6, 1.0])


def _bump(center, width):
    return jnp.exp(-0.5 * ((_X - center) / width) ** 2)


def _build_ansatz_config(mode, values, rbf):
    ansatz = Ansatz([mode])
    data = DataCreation(_X, ansatz, values, _DATA_Z, rbf)

    grad_cfg = [{name: {"l2": True, "w1": True} for name in ansatz.modes[0].get_param_names()}]
    coeff_cfg = [{name: 0.1 for name in ansatz.modes[0].get_coeff_names()}]
    ansatz_grad = OptimPhaseGradConfig(ansatz, grad_cfg)
    ansatz_coeff_reg = OptimPhaseCoeffConfig(ansatz, coeff_cfg)

    weights = LossWeights(l2=1., w2=0., penalty_exceed=0., reg_norm_pod=0.,
                          normalize=False, frobenius_or_l2_squared=1.)
    w2_config = W2Config(tau_a=1., tau_b=1., rescale=True)

    cfg = AnsatzConfig(data, weights, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg)
    return cfg, ansatz, data


def _transported_fixture():
    rbf = {
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "gamma_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
    }
    mode = TransportedMode(_X, rbf["beta_coeff"], rbf["gamma_coeff"])
    values = [{
        "pod_local": lambda z: _bump(0.5, 0.1),
        "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
        "gamma_value": lambda z: 0.1 * jnp.sin(z),
    }]
    cfg, ansatz, data = _build_ansatz_config(mode, values, rbf)

    true_pod_local = values[0]["pod_local"](0.0)  # z-independent
    true_params = ansatz.dict_to_arr([{
        "pod_local": true_pod_local,
        "beta_coeff": data.coeffs_exact["beta_coeff"],
        "gamma_coeff": data.coeffs_exact["gamma_coeff"],
    }])
    return cfg, ansatz, data, true_params


def _stretched_fixture():
    rbf = {
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "gamma_shift_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "gamma_stretch_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
    }
    mode = StretchedMode(_X, rbf["beta_coeff"], rbf["gamma_shift_coeff"], rbf["gamma_stretch_coeff"])
    values = [{
        "pod_local": lambda z: _bump(0.5, 0.1),
        "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
        "gamma_shift_value": lambda z: 0.05 * jnp.sin(z),
        "gamma_stretch_value": lambda z: 1.0 + 0.1 * jnp.sin(z),  # never ~0
    }]
    cfg, ansatz, data = _build_ansatz_config(mode, values, rbf)

    true_pod_local = values[0]["pod_local"](0.0)
    true_params = ansatz.dict_to_arr([{
        "pod_local": true_pod_local,
        "beta_coeff": data.coeffs_exact["beta_coeff"],
        "gamma_shift_coeff": data.coeffs_exact["gamma_shift_coeff"],
        "gamma_stretch_coeff": data.coeffs_exact["gamma_stretch_coeff"],
    }])
    return cfg, ansatz, data, true_params


def test_Q_transported_forward_model_self_consistent_at_true_params():
    """TransportedMode: DataCreation's own ground-truth generation should
    agree with the Ansatz's reconstruction at DataCreation's own fitted
    ("true") coefficients -- aggregate calc_l2_training should be small."""
    cfg, ansatz, data, true_params = _transported_fixture()

    l2_train = float(cfg.calc_l2_training(true_params))
    assert l2_train < _ATOL, f"calc_l2_training at true params = {l2_train} >= {_ATOL}"


def test_Q_stretched_forward_model_self_consistent_at_true_params():
    """Same check for StretchedMode."""
    cfg, ansatz, data, true_params = _stretched_fixture()

    l2_train = float(cfg.calc_l2_training(true_params))
    assert l2_train < _ATOL, f"calc_l2_training at true params = {l2_train} >= {_ATOL}"


def test_Q_transported_l2_squared_gradient_near_zero_at_true_params():
    """calc_l2_training's (autodiff) gradient should be near zero at the
    true params -- not just the loss VALUE. A squared-residual loss has
    gradient 2*u(p)*u'(p), which vanishes smoothly as u(p) -> 0 regardless
    of u'(p)."""
    cfg, ansatz, data, true_params = _transported_fixture()

    grad_norm = float(jnp.linalg.norm(jax.grad(cfg.calc_l2_training)(true_params)))
    assert grad_norm < _GRAD_NEAR_ZERO_ATOL, f"grad norm {grad_norm} >= {_GRAD_NEAR_ZERO_ATOL}"


def test_Q_stretched_l2_squared_gradient_near_zero_at_true_params():
    """Same check for StretchedMode."""
    cfg, ansatz, data, true_params = _stretched_fixture()

    grad_norm = float(jnp.linalg.norm(jax.grad(cfg.calc_l2_training)(true_params)))
    assert grad_norm < _GRAD_NEAR_ZERO_ATOL, f"grad norm {grad_norm} >= {_GRAD_NEAR_ZERO_ATOL}"
