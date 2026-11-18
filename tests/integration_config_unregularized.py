"""The integration-test config contract -- Formulation 3.5 (UNREGULARIZED) only.

This config uses theta ~ 0 and lambda_reg = 0 throughout (see
make_integration_case()), i.e. it exercises the paper's Formulation 3.5 --
snapshot-by-snapshot fitting, no RKHS regularization. It does NOT exercise the
RKHS-regularized Formulation 3.8: no config here calibrates a meaningful kernel
length scale or reg weight, and there is no interpolation-error check (3.8's
distinguishing benefit, per the paper, shows up in interpolation at held-out z
and in otherwise-ill-posed problems like Example 3.7 "crossing waves" -- neither
of which this config is designed to probe). A green result here says nothing
about the regularized formulation.

A separate `integration_config_regularized.py` / `test_integration_regularized.py`
pair is planned for Formulation 3.8, with its own (more delicate) hyperparameter
choices and its own interpolation-based acceptance criteria -- not a variant of
this file.

``tests/test_integration_unregularized.py`` is written against ``IntegrationCase``
and ``make_integration_case()`` below. The test logic is already implemented;
only ``make_integration_case()`` needed to be filled in with a concrete
small/fast synthetic problem, which it now is.

See the chat discussion (also summarised in CODEBASE_NOTES.md / future.md) for
why each piece is needed.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from W_POD.util import RBF_interpol, DataCreation
from W_POD.modes import TransportedMode
from W_POD.ansatz import (
    Ansatz, AnsatzConfig, LossWeights, W2Config,
    OptimPhaseGradConfig, OptimPhaseCoeffConfig,
)
from W_POD.solver import OptimCfg, IteratedSolver


@dataclass
class IntegrationCase:
    data: "W_POD.util.DataCreation"
    ansatz: "W_POD.ansatz.Ansatz"
    solver: "W_POD.solver.IteratedSolver"          # already wired: ansatz_cfgs + optim_cfgs + n_iter
    initial: jax.Array                             # deterministic starting params, ansatz.dict_to_arr(...)

    # --- acceptance criteria -------------------------------------------------
    max_recon_error: float                         # relative Frobenius error the solve must reach
    gamma_tol: Optional[float] = None               # None -> skip the gamma-recovery check
    n_pod_modes_for_baseline: Optional[int] = None   # None -> len(ansatz.modes)


def make_integration_case() -> IntegrationCase:
    """Build the small/fast Formulation-3.5 (unregularized) problem used by
    tests/test_integration_unregularized.py.

    A concrete, DETERMINISTIC, fast (seconds) single-TransportedMode problem:
    x has 100 points, 5 snapshots, gamma(z) = 0.1 + 0.2*z (monotonic and
    nonzero at every sampled z, so the transport is real -- an earlier draft
    used a periodic gamma sampled at its own zero-crossings, which made the
    snapshot matrix accidentally near-rank-1 and let plain POD win).

    Recipe below kept as reference for building the Formulation 3.8
    (regularized) config later (see CODEBASE_NOTES.md sec 1-2 and
    tests/advdiff.py for a full-scale worked example to shrink down):

    1. Problem (DataCreation(x, ansatz_exact, ansatz_values, data_z, RBF)):
       - x: a small 1D mesh, e.g. jnp.linspace(0., 1., ~30-50 pts)
       - data_z: ~4-6 snapshot parameter values
       - ansatz_values: closed-form ground-truth pod / alpha / beta / gamma
         functions of z, one dict per mode (see modes.py's get_values_grid_exact
         for the exact key names each mode needs, e.g. TransportedMode needs
         'pod_local', 'beta_value', 'gamma_value')
       - RBF: one RBF_interpol per coefficient name, matching data_z

    2. Model (Ansatz([...])): which modes, and one RBF_interpol per coefficient
       (theta, sigma, bias, kernel, nugget) -- can reuse the RBFs from step 1
       or use separate ones for the *fitted* ansatz.

    3. AnsatzConfig(data, weights_loss, w2_config, ansatz, ansatz_grad, ansatz_coeff_reg):
       - LossWeights(l2, w2, penalty_exceed, reg_norm_pod, normalize,
         frobenius_or_l2_squared=1.)
       - W2Config(tau_a, tau_b, rescale)          # (1., 1., True) = balanced OT
       - OptimPhaseGradConfig(ansatz, grad_cfg)   # per mode: {param_name: {'l2': bool, 'w1': bool}}
       - OptimPhaseCoeffConfig(ansatz, coeff_cfg) # per mode: {coeff_name: lambda_reg}

    4. OptimCfg(stop_if_linesearch_fails, verbose=False, log_w1, maxiter,
       scipy_optim_method='SLSQP') -- keep maxiter small (~20-50) for test speed.

    5. IteratedSolver(data, [ansatz_cfg], [optim_cfg], n_iter=1 or 2).

    6. initial: build a per-mode params dict (mode init shape, coeff RBF
       .get_coeff(...) of a deterministic seed function or a fixed-seed GP
       draw) and ansatz.dict_to_arr([...]) it into a flat array.

    7. max_recon_error / gamma_tol: leave as TODO / a loose placeholder until
       calibrated against an actual solve (values to be picked later, per
       discussion).

    Returns
    -------
    IntegrationCase
    """
    x = jnp.linspace(0., 1., 100)
    data_z = jnp.linspace(0., 2., 5)

    def beta(z):
        return z + 0.1

    def gamma(z):
        return 0.1 + 0.2 * z

    pod_local = jnp.exp(-0.5 * jnp.power((x - 0.4) / 0.2, 2.))
    RBF_beta = RBF_interpol(1e-6, 1., data_z)
    RBF_gamma = RBF_interpol(1e-6, 1., data_z)
    ansatz_exact = Ansatz([TransportedMode(x, RBF_beta, RBF_gamma)])

    data = DataCreation(x, ansatz_exact, [{
        'pod_local': lambda z: pod_local,
        'beta_value': beta,
        'gamma_value': gamma,
    }], data_z, {'beta_coeff': RBF_beta, 'gamma_coeff': RBF_gamma})

    ansatz = Ansatz([TransportedMode(x, RBF_beta, RBF_gamma)])
    weights_loss = LossWeights(1., 1., 0., 0., True, 1.)
    w2_config = W2Config(1., 1., True)
    phase_grad_config = OptimPhaseGradConfig(ansatz, [{
        'pod_local': {'l2': True, 'w1': True},
        'beta_coeff': {'l2': True, 'w1': True},
        'gamma_coeff': {'l2': True, 'w1': True},
    }])
    phase_coeff_config = OptimPhaseCoeffConfig(
        ansatz, [{'beta_coeff': 0., 'gamma_coeff': 0.}]
    )
    ansatz_cfg = AnsatzConfig(data, weights_loss, w2_config, ansatz,
                              phase_grad_config, phase_coeff_config)
    optim_cfg = OptimCfg(True, True, True, 50, 'SLSQP')
    solver = IteratedSolver(data, [ansatz_cfg], [optim_cfg], n_iter=1)

    RBF_init = RBF_interpol(0.5, 1., data_z)

    key = jax.random.key(37)
    subkey_beta, subkey_gamma, key = jax.random.split(key, 3)
    beta_values_init = jax.random.multivariate_normal(
        subkey_beta, jnp.zeros(data_z.shape[0]), RBF_init._K + 1e-3 * jnp.eye(data_z.shape[0])
    )
    initial_beta_coeff = RBF_beta.get_coeff(beta_values_init - jnp.min(beta_values_init))
    initial_gamma_coeff = RBF_gamma.get_coeff(
        jax.random.multivariate_normal(subkey_gamma, jnp.zeros(data_z.shape[0]), RBF_init._K + 1e-3 * jnp.eye(data_z.shape[0]))
    )

    init_pod_local = jnp.exp(-0.6 * jnp.power((x - 0.5) / 1., 2.))

    x0 = ansatz.dict_to_arr([{
        'pod_local': init_pod_local,
        'beta_coeff': initial_beta_coeff,
        'gamma_coeff': initial_gamma_coeff,
    }])

    return IntegrationCase(
        data=data,
        ansatz=ansatz,
        solver=solver,
        initial=x0,
        max_recon_error=1.0,  # TODO: tighten once calibrated against an actual solve
        gamma_tol=None,       # TODO: set once calibrated (currently skips the gamma-recovery check)
    )
