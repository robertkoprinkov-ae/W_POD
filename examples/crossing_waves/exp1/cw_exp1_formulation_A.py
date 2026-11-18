from copy import copy

from W_POD.solver import IteratedSolver, OptimCfg
from W_POD.util import DataCreation, RBF_interpol
from W_POD.ansatz import Ansatz, AnsatzConfig, LossWeights, W2Config, OptimPhaseGradConfig, OptimPhaseCoeffConfig
from W_POD.modes import TransportedMode
from W_POD.visualizer import Visualizer

import jax.numpy as jnp
import jax
import os
import time

import numpy as np
import W_POD.hess_diag_probed as hdp

# BATCHED PROBE
# Batched Hessian probe. Script-level patch of W_POD.hess_diag_probed._probe_diag (frozen src untouched):
# the HVPs go through jax.lax.map(..., batch_size=8) (the line commented out in src) instead of one call per colour.
def _probe_diag_batched(hvp, n, idx, colours, batch_size=8):
    out = np.zeros(n)
    ncol = int(colours.max()) + 1
    V = np.zeros((ncol, n))
    for c in range(ncol):
        V[c, idx[colours == c]] = 1.0
    print('Starting probe (batched: %d HVPs, batch_size %d)' % (ncol, batch_size))
    t0 = time.time()
    bs = batch_size
    while True:  # halve the batch on GPU OOM
        try:
            Hv = np.asarray(jax.lax.map(hvp, jnp.array(V), batch_size=bs))
            break
        except jax.errors.JaxRuntimeError as e:
            if 'RESOURCE_EXHAUSTED' not in str(e) or bs == 1:
                raise
            print('Probe OOM at batch_size %d, retrying with %d' % (bs, bs // 2))
            bs //= 2
    print('Ending probe (%.1f s, batch_size %d)' % (time.time() - t0, bs))
    for c in range(ncol):
        sel = idx[colours == c]
        out[sel] = Hv[c][sel]
    return out, ncol

hdp._probe_diag = _probe_diag_batched

casename = 'cw1_formulation_A'  

os.makedirs('img/%s' % casename, exist_ok=True)

x = jnp.linspace(0., 1., 400, endpoint=False)
data_z = jnp.linspace(-10, 10., 200)

scaling_x = 400

def u(z):
    mu = 200
    sigma = 4.

    delta_1 = 0.15*jnp.power(z, 3.) + 0.8 *z + 1.5
    delta_2 = -18*z + 2
    return jnp.abs(jnp.sin(jnp.pi * z / 10.)) * jnp.exp(-jnp.power((scaling_x*x-mu-delta_1)/sigma, 2.)) +\
           jnp.abs(jnp.cos(jnp.pi * z / 10.)) * jnp.exp(-jnp.power((scaling_x*x-mu-delta_2)/sigma, 2.))

data_exact = jnp.zeros((data_z.shape[0], x.shape[0]))

for z_i, z in enumerate(data_z):
    data_exact = data_exact.at[z_i, :].set(jnp.maximum(u(z), 1e-6))
data_exact = data_exact / data_exact.max()

class DataHolder():

    def __init__(self, x, data_z, data_exact):
        self.x = x

        self.dx = self.x[1] - self.x[0]

        self.data_z = data_z
        self.data_exact = data_exact

data = DataHolder(x, data_z, data_exact)

RBF_beta = RBF_interpol(1e-6, 1., data_z, kernel='matern32', bias=False)
RBF_gamma = RBF_interpol(1e-6, 1., data_z, kernel='SE', bias=False)

ansatz = Ansatz([
    TransportedMode(
        x, RBF_beta, RBF_gamma#, gamma_increasing=True
    ),
    TransportedMode(
        x, RBF_beta, RBF_gamma#, gamma_decreasing=True
    )
])



n_repeat = 1

n_trials = 10

init_gamma = jnp.zeros((n_trials, data_z.shape[0]), dtype='float64')

key = jax.random.key(43)

ansatz_cfgs = [AnsatzConfig(
    data,
    LossWeights(1., 1., 0., 0., True, 1.),
    W2Config(1., 1., True),
    ansatz,
    OptimPhaseGradConfig(ansatz,
    [{
        'beta_coeff': {'l2': True, 'w1': True},
        'gamma_coeff': {'l2': True, 'w1': True},
        'pod_local': {'l2': True, 'w1': True}},
        {
            'beta_coeff': {'l2': True, 'w1': True},
            'gamma_coeff': {'l2': True, 'w1': True},
            'pod_local': {'l2': True, 'w1': True}}]),
    OptimPhaseCoeffConfig(ansatz, [{
        'beta_coeff': 0.,
        'gamma_coeff': 0.}, {
        'beta_coeff': 0.,
        'gamma_coeff': 0.
    }]),
    variable_scaling_method='hessian',
)]
ansatz_cfgs = [ansatz_cfgs[0], ansatz_cfgs[0], ansatz_cfgs[0]]


optim_cfg = [
        OptimCfg(True, True, True, 1000, 'SLSQP'),
        OptimCfg(True, True, True, 1000, 'SLSQP'),
        OptimCfg(True, True, True, 1000, 'SLSQP'),
]

solver = IteratedSolver(data, ansatz_cfgs, optim_cfg, n_repeat)
lb, ub = ansatz.get_bounds_arr()

vis = Visualizer(data, solver)

def pod_init(i):
    k_mu, k_w = jax.random.split(jax.random.fold_in(jax.random.key(1044), i))
    mu = jnp.array([0.5, 0.5])
    u01 = jax.random.uniform(k_w, (2,), minval=0., maxval=1.)
    w = 0.01 * 5. ** (2. * u01 - 1.)
    w = jnp.array([w[0], w[0]])
    pods = [jnp.maximum(1e-3, jnp.exp(-jnp.power((data.x - mu[m]) / w[m], 2.))) for m in range(2)]
    return mu, w, pods

if __name__ == '__main__':
    for i in range(n_trials):
        init_gamma_1 = -(1 + data_z + 0.01 * jnp.power(data_z, 3.)) / 399.
        init_gamma_2 = -(1 - data_z) / 399.

        pod_mu, pod_w, pod_local_inits = pod_init(i)
        print('pod_init', i, 'mu', pod_mu, 'w', pod_w)
        
        init_dict = [{
            'beta_coeff': RBF_beta.get_coeff(jnp.ones(data.data_z.shape[0])),
            'gamma_coeff': RBF_gamma.get_coeff(init_gamma_1),
            'pod_local': pod_local_inits[0]}, {
            'beta_coeff': RBF_beta.get_coeff(jnp.ones(data.data_z.shape[0])),
            'gamma_coeff': RBF_gamma.get_coeff(init_gamma_2),
            'pod_local': pod_local_inits[1]
        }
        ]

        init_arr = ansatz.dict_to_arr(init_dict)
        
        print('start', i)
        time_start = time.time()

        try:
            solver.solve_scipy(ansatz.dict_to_arr(init_dict), directory_out='data/%s/every_%d/' % (casename, i))
        except Exception as e:
            print('TRIAL CRASHED', i, repr(e)[:300], flush=True)
            continue
        time_end = time.time()
        print('Done', i, '; seconds taken: ', (time_end-time_start))

        vis.plot_training(solver.params_result, filename='img/%s/training_%d.png' % (casename, i))
        vis.plot_pod_and_coeff(solver.params_result, filename='img/%s/pod_%d.png' % (casename, i))
        vis.plot_hist(filename='img/%s/hist_%d.png' % (casename, i))
