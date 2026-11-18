# this file is partially claude written, adapted from the main formulation B file
from copy import copy

from W_POD.solver import IteratedSolver, OptimCfg, L2_W1_optimizer
from W_POD.util import DataCreation, RBF_interpol
from W_POD.ansatz import Ansatz, AnsatzConfig, LossWeights, W2Config, OptimPhaseGradConfig, OptimPhaseCoeffConfig
from W_POD.modes import TransportedMode
from W_POD.visualizer import Visualizer

import jax.numpy as jnp
import jax
import os
import time
import glob
import re
import h5py

casename = 'cw1_formulation_B_budget_postproc'
input_casename = 'cw1_formulation_B_budget'  

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


# the kernels of the reg run, only used to read the reg params
RBF_beta_reg = RBF_interpol(0.2, 1., data_z, kernel='matern32', bias=True)
RBF_gamma_reg = RBF_interpol(3.0, 0.3, data_z, kernel='SE', bias=True)  # THG3 (2026-11-03), was 2.0
ansatz_reg = Ansatz([TransportedMode(x, RBF_beta_reg, RBF_gamma_reg), TransportedMode(x, RBF_beta_reg, RBF_gamma_reg)])

n_trials = 10

# (beta length, gamma length, maxiter, beta reg, gamma reg) per stage; length 1e-6 + reg 0 = noreg_fp32.py
phases = [(0.025, 0.375, 50, 1e-4, 1e-5), (1e-6, 1e-6, 50, 0., 0.)]

def make_phase(l_beta, l_gamma, maxiter, c_beta, c_gamma):
    rbf_beta = RBF_interpol(l_beta, 1., data_z, kernel='matern32', bias=True)
    rbf_gamma = RBF_interpol(l_gamma, 0.3 if c_gamma > 0. else 1., data_z, kernel='SE', bias=True)
    an = Ansatz([TransportedMode(x, rbf_beta, rbf_gamma), TransportedMode(x, rbf_beta, rbf_gamma)])
    cfg = AnsatzConfig(data, LossWeights(1., 1., 0., 0., True, 1.), W2Config(1., 1., True), an,
        OptimPhaseGradConfig(an, [{n: {'l2': True, 'w1': True} for n in ['beta_coeff', 'gamma_coeff', 'pod_local']}] * 2),
        OptimPhaseCoeffConfig(an, [{'beta_coeff': c_beta, 'gamma_coeff': c_gamma}] * 2), variable_scaling_method='hessian')
    return rbf_beta, rbf_gamma, an, L2_W1_optimizer(data, cfg, OptimCfg(True, True, True, maxiter, 'SLSQP'))

stages = [make_phase(*p) for p in phases]
reg_stage = (RBF_beta_reg, RBF_gamma_reg, ansatz_reg)
ansatz = stages[-1][2]
stage_ansatz = [stages[0][2]] + [s[2] for s in stages]  # parametrization of params_0 .. params_3

def convert(params_arr, old, new, nugget):
    # beta, gamma (before the softplus) at data_z, refitted with the next stage's kernels; pod_local copied
    out = []
    for d in old[2].arr_to_dict(params_arr):
        b = jax.vmap(lambda z: old[0].predict(z, d['beta_coeff']))(data_z)
        g = jax.vmap(lambda z: old[1].predict(z, d['gamma_coeff']))(data_z)
        out.append({'beta_coeff': new[0].get_coeff(b, nugget=nugget), 'gamma_coeff': new[1].get_coeff(g, nugget=nugget), 'pod_local': d['pod_local']})
    return new[2].dict_to_arr(out)

def rel_error(ansatz_, params_arr):
    rec = jax.vmap(lambda z: ansatz_.get_values_grid(z, params_arr))(data_z)
    return float(jnp.linalg.norm(rec - data_exact) / jnp.linalg.norm(data_exact))

# Visualizer needs an IteratedSolver (L2_W1_optimizer.calc_gradients_per_param is broken in this W_POD)
vis = Visualizer(data, IteratedSolver(data, [stages[-1][3].ansatz_cfg], [stages[-1][3].optim_cfg], 1))

if __name__ == '__main__':
    for i in range(n_trials):
        # read end result of formulation B run
        reg_dir = 'data/%s/every_%d' % (input_casename, i)
        stage = max(int(re.search(r'params_(\d+)\.hdf5$', f).group(1)) for f in glob.glob(reg_dir + '/params_*.hdf5'))

        # convert parameters to length scale used in the postprocessing & run postprocessing
        reg_filename = '%s/params_%d.hdf5' % (reg_dir, stage)
        with h5py.File(reg_filename, 'r') as f:
            reg_arr = jnp.array(f['params_arr'])
        params = convert(reg_arr, reg_stage, stages[0], 1e-5)
        print('init from', reg_filename, '; rel. error reg params: %.6e, converted: %.6e'
              % (rel_error(ansatz_reg, reg_arr), rel_error(stages[0][2], params)), flush=True)

        print('start', i)
        time_start = time.time()
        os.makedirs('data/%s/every_%d' % (casename, i), exist_ok=True)
        with h5py.File('data/%s/every_%d/params_0.hdf5' % (casename, i), 'w') as f:  # solver.write_hdf5 needs a finished solve
            f['params_arr'] = jax.device_get(params)
        for k, stage in enumerate(stages):
            if k > 0:
                params = convert(params, stages[k-1], stage, 0. if phases[k][0] < 1e-3 else 1e-5)
            print('Phase %d' % k, phases[k], '; rel. error at start: %.6e' % rel_error(stage[2], params), flush=True)
            stage[3].solve_scipy(params)
            params = stage[3].params_result
            stage[3].write_hdf5(params, filename='data/%s/every_%d/params_%d.hdf5' % (casename, i, k+1))
        time_end = time.time()
        print('Done', i, '; seconds taken: ', (time_end-time_start))

        vis._solver._log_l2, vis._solver._log_w1, vis._solver._log_tot = stage[3]._log_l2, stage[3]._log_w1, stage[3]._log_tot
        vis.plot_training(params, filename='img/%s/training_%d.png' % (casename, i))
        vis.plot_pod_and_coeff(params, filename='img/%s/pod_%d.png' % (casename, i))
        vis.plot_hist(filename='img/%s/hist_%d.png' % (casename, i))
