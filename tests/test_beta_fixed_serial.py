from W_POD.solver import IteratedSolver
from W_POD.util import DataCreation, TransportedMode, RBF_interpol
from W_POD.visualizer import Visualizer

import jax.numpy as jnp

def alpha(z):
    return jnp.zeros_like(z)

def beta(z):
    return jnp.ones_like(z)

def gamma(z):
    return jnp.cos(z).reshape((1, -1))

x = jnp.linspace(0., 10., 100)
global_mode = jnp.ones_like(x)
local_mode = jnp.exp(-0.5*jnp.power((x-4)/1., 2.))

data_z = jnp.array([0.5, 1., 1.8])

RBF = RBF_interpol(0.1, data_z)
data = DataCreation(x, global_mode, local_mode, TransportedMode(x), alpha, beta, gamma, data_z, RBF)

solvers_cfg = [
    {
    'l2': {
        'alpha': False,
        'beta': False,
        'gamma': False,
        'phi': False,
        'psi': False
        },
    'w1': {
        'alpha': False,
        'beta': False,
        'gamma': True,
        'phi': False,
        'psi': True
        },
    'maxiter': 5,
    'stop_if_linesearch_fails': True,
    'l2_weight': 0.,
    'w1_weight': 1.,
    'reg_alpha_coeff': 0.,
    'reg_beta_coeff': 0.,
    'reg_gamma_coeff': 1e-3,
    'verbose': False,
    'log_w1': False,
    'local_mode_type': TransportedMode,
    'theta_alpha': 0.1,
    'theta_beta': 0.1,
    'theta_gamma': 0.1
    }, 
    {
    'l2': {
        'alpha': True,
        'beta': False,
        'gamma': True,
        'phi': True,
        'psi': True
        },
    'w1': {
        'alpha': True,
        'beta': False,
        'gamma': True,
        'phi': True,
        'psi': True
    },
    'maxiter': 5,
    'stop_if_linesearch_fails': True,
    'l2_weight': 1.,
    'w1_weight': 0.1,
    'reg_alpha_coeff': 1e-3,
    'reg_beta_coeff': 1e-3,
    'reg_gamma_coeff': 1e-3,
    'verbose': False,
    'log_w1': False,
    'local_mode_type': TransportedMode,
    'theta_alpha': 0.1,
    'theta_beta': 0.1,
    'theta_gamma': 0.1
    }
]

n_repeat = 1
solver = IteratedSolver(data, solvers_cfg, n_repeat)

print(data._gamma_coeff_exact, data._gamma_values)
print(data._alpha_coeff_exact)
print(data._beta_coeff_exact)

alpha_init = jnp.ones_like(data._alpha_coeff_exact)
gamma_init = jnp.ones_like(data._gamma_coeff_exact)

solver.solve((global_mode, local_mode, alpha_init, data._beta_coeff_exact, gamma_init))

print(solver.params_result[2])
print('Done')
vis = Visualizer(data, solver)
print(data._alpha_coeff_exact)

vis.plot_training(pod_global=global_mode, pod_local=local_mode, alpha_coeff=alpha_init, beta_coeff=data._beta_coeff_exact, gamma_coeff=gamma_init, filename='img/test_serial/init.png')
print(data._alpha_coeff_exact)
vis.plot_pod_and_gamma(alpha_coeff=alpha_init, beta_coeff=data._beta_coeff_exact, gamma_coeff=gamma_init, filename='img/test_serial/pod_init.png')
print(data._alpha_coeff_exact)


vis.plot_training(filename='img/test_serial/training.png')
vis.plot_pod_and_gamma(filename='img/test_serial/pod.png')
vis.plot_hist(filename='img/test_serial/hist.png')

print(data._alpha_coeff_exact)

vis.plot_training(alpha_coeff=data._alpha_coeff_exact, beta_coeff=data._beta_coeff_exact, gamma_coeff=data._gamma_coeff_exact, filename='img/test_serial/better.png')

vis.plot_every(directory='img/test_serial/every_0/')

solver.write_hdf5_every('../data/output/test_serial/')
