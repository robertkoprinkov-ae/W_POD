from W_POD.solver import IteratedSolver
from W_POD.util import DataCreation, StretchedMode, RBF_interpol
from W_POD.visualizer import Visualizer

import jax.numpy as jnp
import jax

import numpy

from mpi4py import MPI

jax.config.update('jax_enable_x64', True)

def alpha(z):
    return jnp.zeros_like(z)

def beta(z):
    return jnp.ones_like(z)

def gamma(z):
    gamma_vals = jnp.concatenate((3*jnp.cos(z).reshape((1, -1)),
                                 jnp.array(z).reshape((1, -1))), axis=0)
    return gamma_vals

x = jnp.linspace(0., 10., 100)
global_mode = jnp.ones_like(x)

# does the center here matter?
local_mode = jnp.exp(-0.5*jnp.power((x-5.)/1., 2.))

data_z = jnp.array([0.2, 0.5, 1., 1.8, 2.5])

RBF = RBF_interpol(0.1, data_z)
data = DataCreation(x, global_mode, local_mode, StretchedMode(x), alpha, beta, gamma, data_z, RBF)

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
    'maxiter': 20,
    'stop_if_linesearch_fails': True,
    'l2_weight': 0.,
    'w1_weight': 1.,
    'reg_alpha_coeff': 0.,
    'reg_beta_coeff': 0.,
    'reg_gamma_coeff': 1e-3,
    'verbose': False,
    'log_w1': False,
    'local_mode_type': StretchedMode,
    'theta_alpha': 0.1,
    'theta_beta': 0.1,
    'theta_gamma': 0.1,
    'scipy_optim_method': 'L-BFGS-B'
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
    'maxiter': 50,
    'stop_if_linesearch_fails': True,
    'l2_weight': 1.,
    'w1_weight': 0.1,
    'reg_alpha_coeff': 1e-3,
    'reg_beta_coeff': 0.,
    'reg_gamma_coeff': 1e-3,
    'verbose': False,
    'log_w1': False,
    'local_mode_type': StretchedMode,
    'theta_alpha': 0.1,
    'theta_beta': 0.1,
    'theta_gamma': 0.1,
    'scipy_optim_method': 'L-BFGS-B'
    }
]

n_repeat = 15
solver = IteratedSolver(data, solvers_cfg, n_repeat)

n_trials = 10

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

init_gamma = jnp.zeros((n_trials, 2*data.data_z.shape[0]), dtype='float64')

if MPI_rank == 0:
    key = jax.random.key(43)

    for i in range(n_trials):
        subkey_gamma, key = jax.random.split(key)

        init_gamma = init_gamma.at[i, :].set(jnp.exp(jax.random.uniform(subkey_gamma, 2*data.data_z.shape[0], minval=jnp.log(1/3.), maxval=jnp.log(3.))))
        init_gamma = init_gamma.at[i, :data.data_z.shape[0]].set(jnp.zeros(data.data_z.shape[0]))

gamma_numpy = numpy.array(init_gamma, dtype='float64')

MPI_comm.Bcast([gamma_numpy, MPI.DOUBLE], root=0)

init_gamma = jnp.array(gamma_numpy, dtype='float64')

for i in range(n_trials):

    if i % MPI_size != MPI_rank:
        continue

    print('[%d/%d] Starting trial %d' % (MPI_rank, MPI_size, i))
    
    pod_global_init = jnp.mean(data.data_exact, axis=0)
    pod_global_init /= jnp.linalg.norm(pod_global_init)
    pod_global_init = 1e-9*jnp.ones_like(data.x)
    pod_local_init = jnp.exp(-0.5*jnp.power((data.x-(data.x.max() + data.x.min())/2)/2., 2.))

    alpha_values_init = data.data_exact @ pod_global_init
    alpha_coeff_init = jnp.zeros_like(data.data_z)
    beta_coeff_init = data._beta_coeff_exact
    gamma_coeff_init = init_gamma[i, :].reshape((2, -1))
    
    alpha_coeff_init = RBF.get_coeff(alpha_values_init)

    vis = Visualizer(data, solver)
    
    if i==0:
        vis.plot_training(pod_global=data.global_mode, pod_local=data.local_mode, alpha_coeff=data._alpha_coeff_exact, beta_coeff=data._beta_coeff_exact, gamma_coeff=data._gamma_coeff_exact, filename='img/test_beta_fixed_stretched_mode/exact.png')
    vis.plot_training(pod_global=pod_global_init, pod_local=pod_local_init, alpha_coeff=alpha_coeff_init, beta_coeff=beta_coeff_init, gamma_coeff=gamma_coeff_init, filename='img/test_beta_fixed_stretched_mode/init_%d.png' % i)
    
    solver.solve_scipy((pod_global_init, pod_local_init, alpha_coeff_init, beta_coeff_init, gamma_coeff_init))

    vis.plot_training(filename='img/test_beta_fixed_stretched_mode/training_%d.png' % i)
    vis.plot_pod_and_gamma(filename='img/test_beta_fixed_stretched_mode/pod_%d.png' % i)
    vis.plot_hist(filename='img/test_beta_fixed_stretched_mode/hist_%d.png' % i)
    
    vis.plot_every(directory='img/test_beta_fixed_stretched_mode/every_%d/' % i)

