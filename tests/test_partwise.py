from W_POD.solver import IteratedSolver
from W_POD.util import DataCreation, StretchedMode, RBF_interpol
from W_POD.visualizer import Visualizer

import jax.numpy as jnp
import jax

import numpy

from mpi4py import MPI

import scipy

jax.config.update('jax_enable_x64', True)

def alpha(z):
    return z

def beta(z):
    return jnp.maximum(1.5 + 1.*jnp.sin(z), 0.)

def gamma(z):
    return jnp.concatenate((3.*jnp.sin(z).reshape((1, -1)),
                            jnp.array(z).reshape((1, -1))), axis=0)

x = jnp.linspace(0., 10., 100)
global_mode = jnp.exp(-0.5*jnp.power((x - 2.0)/10., 2.))
local_mode = jnp.exp(-0.5*jnp.power((x-4.5)/0.5, 2.))

data_z = jnp.array([0.2, 0.4, 0.9, 1.3, 1.75, 2.0, 2.5])

RBF_exact = RBF_interpol(0.02, data_z)
data = DataCreation(x, global_mode, local_mode, StretchedMode(x), alpha, beta, gamma, data_z, RBF_exact)

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
        'beta': True,
        'gamma': True,
        'phi': False,
        'psi': True
        },
    'maxiter': 50,
    'stop_if_linesearch_fails': True,
    'l2_weight': 0.,
    'w1_weight': 1.,
    'reg_alpha_coeff': 0.,
    'reg_beta_coeff': 0.,
    'reg_gamma_coeff': 1e-3,
    'verbose': False,
    'log_w1': False,
    'local_mode_type': StretchedMode,
    'theta_alpha': 0.02,
    'theta_beta': 0.02,
    'theta_gamma': 0.02,
    }, 
]

n_repeat = 1
solver = IteratedSolver(data, solvers_cfg, n_repeat)

n_trials = 10

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

init_gamma = jnp.zeros((n_trials, 2*data.data_z.shape[0]), dtype='float64')

if MPI_rank == 0:
    key = jax.random.key(43)

    for i in range(n_trials):
        subkey_gamma_0, subkey_gamma_1, key = jax.random.split(key, 3)
        
        init_gamma = init_gamma.at[i, :data.data_z.shape[0]].set(jax.random.uniform(subkey_gamma_0, data.data_z.shape[0], minval=-3., maxval=3.))
        init_gamma = init_gamma.at[i, data.data_z.shape[0]:].set(jnp.exp(jax.random.uniform(subkey_gamma_1, data.data_z.shape[0], minval=jnp.log(1./3.), maxval=jnp.log(3.))))

gamma_numpy = numpy.array(init_gamma, dtype='float64')

MPI_comm.Bcast([gamma_numpy, MPI.DOUBLE], root=0)

init_gamma = jnp.array(gamma_numpy, dtype='float64')

for i in range(n_trials):

    if i % MPI_size != MPI_rank:
        continue

    print('[%d/%d] Starting trial %d' % (MPI_rank, MPI_size, i))
    
    RBF_init = RBF_interpol(0.02, data_z)
    
    pod_local_init = jnp.exp(-0.5*jnp.power((data.x-(data.x.max() + data.x.min())/2)/2., 2.))

    beta_coeff_init = data._beta_coeff_exact
    gamma_coeff_init = init_gamma[i, :].reshape((2, -1))
    
    # warmstart
    key = jax.random.key(65)
    subkey_alpha, subkey_init, key = jax.random.split(key, 3)
    init = jax.random.normal(subkey_init, data.x.shape[0]+data.data_z.shape[0])
    init = jnp.maximum(init, 1e-6)

    init = init.at[:data.data_z.shape[0]].set(jax.random.uniform(subkey_alpha, shape=data.data_z.shape[0]))
    weights = [0., 10., 15., 100.]
    for weight in weights:
        def cost_sp(x):
            alpha = x[:data.data_z.shape[0]].reshape((-1, 1))
            pod = x[data.data_z.shape[0]:].reshape((1, -1))

            data_rec = alpha @ pod
            residual = data_rec - data.data_exact

            errors_neg = jnp.where(residual > 0., residual, 0.)

            return jnp.sum(jnp.power(residual, 2.)) + weight * jnp.sum(jnp.power(errors_neg, 2.))
        bounds = jnp.concatenate((jnp.zeros((x.shape[0]+data.data_z.shape[0], 1)), 100.*jnp.ones((x.shape[0]+data.data_z.shape[0], 1))), axis=1)
        grad_cost = jax.grad(cost_sp)

        result = scipy.optimize.minimize(cost_sp, init, jac=grad_cost, method='L-BFGS-B', bounds=bounds)
        init = result.x
    pod_global_init = init[data.data_z.shape[0]:]
    alpha_values_init = init[:data.data_z.shape[0]]
    alpha_coeff_init = RBF_init.get_coeff(alpha_values_init)
    
    for z_i in range(data.data_z.shape[0]):
        data.data_exact = data.data_exact.at[z_i, :].set(data.data_exact[z_i, :] - alpha_values_init[z_i] * pod_global_init)
    
    data.data_exact = data.data_exact - data.data_exact.min() + 1e-6
    alpha_coeff_init = jnp.zeros_like(alpha_coeff_init)

    vis = Visualizer(data, solver)
    
    if i==0:
        vis.plot_training(pod_global=data.global_mode, pod_local=data.local_mode, alpha_coeff=data._alpha_coeff_exact, beta_coeff=data._beta_coeff_exact, gamma_coeff=data._gamma_coeff_exact, filename='img/partwise/exact.png')
    vis.plot_training(pod_global=pod_global_init, pod_local=pod_local_init, alpha_coeff=alpha_coeff_init, beta_coeff=beta_coeff_init, gamma_coeff=gamma_coeff_init, filename='img/partwise/init_%d.png' % i)
    
    solver.solve((pod_global_init, pod_local_init, alpha_coeff_init, beta_coeff_init, gamma_coeff_init))
    print('[%d/%d] Done %i' % (MPI_rank, MPI_size, i)) 
    vis.plot_OT_training(filename='img/partwise/OT_training_%d.png' % i)

    vis.plot_training(filename='img/partwise/training_%d.png' % i)
    vis.plot_pod_and_gamma(filename='img/partwise/pod_%d.png' % i)
    vis.plot_hist(filename='img/partwise/hist_%d.png' % i)

    vis.plot_every(directory='img/partwise/every_%d/' % i)
