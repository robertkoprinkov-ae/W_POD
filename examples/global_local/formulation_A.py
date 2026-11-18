from W_POD.solver import IteratedSolver, L2_W1_optimizer, OptimCfg
from W_POD.util import DataCreation, RBF_interpol
from W_POD.ansatz import Ansatz, AnsatzConfig, LossWeights, W2Config, OptimPhaseGradConfig, OptimPhaseCoeffConfig
from W_POD.modes import TransportedMode, GlobalMode
from W_POD.visualizer import Visualizer

import jax.numpy as jnp
import jax

import numpy

from mpi4py import MPI

import os
import time

casename = 'global_local_formulation_A'

os.makedirs('data/%s' % casename, exist_ok=True)
os.makedirs('img/%s' % casename, exist_ok=True)

jax.config.update('jax_enable_x64', True)

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

# define problem/generate data
def alpha(z):
    return z

def beta(z):
    return jnp.maximum(1.5 + 1.*jnp.sin(z), 0.)

def gamma(z):
    return 4.5*jnp.sin(z)

x = jnp.linspace(0., 10., 100)
global_mode = jnp.exp(-0.5*jnp.power((x - 2.0)/10., 2.))
local_mode = jnp.exp(-0.5*jnp.power((x-4.5)/0.5, 2.))

data_z = jnp.array([0.2, 0.4, 0.9, 1.3, 1.75, 2.0, 2.5])

lengths_scale = 1e-6
reg_coeff = 0.
RBF_alpha = RBF_interpol(lengths_scale, 1., data_z, bias=False)

RBF_beta = RBF_interpol(lengths_scale, 1., data_z, bias=False)
RBF_gamma = RBF_interpol(lengths_scale, 1., data_z, bias=False)

ansatz_exact = Ansatz([
    GlobalMode(x, RBF_alpha),
    TransportedMode(x, RBF_beta, RBF_gamma)
])
data = DataCreation(
    x,
    ansatz_exact,
    [
        {'pod_global': lambda z: global_mode, 'alpha_value': alpha},
        {'pod_local': lambda z: local_mode, 'beta_value': beta, 'gamma_value': gamma}
    ],
    data_z,
    {'alpha_coeff': RBF_alpha, 'beta_coeff': RBF_beta, 'gamma_coeff': RBF_gamma}
)

# add some noise (optional, here set to 0)
key_noise = jax.random.key(37)
for z_i in range(data_z.shape[0]):
    subkey_noise, key_noise = jax.random.split(key_noise)
    noisy_component = 0.0*jax.random.multivariate_normal(subkey_noise, jnp.zeros(x.shape[0]), jnp.eye(x.shape[0]))
    data.data_exact = data.data_exact.at[z_i].set(jnp.maximum(data.data_exact[z_i] + noisy_component, 1e-6))


ansatz = Ansatz([
    GlobalMode(x, RBF_alpha),
    TransportedMode(
        x, RBF_beta, RBF_gamma
    )
])



n_repeat = 4

n_trials = 10

init_gamma = jnp.zeros((n_trials, data.data_z.shape[0]), dtype='float64')

key = jax.random.key(43)


pod_global_init = jnp.ones(data.data_exact.shape[1])
pod_global_init /= jnp.linalg.norm(pod_global_init)

gamma_init_RBF = RBF_interpol(0.1, 1., data_z)
K_gamma_init = gamma_init_RBF._K
K_gamma_init = K_gamma_init + 1e-3*jnp.eye(K_gamma_init.shape[0])

for i in range(n_trials):
    subkey_gamma, key = jax.random.split(key)

    init_gamma = init_gamma.at[i, :].set(jax.random.multivariate_normal(subkey_gamma, jnp.zeros(data.data_z.shape[0]), K_gamma_init))

init_alpha = jnp.zeros((n_trials, data.data_z.shape[0]), dtype='float64')

alpha_init_RBF = RBF_interpol(2., 0.1, data_z)
K_alpha_init = alpha_init_RBF._K
K_alpha_init = K_alpha_init + 1e-3*jnp.eye(K_alpha_init.shape[0])
for i in range(n_trials):
    subkey_alpha, key = jax.random.split(key)

    init_alpha = init_alpha.at[i, :].set(jax.random.multivariate_normal(subkey_alpha, jnp.zeros(data.data_z.shape[0]), K_alpha_init))

init_alpha -= init_alpha.min()

print(init_alpha)

def bcast_jnp(arr):
    arr_numpy = numpy.array(arr, dtype='float64')

    MPI_comm.Bcast([arr_numpy, MPI.DOUBLE], root=0)

    return jnp.array(arr_numpy, dtype='float64')

init_gamma = bcast_jnp(init_gamma)
init_alpha = bcast_jnp(init_alpha)

for i in range(n_trials):

    if i % MPI_size != MPI_rank:
        continue

    print('[%d/%d] Starting trial %d' % (MPI_rank, MPI_size, i))

    ansatz_cfgs = [AnsatzConfig(
        data,
        LossWeights(1., 1., 0., 0., True, 1.),
        W2Config(
            1.,
            1.,
            True
        ),
        ansatz,

        OptimPhaseGradConfig(ansatz, [{
            'alpha_coeff': {'l2': True, 'w1': True},
            'pod_global': {'l2': True, 'w1': True},}, {
            'beta_coeff': {'l2': True, 'w1': True},
            'gamma_coeff': {'l2': True, 'w1': True},
            'pod_local': {'l2': True, 'w1': True}
        }]),
        OptimPhaseCoeffConfig(ansatz, [{
            'alpha_coeff': reg_coeff}, {
            'beta_coeff': reg_coeff,
            'gamma_coeff': reg_coeff
        }]),
        variable_scaling_method='hessian',
    )
    ]

    optim_cfg = [OptimCfg(
        True, True, True, 100, 'SLSQP', '[%d/%d] ' % (MPI_rank, MPI_size)
    )]

    alpha_coeff_init = RBF_alpha.get_coeff(init_alpha[i, :])

    x_mid = 0.5*(data.x.min() + data.x.max())
    pod_local_init = jnp.exp(-jnp.power((x-x_mid)/0.25, 2.))
    pod_local_init = jnp.maximum(1e-3, pod_local_init)
    scaling_local = jnp.linalg.norm(pod_local_init)
    pod_local_init /= scaling_local
    beta_coeff_init = RBF_beta.get_coeff(data.data_exact.max() / pod_local_init.max() * jnp.ones(data.data_z.shape[0]))
    gamma_coeff_init = RBF_gamma.get_coeff(init_gamma[i, :])

    init_dict = [{
        'alpha_coeff': alpha_coeff_init,
        'pod_global': pod_global_init}, {
        'beta_coeff': beta_coeff_init,
        'gamma_coeff': gamma_coeff_init,
        'pod_local': pod_local_init
    }]

    ansatz_cfgs[0].update_scaling(ansatz.dict_to_arr(init_dict))
    solver = IteratedSolver(data, ansatz_cfgs, optim_cfg, n_repeat)


    print(init_dict)
    vis = Visualizer(data, solver)
    vis.plot_training(ansatz.dict_to_arr(init_dict), filename='img/%s/init_%d.png' % (casename, i))
    vis.plot_pod_and_coeff(ansatz.dict_to_arr(init_dict), filename='img/%s/pod_init_%d.png' % (casename, i))

    time_start = time.time()
    solver.solve_scipy(ansatz.dict_to_arr(init_dict))
    time_end = time.time()
    print('[%d/%d] Done; time taken: ' % (MPI_rank, MPI_size), time_end-time_start)

    vis.plot_training(solver.params_result, samples=True, filename='img/%s/training_%d.png' % (casename, i))
    test_z = jnp.array([1.15, 1.5, 2.25])
    test_data = jnp.zeros((test_z.shape[0], x.shape[0]))

    for z_i, z in enumerate(test_z):
        dict_exact = [{
            'pod_global': global_mode,
            'alpha_value': alpha(z)
        }, {
            'pod_local': local_mode,
            'beta_value': beta(z),
            'gamma_value': gamma(z)
        }]
        test_data = test_data.at[z_i, :].set(ansatz.get_values_grid_exact(dict_exact))

    vis.plot_test(solver.params_result, test_z, test_data, samples=True, filename='img/%s/test_%d.png' % (casename, i))
    vis.plot_pod_and_coeff(solver.params_result, filename='img/%s/pod_%d.png' % (casename, i))
    vis.plot_hist(filename='img/%s/hist_%d.png' % (casename, i))
    solver.write_hdf5_every('data/%s/params_%d/' % (casename, i))
    print('[%d/%d] Done %d' % (MPI_rank, MPI_size, i))


