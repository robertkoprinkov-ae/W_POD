from copy import copy


from W_POD.solver import IteratedSolver, OptimCfg
from W_POD.util import DataCreation, RBF_interpol
from W_POD.ansatz import Ansatz, AnsatzConfig, LossWeights, W2Config, OptimPhaseGradConfig, OptimPhaseCoeffConfig
from W_POD.modes import TransportedMode
from W_POD.visualizer import Visualizer

import jax.numpy as jnp
import jax
import os

jax.config.update('jax_enable_x64', True)

casename = 'cw2_formulation_A'

os.makedirs('img/%s' % casename, exist_ok=True)

x = jnp.linspace(0., 1., 400)
data_z = jnp.linspace(-10, 10., 10)

scaling_x = 400

def u(z):
    mu = 200
    sigma = 40

    delta_1 = 0.15*jnp.power(z, 3.) + 0.8 *z + 1.5
    delta_2 = -18*z + 2
    return jnp.abs(jnp.sin(jnp.pi * z / 10.)) * jnp.exp(-jnp.power((scaling_x*x-mu-delta_1)/sigma, 2.)) +\
           jnp.abs(jnp.cos(jnp.pi * z / 10.)) * jnp.exp(-jnp.power((scaling_x*x-mu-delta_2)/sigma, 2.))

data_exact = jnp.zeros((data_z.shape[0], x.shape[0]))

for z_i, z in enumerate(data_z):
    data_exact = data_exact.at[z_i, :].set(u(z))
data_exact = data_exact / data_exact.max()
data_exact = jnp.maximum(data_exact, 1e-6)

class DataHolder():

    def __init__(self, x, data_z, data_exact):
        self.x = x

        self.dx = self.x[1] - self.x[0]

        self.data_z = data_z
        self.data_exact = data_exact

data = DataHolder(x, data_z, data_exact)

RBF_beta = RBF_interpol(1e-6, 1., data_z)
RBF_gamma = RBF_interpol(1e-6, 1., data_z)

ansatz = Ansatz([
    TransportedMode(
        x, RBF_beta, RBF_gamma
    ),
    TransportedMode(
        x, RBF_beta, RBF_gamma
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
    }])
    , variable_scaling_method='hessian'
)]
ansatz_cfgs = [ansatz_cfgs[0], ansatz_cfgs[0], ansatz_cfgs[0]]

optim_cfg = [OptimCfg(True, True, True, 20, 'SLSQP'),
    OptimCfg(True, True, True, 200, 'SLSQP'),
    OptimCfg(True, True, True, 200, 'SLSQP')
]

solver = IteratedSolver(data, ansatz_cfgs, optim_cfg, n_repeat)
lb, ub = ansatz.get_bounds_arr()

vis = Visualizer(data, solver)

if __name__ == '__main__':
    from mpi4py import MPI
    MPI_rank, MPI_size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    for c in optim_cfg: c.callback_prefix = '[%d/%d] ' % (MPI_rank, MPI_size)
    for i in range(n_trials):
        subkey_gamma_1, subkey_gamma_2, key = jax.random.split(key, 3)

        RBF_gamma_init = RBF_interpol(20., 0.6, data_z)
        gamma_K = RBF_gamma_init._K
        gamma_K = gamma_K + 1e-3 * jnp.eye(gamma_K.shape[0])
        init_gamma_1 = jax.random.multivariate_normal(subkey_gamma_1, jnp.zeros(data_z.shape[0]), gamma_K)
        init_gamma_2 = jax.random.multivariate_normal(subkey_gamma_2, jnp.zeros(data_z.shape[0]), gamma_K)
        if i % MPI_size != MPI_rank: 
            continue
        
        init_dict = [{
            'beta_coeff': RBF_beta.get_coeff(jnp.ones(data.data_z.shape[0])),
            'gamma_coeff': RBF_gamma.get_coeff(init_gamma_1),
            'pod_local': jnp.maximum(jnp.exp(-jnp.power(100*(data.x-0.5), 2.)), 1e-6)}, {
            'beta_coeff': RBF_beta.get_coeff(jnp.ones(data.data_z.shape[0])),
            'gamma_coeff': RBF_gamma.get_coeff(init_gamma_2),
            'pod_local': jnp.maximum(jnp.exp(-jnp.power(100*(data.x-0.5), 2.)), 1e-6)
        }
        ]

        init_arr = ansatz.dict_to_arr(init_dict)

        # the starting point must satisfy the bounds and beta >= 1e-3 at every z_j (TransportedMode.get_constraints);
        # the monotonic shift constraints are deliberately not enforced at the start
        viol_b = float(jnp.maximum(jnp.max(lb - init_arr), jnp.max(init_arr - ub)))
        viol_beta = max(float(jnp.max(1e-3 - RBF_beta.predict(data_z, d['beta_coeff']))) for d in init_dict)
        lin = ansatz.get_constraints()
        Ax = jnp.asarray(lin.A) @ init_arr
        viol_c = float(jnp.maximum(jnp.max(jnp.asarray(lin.lb) - Ax), jnp.max(Ax - jnp.asarray(lin.ub))))
        print('x0 max violation: bounds %.3e, beta >= 1e-3 %.3e, all linear constraints (incl. monotonic) %.3e' % (viol_b, viol_beta, viol_c))
        assert viol_b <= 0. and viol_beta <= 0., 'starting point violates the bounds or the beta constraint'
        ansatz_cfgs[0].update_scaling(init_arr)

        vis.plot_training(ansatz.dict_to_arr(init_dict), filename='img/%s/init_%d.png' % (casename, i))
        print('start')
        vis.plot_pod_and_coeff(ansatz.dict_to_arr(init_dict), filename='img/%s/pod_init_%d.png' % (casename, i))

        solver.solve_scipy(ansatz.dict_to_arr(init_dict))
        print('Done')

        vis.plot_training(solver.params_result, filename='img/%s/training_%d.png' % (casename, i))
        vis.plot_pod_and_coeff(solver.params_result, filename='img/%s/pod_%d.png' % (casename, i))
        vis.plot_hist(filename='img/%s/hist_%d.png' % (casename, i))

        solver.write_hdf5_every(directory='data/%s/every_%d/' % (casename, i))
