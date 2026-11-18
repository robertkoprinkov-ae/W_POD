from typing import List

import matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ott.geometry import pointcloud
from ott.solvers import linear
import ott

import jax

import scipy

import numpy

from W_POD.util import RBF_interpol

import time as timelib

import h5py
from src_wildfire.visu import Visualizer
from src_wildfire.losses import Losses, Losses_scaled
from src_wildfire.ansatz import CoordinateTransformation, IdentityCoordinateTransformation, Mode, CoordinateFrame, Ansatz

jax.config.update('jax_enable_x64', True)

start = timelib.time()

DOWNSCALED = True

time, mask, ux, uy = None, None, None, None

grid = jnp.load('Wildlandfire_2d/1D_Grid_full_burela.npy')
data_z = jnp.load('Wildlandfire_2d/data_z_full_burela.npy')
snapshots = jnp.load('Wildlandfire_2d/data_full_burela.npy')


snapshots = snapshots / snapshots.max()

time_idx = jnp.linspace(0., snapshots.shape[1]-1, 50).round().astype(int)
data_exact_2d = snapshots[:, time_idx, 0, :, :]
data_exact = data_exact_2d.reshape((data_exact_2d.shape[0] * data_exact_2d.shape[1],
                                   data_exact_2d.shape[2], data_exact_2d.shape[3]))

data_z_2d = data_z.reshape((snapshots.shape[0], snapshots.shape[1], 2))[:, time_idx, :]
data_z = data_z_2d.reshape((-1, 2))

print(data_z)
print(data_z.shape[0], snapshots.shape[1])

x_crds = grid[0]
y_crds = grid[1]

x_crds = jnp.linspace(0., 1., grid[0].shape[0])
y_crds = jnp.linspace(0., 1., grid[1].shape[0])

dx = x_crds[1] - x_crds[0]

H, W = x_crds.shape[0], y_crds.shape[0]


xx, yy = jnp.meshgrid(x_crds, y_crds)
pts = jnp.concatenate((xx.reshape((-1, 1)), yy.reshape((-1, 1))), axis=1)

print('defining rbf')

# define ansatz
rbf_radial = RBF_interpol(10., 1., data_z, nugget=1e-6, bias=True)
rbf_shift_x = RBF_interpol(1e-6, 1., data_z, nugget=1e-6, bias=False)
rbf_shift_y = RBF_interpol(1e-6, 1., data_z, nugget=1e-6, bias=False)

rbf_amplitude = RBF_interpol(1e-6, 1., data_z, nugget=1e-6, bias=False)
print('about to define ansatz')
ansatz = Ansatz([CoordinateFrame(pts, x_crds, y_crds, data_z, CoordinateTransformation(24, rbf_radial, rbf_shift_x, rbf_shift_y, fix_center=True),
                         [Mode(pts.shape[0], False, rbf_amplitude), 
                          Mode(pts.shape[0], False, rbf_amplitude), 
                          Mode(pts.shape[0], False, rbf_amplitude), 
                          Mode(pts.shape[0], False, rbf_amplitude)]),
                 CoordinateFrame(pts, x_crds, y_crds, data_z, IdentityCoordinateTransformation(), [
                     Mode(pts.shape[0], False, rbf_amplitude), 
                     Mode(pts.shape[0], False, rbf_amplitude)], fixed=True)], data_z)
print('ansatz defined')
vis = Visualizer(pts, x_crds, y_crds, data_z, data_exact, ansatz)
print('vis defined')

radial_coeff = rbf_radial.get_coeff(jnp.ones(data_z.shape[0]))

coordinate_params = {'radial_%d_coeff' % radial_station_i: jnp.copy(radial_coeff) for radial_station_i in range(ansatz.coordinate_frames[0].coordinate_transformation.n_discret_theta)} | {
    'shift_x_coeff': rbf_shift_x.get_coeff(jnp.zeros(data_z.shape[0])),
    'shift_y_coeff': rbf_shift_y.get_coeff(jnp.zeros(data_z.shape[0])),
}

theta_anisotropic = jnp.array([0.05, 0.05])
mode_init = jnp.exp(-jnp.sum(jnp.power((pts - jnp.array([0.5, 0.5])) / theta_anisotropic, 2.), axis=1))

rbf_amplitude_init = RBF_interpol(200, 0.2, data_z, nugget=1e-6, bias=False)
K_amplitude_init = rbf_amplitude_init._K
K_amplitude_init += 1e-6*jnp.eye(K_amplitude_init.shape[0])

ampls = jnp.zeros((6, data_z.shape[0]))

import sys
seed = int(sys.argv[1])
print("seed", seed, flush=True)
key = jax.random.key(seed)
for mode_i in range(ampls.shape[0]):
    subkey, key = jax.random.split(key)
    draw = jax.random.multivariate_normal(subkey, jnp.zeros(data_z.shape[0]), K_amplitude_init)
    ampls = ampls.at[mode_i, :].set(draw)

ampls = ampls.at[0, :].set((data_z[:, 1] - data_z[:, 1].min()) / (data_z[:, 1].max() - data_z[:, 1].min()))
ampls = ampls.at[1:4, :].set(ampls[1:4, :] + 1.)

stationary_mode_mean = 1. - (data_z[:, 1] - data_z[:, 1].min()) / (data_z[:, 1].max() - data_z[:, 1].min())
ampls = ampls.at[4:6, :].set(stationary_mode_mean + ampls[4:6, :])
ampls = jnp.maximum(ampls, 0.)

scaling = 1. / jnp.sum(ampls, axis=0)
ampls = ampls * scaling

coord_frame_params_0 = {
    'coordinate_transformation': coordinate_params,
    'mode_0': {
        'amplitude_coeff': rbf_amplitude.get_coeff(ampls[0, :]),
        'mode': mode_init
    },
    'mode_1': {
        'amplitude_coeff': rbf_amplitude.get_coeff(ampls[1, :]),
        'mode': mode_init
    },
    'mode_2': {
        'amplitude_coeff': rbf_amplitude.get_coeff(ampls[2, :]),
        'mode': mode_init
    },
    'mode_3': {
        'amplitude_coeff': rbf_amplitude.get_coeff(ampls[3, :]),
        'mode': mode_init
    }
}

coord_frame_params_1 = {
    'coordinate_transformation': jnp.zeros(0),
    'mode_0': {
        'amplitude_coeff': rbf_amplitude.get_coeff(ampls[4, :]),
        'mode': mode_init
    },
    'mode_1': {
        'amplitude_coeff': rbf_amplitude.get_coeff(ampls[5, :]),
        'mode': mode_init
    }
}

x0_cf_0 = ansatz.coordinate_frames[0].get_params_recursive(coord_frame_params_0)
x0_cf_1 = ansatz.coordinate_frames[1].get_params_recursive(coord_frame_params_1)

x0 = ansatz.get_params({'coord_frame_0': x0_cf_0, 'coord_frame_1': x0_cf_1})

# clip x0 into the bounds (radial coefficients start at 0 < 1e-4, mode values far from the center below 1e-6)
bounds = ansatz.get_bounds()
x0 = jnp.clip(x0, bounds.lb, bounds.ub)

reg_lambda = 0.

print('start compilation')

losses_unscaled = Losses(pts, dx, data_z, data_exact, ansatz, l2_or_l1=1, DOWNSCALED_SIZE=25)
losses_scaled = Losses_scaled(losses_unscaled, x0, data_exact, scaling='hessian')
print('start compilation')

import time
time_start = time.time()
print(losses_scaled.cost(x0), losses_scaled.cost_grad(x0))
print('Compile time', time.time() - time_start)
if __name__ == '__main__':

    params = losses_scaled.optimize(x0, 20)
    params = losses_scaled.optimize(params, 200)
    params = losses_scaled.optimize(params, 600)
    params = losses_scaled.optimize(params, 1000)
    params = losses_scaled.optimize(params, 2000)

    import h5py

    f = h5py.File('wildfire/seed_reg_4_2_s%d.hdf5' % seed, 'w')
    f.create_dataset('params', data=params)
    f.create_dataset('hist_grad_modes', data=jnp.array(losses_scaled.hist_grad_modes))
    f.create_dataset('hist_grad_remainder', data=jnp.array(losses_scaled.hist_grad_remainder))
    
    f.close()
    print(losses_scaled.norm_l2, losses_scaled.norm_w1)
    exit()
    data_arrs = {data_name: [] for data_name in param_hist[0].keys()}

    for entry in param_hist:
        for data_name in entry.keys():
            data_arrs[data_name].append(entry[data_name])

    data_arrs_np = {data_name: numpy.array(data_arrs[data_name]) for data_name in data_arrs.keys()}

    for data_name in data_arrs_np.keys():
        f.create_dataset(data_name, data=data_arrs_np[data_name])
    f.create_dataset('norm_l2s', data=numpy.array(norm_l2s))
    f.create_dataset('norm_w2s', data=numpy.array(norm_w1s))
    f.create_dataset('n_iters', data=numpy.array(n_iters))

    for rbfname, rbf in [('radial', rbf_radial), ('shift_x', rbf_shift_x), ('shift_y', rbf_shift_y), ('amplitude', rbf_amplitude)]:
        f.attrs['%s_sigma' % rbfname] = rbf.sigma
        f.attrs['%s_theta' % rbfname] = rbf.theta
    f.close()
