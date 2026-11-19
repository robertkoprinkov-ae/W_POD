from typing import List

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

import numpy
import scipy

import interpax

# The following 90 lines are a partially-claude written wrapper around interpax 
# that lets the value decay smoothly outside of the interpolation domain.
"""Catmull-Rom (or any interpax method) 2D interpolation with smooth,
decaying extrapolation outside the knot box.

Inside the grid the result is exactly ``interpax.interp2d``. Outside, the
field decays toward ``asymptote`` with rate ``k``. The decay is joined
to the boundary as C0 (value only; a single interp2d pass and not much memory).

The custom-jvp ``clamp`` makes the gradient correct even for query points
that sit exactly on a domain boundary (plain ``jnp.clip`` returns a half
gradient at the tie). Everything is jittable and differentiable.

Switches on ``interp2d_decay``
------------------------------
n_chunks : int | None
    If set (>1), the query points are processed in this many sequential
    chunks via ``jax.lax.map``, cutting peak memory ~n_chunks-fold at the
    cost of running serially. ``None``/1 means no chunking. Useful when the
    full 250k-point batch OOMs; independent of any outer lax.map batch_size.
"""

@jax.custom_jvp
def clamp(v, lo, hi):
    """Like ``jnp.clip`` but with gradient 1 (not 0.5) on the boundary."""
    return jnp.clip(v, lo, hi)


@clamp.defjvp
def _clamp_jvp(primals, tangents):
    v, lo, hi = primals
    vdot, _, _ = tangents
    inside = (v >= lo) & (v <= hi)          # ties count as inside -> grad 1
    return jnp.clip(v, lo, hi), jnp.where(inside, vdot, 0.0)

def _core_c0(xq, yq, x, y, f, method, k, a):
    """C0 decay: matches value only. One interp2d pass."""
    bx = clamp(xq, x[0], x[-1])
    by = clamp(yq, y[0], y[-1])
    sx = jnp.abs(xq - bx)
    sy = jnp.abs(yq - by)
    c = interpax.interp2d(bx, by, x, y, f, method=method) - a
    return a + c * jnp.exp(-k * sx) * jnp.exp(-k * sy)

@jax.checkpoint
def interp2d_decay(xq, yq, x, y, f, k=1.0, asymptote=0.0, n_chunks=None):
    """Interpolate ``f`` on grid ``(x, y)``, decaying smoothly outside the box.

    Parameters
    ----------
    xq, yq : array
        Query coordinates (same shape, broadcast together).
    x, y : array
        1D grid coordinates ("knots"), strictly increasing.
    f : array, shape (len(x), len(y), ...)
        Function values on the grid.
    k : float
        Decay rate outside the box. Larger = faster decay toward ``asymptote``.
    asymptote : float
        Value the field settles to far outside the box. Set to your data's
        baseline to avoid a large swing when the edge values are nonzero.
    n_chunks : int or None
        Process queries in this many sequential chunks to cut peak memory.

    Returns
    -------
    array
        Interpolated / extrapolated values, same shape as xq, yq.
    """
    method = 'catmull-rom'

    a = asymptote
    core = _core_c0

    if n_chunks is None or n_chunks <= 1:
        return core(xq, yq, x, y, f, method, k, a)

    # ---- chunk the query axis to cut peak memory -------------------------
    shape = xq.shape
    xf = xq.reshape(-1)
    yf = yq.reshape(-1)
    n = xf.shape[0]
    pad = (-n) % n_chunks
    xp = jnp.concatenate([xf, jnp.full(pad, x[0], xf.dtype)]).reshape(n_chunks, -1)
    yp = jnp.concatenate([yf, jnp.full(pad, y[0], yf.dtype)]).reshape(n_chunks, -1)
    out = jax.lax.map(lambda ab: core(ab[0], ab[1], x, y, f, method, k, a), (xp, yp))
    return out.reshape(-1)[:n].reshape(shape)



class IdentityCoordinateTransformation:
    def __init__(self):
        pass

    def n_param(self):
        return 0

    def get_bounds(self):
        return jnp.zeros(0), jnp.zeros(0)

    def get_reg(self, params):
        return 0.

    def split_params(self, params):
        return {}

    def split_params_recursive(self, params):
        return {}

    def get_params(self, params_split):
        return jnp.zeros(0)

    def get_params_recursive(self, params_split):
        return jnp.zeros(0)

    def transform(self, pts, z, params):
        return jnp.copy(pts)

    def inv_transform(self, pts, z, params):
        return jnp.copy(pts)

    def plot_coeff(self, *args):
        pass


class CoordinateTransformation:
    def __init__(self, n_discret_theta, rbf_radial_shifts, rbf_shift_x, rbf_shift_y, fix_center=False):
        self.n_discret_theta = n_discret_theta
        self.rbf_radial_shifts = rbf_radial_shifts
        self.rbf_shift_x = rbf_shift_x
        self.rbf_shift_y = rbf_shift_y

        self.fix_center = fix_center

        self.theta_discret = jnp.linspace(0., 2 * jnp.pi, n_discret_theta + 1)[:-1]

        self.radial_station_param_names = ['radial_%d_coeff' % theta_i for theta_i in range(n_discret_theta)]
        self.param_names = self.radial_station_param_names + ['shift_x_coeff', 'shift_y_coeff']

        delimiters = [0]
        for i in range(n_discret_theta):
            delimiters.append(delimiters[-1] + self.rbf_radial_shifts.n_param())
        delimiters.append(delimiters[-1] + self.rbf_shift_x.n_param())
        delimiters.append(delimiters[-1] + self.rbf_shift_y.n_param())

        self.delimiters_by_param_name = {}
        for param_i, param_name in enumerate(self.param_names):
            self.delimiters_by_param_name[param_name] = (delimiters[param_i], delimiters[param_i + 1])

    def n_param(self):
        return self.n_discret_theta * self.rbf_radial_shifts.n_param() + self.rbf_shift_x.n_param() + self.rbf_shift_y.n_param()

    def get_bounds(self):
        lb = -jnp.inf * jnp.ones(self.n_param())
        lb = lb.at[: self.n_discret_theta * self.rbf_radial_shifts.n_param()].set(1e-4)
        ub = jnp.inf * jnp.ones(self.n_param())

        return lb, ub

    def get_reg(self, params):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)
        reg = self.rbf_shift_x.calc_RKHS_reg(params_by_name['shift_x_coeff'])
        reg += self.rbf_shift_y.calc_RKHS_reg(params_by_name['shift_y_coeff'])

        for radial_station_name in self.radial_station_param_names:
            reg += self.rbf_radial_shifts.calc_RKHS_reg(params_by_name[radial_station_name])

        return reg

    def split_params(self, params):
        params_by_name = {}
        for param_name in self.param_names:
            start_i, end_i = self.delimiters_by_param_name[param_name]
            params_by_name[param_name] = params[start_i: end_i]
        return params_by_name

    def split_params_recursive(self, params):
        # leaf node
        return self.split_params(params)

    def get_params(self, params_split):
        params = jnp.zeros(self.n_param())

        for param_name in self.param_names:
            start_i, end_i = self.delimiters_by_param_name[param_name]
            params = params.at[start_i: end_i].set(params_split[param_name])
        return params

    def get_params_recursive(self, params_split):
        # leaf node
        return self.get_params(params_split)

    def get_shift_center(self, pts, z, params):
        params_by_name = self.split_params(params)

        center_x = (pts[:, 0].max() + pts[:, 0].min()) / 2.
        center_y = (pts[:, 1].max() + pts[:, 1].min()) / 2.

        if self.fix_center:
            return (center_x + 0., center_y + 0.)

        rigid_body_shift_x = self.rbf_shift_x.predict(z, params_by_name['shift_x_coeff'])
        rigid_body_shift_y = self.rbf_shift_y.predict(z, params_by_name['shift_y_coeff'])

        return (center_x + rigid_body_shift_x, center_y + rigid_body_shift_y)

    def transform(self, pts, z, params):
        params_by_name = self.split_params(params)

        # apply coordinate transformation to the pts
        # calculate azimuth for each point
        center_x = (pts[:, 0].max() + pts[:, 0].min()) / 2.
        center_y = (pts[:, 1].max() + pts[:, 1].min()) / 2.
        pts_centered_x = pts[:, 0] - center_x
        pts_centered_y = pts[:, 1] - center_y
        pts_theta = (jnp.arctan2(pts_centered_y, pts_centered_x) + 2 * jnp.pi) % (2 * jnp.pi)

        # evaluate values of radial shift at the polar interpolation pts
        coeffs_radial_shifts = jnp.zeros((self.n_discret_theta, self.rbf_radial_shifts.n_param()))
        for i in range(self.n_discret_theta):
            coeffs_radial_shifts = coeffs_radial_shifts.at[i, :].set(params_by_name['radial_%d_coeff' % i])

        radial_shifts_interp_pts = jax.vmap(lambda coeff_i: self.rbf_radial_shifts.predict(z, coeff_i), (0,))(
            coeffs_radial_shifts)

        # calculate amount of radial shift for each pt by interpolating
        radial_shifts_pts = jnp.interp(pts_theta, self.theta_discret, radial_shifts_interp_pts, period=2 * jnp.pi)

        # assumes the amount of the shift is relative, i.e. given as a factor
        incr_per_pt = radial_shifts_pts

        # calculate shift_x and shift_y, rigid body shifts
        rigid_body_shift_x, rigid_body_shift_y = None, None
        if self.fix_center:
            rigid_body_shift_x = 0.
            rigid_body_shift_y = 0.
        else:
            rigid_body_shift_x = self.rbf_shift_x.predict(z, params_by_name['shift_x_coeff'])
            rigid_body_shift_y = self.rbf_shift_y.predict(z, params_by_name['shift_y_coeff'])

        shifted_x = rigid_body_shift_x + center_x + pts_centered_x * incr_per_pt  # radial_shifts_pts
        shifted_y = rigid_body_shift_y + center_y + pts_centered_y * incr_per_pt  # radial_shifts_pts

        shifted = jnp.concatenate((shifted_x.reshape((-1, 1)), shifted_y.reshape((-1, 1))), axis=1)

        return shifted

    def inv_transform(self, pts, z, params):
        params_by_name = self.split_params(params)

        center_x = (pts[:, 0].max() + pts[:, 0].min()) / 2.
        center_y = (pts[:, 1].max() + pts[:, 1].min()) / 2.

        # calculate shift_x and shift_y, rigid body shifts
        rigid_body_shift_x, rigid_body_shift_y = None, None
        if self.fix_center:
            rigid_body_shift_x = 0.
            rigid_body_shift_y = 0.
        else:
            rigid_body_shift_x = self.rbf_shift_x.predict(z, params_by_name['shift_x_coeff'])
            rigid_body_shift_y = self.rbf_shift_y.predict(z, params_by_name['shift_y_coeff'])

        pts_centered_x = pts[:, 0] - rigid_body_shift_x - center_x
        pts_centered_y = pts[:, 1] - rigid_body_shift_y - center_y
        pts_theta = (jnp.arctan2(pts_centered_y, pts_centered_x) + 2 * jnp.pi) % (2 * jnp.pi)

        # evaluate values of radial shift at the polar interpolation pts
        coeffs_radial_shifts = jnp.zeros((self.n_discret_theta, self.rbf_radial_shifts.n_param()))
        for i in range(self.n_discret_theta):
            coeffs_radial_shifts = coeffs_radial_shifts.at[i, :].set(params_by_name['radial_%d_coeff' % i])
        radial_shifts_interp_pts = jax.vmap(lambda coeff_i: self.rbf_radial_shifts.predict(z, coeff_i), (0,))(
            coeffs_radial_shifts)
        
        # calculate amount of radial shift for each pt by interpolating
        radial_shifts_pts = jnp.interp(pts_theta, self.theta_discret, radial_shifts_interp_pts, period=2 * jnp.pi)
        
        # cf transform function above
        incr_per_pt = 1. / radial_shifts_pts

        old_x = center_x + pts_centered_x * incr_per_pt
        old_y = center_y + pts_centered_y * incr_per_pt

        old_pts = jnp.concatenate((old_x.reshape((-1, 1)), old_y.reshape((-1, 1))), axis=1)
        return old_pts

    def plot_coeff(self, params, data_z, filename):
        params_by_name = self.split_params(params)

        fig = plt.figure(figsize=(5. * data_z.shape[0], 10.))

        z_discret = jnp.linspace(data_z.min(), data_z.max(), 100)
        ax_x = fig.add_subplot(2, data_z.shape[0], data_z.shape[0] + 1)
        ax_x.plot(z_discret, self.rbf_shift_x.predict(z_discret, params_by_name['shift_x_coeff']))
        ax_x.plot(data_z, self.rbf_shift_x.predict(data_z, params_by_name['shift_x_coeff']), marker='x', color='red')
        ax_x.grid()
        ax_x.set_xlabel('z')
        ax_x.set_ylabel('shift_x')

        ax_y = fig.add_subplot(2, data_z.shape[0], data_z.shape[0] + 2)
        ax_y.plot(z_discret, self.rbf_shift_y.predict(z_discret, params_by_name['shift_y_coeff']))
        ax_y.scatter(data_z, self.rbf_shift_y.predict(data_z, params_by_name['shift_y_coeff']), marker='x', color='red')
        ax_y.grid()
        ax_y.set_xlabel('z')
        ax_y.set_ylabel('shift_y')

        radial_shift_vals = jnp.zeros((data_z.shape[0], self.n_discret_theta))
        for radial_i, radial_station_name in enumerate(self.radial_station_param_names):
            radial_shift_vals = radial_shift_vals.at[:, radial_i].set(
                self.rbf_radial_shifts.predict(data_z, params_by_name[radial_station_name]))

        theta_discret_dense = jnp.linspace(0., 2 * jnp.pi, 1000)

        for z_i, z in enumerate(data_z):
            ax_station = fig.add_subplot(2, data_z.shape[0], z_i + 1, projection='polar')
            ax_station.plot(theta_discret_dense, jnp.ones_like(theta_discret_dense), linewidth=2., color='red',
                            linestyle='--')

            radial_shifts_interp = jnp.interp(theta_discret_dense, self.theta_discret, radial_shift_vals[z_i, :],
                                              period=2 * jnp.pi)
            ax_station.plot(theta_discret_dense, radial_shifts_interp, color='black')
            ax_station.scatter(self.theta_discret, radial_shift_vals[z_i, :], marker='x', color='black')

            ax_station.set_xlabel(r'$\theta$')
            ax_station.set_ylabel(r'$r(\theta)$')

            ax_station.set_ylim(0., 2.)

            ax_station.set_title('z=%.2f' % z)

        for station_i in range(2, data_z.shape[0]):
            ax_station = fig.add_subplot(2, data_z.shape[0], data_z.shape[0] + station_i)
            vals = self.rbf_radial_shifts.predict(z_discret,
                                                  params_by_name[self.radial_station_param_names[station_i - 2]])
            ax_station.plot(z_discret, vals)

            data_z_vals = self.rbf_radial_shifts.predict(data_z,
                                                         params_by_name[self.radial_station_param_names[station_i - 2]])
            ax_station.scatter(data_z, data_z_vals, marker='x', color='red')
            ax_station.plot(z_discret, self.rbf_radial_shifts.predict(z_discret, params_by_name[
                self.radial_station_param_names[station_i - 2]]))

        plt.tight_layout()
        plt.savefig(filename)


class Mode:

    def __init__(self, n_pts, amplitude_fixed, rbf_amplitudes=None):
        self.n_pts = n_pts
        self.amplitude_fixed = amplitude_fixed
        self.rbf_amplitudes = rbf_amplitudes

        if self.amplitude_fixed:
            self.param_names = ['mode']
            self.delimiters_per_param = {'mode': (0, self.n_pts)}
        else:
            self.param_names = ['mode', 'amplitude_coeff']
            self.delimiters_per_param = {'mode': (0, self.n_pts),
                                         'amplitude_coeff': (self.n_pts, self.n_pts + self.rbf_amplitudes.n_param())}

    def n_param(self):
        if self.amplitude_fixed:
            return self.n_pts
        else:
            return self.n_pts + self.rbf_amplitudes.n_param()

    def get_bounds(self):
        lb = -jnp.inf * jnp.ones(self.n_param())
        lb = 1e-6 * jnp.ones(self.n_param())

        ub = jnp.inf * jnp.ones(self.n_param())

        return lb, ub

    def split_params(self, params):
        params_by_name = {}
        for param_name in self.param_names:
            start_i, end_i = self.delimiters_per_param[param_name]
            params_by_name[param_name] = params[start_i: end_i]
        return params_by_name

    def split_params_recursive(self, params):
        # leaf node
        return self.split_params(params)

    def get_params(self, params_split):
        params = jnp.zeros(self.n_param())

        for param_name in self.param_names:
            start_i, end_i = self.delimiters_per_param[param_name]
            params = params.at[start_i: end_i].set(params_split[param_name])
        return params

    def get_params_recursive(self, params_split):
        # leaf node
        return self.get_params(params_split)

    def get_mode(self, z, params):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)

        if self.amplitude_fixed:
            return params_by_name['mode']
        else:
            return jnp.maximum(0., self.rbf_amplitudes.predict(z, params_by_name['amplitude_coeff'])) * params_by_name[
                'mode']

    def get_reg(self, params):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)
        return self.rbf_amplitudes.calc_RKHS_reg(params_by_name['amplitude_coeff'])

    def plot_coeff(self, vis, params, ax):
        params_by_name = self.split_params(params)
        vis.plot_data(params_by_name['mode'], ax[0])
        plt.colorbar(ax[0].collections[-1])

        z_discret = jnp.linspace(self.rbf_amplitudes.data_z.min(), self.rbf_amplitudes.data_z.max(), 100)

        if self.amplitude_fixed:
            ax[1].axis('off')
        else:
            ax[1].plot(z_discret, self.rbf_amplitudes.predict(z_discret, params_by_name['amplitude_coeff']))
            ax[1].set_xlabel('z')
            ax[1].set_ylabel('amplitude')
            ax[1].grid()


class CoordinateFrame():

    def __init__(self, pts, x_crds, y_crds, data_z, coordinate_transformation: CoordinateTransformation,
                 modes: List[Mode], fixed=False):
        self.pts = pts
        self.x_crds = x_crds
        self.y_crds = y_crds
        self.data_z = data_z

        self.coordinate_transformation = coordinate_transformation
        self.modes = modes
        self.fixed = fixed
        
        self.n_x, self.n_y = self.x_crds.shape[0], self.y_crds.shape[0] 
        
        self.param_names = ['coordinate_transformation'] + ['mode_%d' % mode_i for mode_i in range(len(self.modes))]

        delimiters = [0, coordinate_transformation.n_param()]

        for mode in self.modes:
            delimiters.append(delimiters[-1] + mode.n_param())

        assert (delimiters[-1] == self.n_param())

        delimiters_per_param = {}
        for param_i, param_name in enumerate(self.param_names):
            delimiters_per_param[param_name] = (delimiters[param_i], delimiters[param_i + 1])

        self.delimiters_per_param = delimiters_per_param

    def n_param(self):
        tot = self.coordinate_transformation.n_param()
        for mode in self.modes:
            tot += mode.n_param()

        return tot

    def get_bounds(self):
        lb, ub = self.coordinate_transformation.get_bounds()

        for mode in self.modes:
            lb_mode, ub_mode = mode.get_bounds()
            lb = jnp.concatenate((lb, lb_mode))
            ub = jnp.concatenate((ub, ub_mode))
        return scipy.optimize.Bounds(lb=lb, ub=ub)

    def split_params(self, params):
        params_split = {}

        for param_name in self.param_names:
            start_i, end_i = self.delimiters_per_param[param_name]
            params_split[param_name] = params[start_i: end_i]
        return params_split

    def split_params_recursive(self, params):
        params_split = {}

        params_split_flat = self.split_params(params)
        params_split['coordinate_transformation'] = self.coordinate_transformation.split_params_recursive(
            params_split_flat['coordinate_transformation'])

        for mode_i, mode in enumerate(self.modes):
            params_split['mode_%d' % mode_i] = self.modes[mode_i].split_params_recursive(
                params_split_flat['mode_%d' % mode_i])
        return params_split

    def get_params(self, params_split):
        params = jnp.zeros(self.n_param())

        for param_name in self.param_names:
            start_i, end_i = self.delimiters_per_param[param_name]
            params = params.at[start_i: end_i].set(params_split[param_name])
        return params

    def get_params_recursive(self, params_split):
        params = jnp.zeros(self.n_param())

        start_i, end_i = self.delimiters_per_param['coordinate_transformation']
        params = params.at[start_i: end_i].set(
            self.coordinate_transformation.get_params_recursive(params_split['coordinate_transformation']))

        for mode_i, mode in enumerate(self.modes):
            start_i, end_i = self.delimiters_per_param['mode_%d' % mode_i]
            params = params.at[start_i: end_i].set(
                self.modes[mode_i].get_params_recursive(params_split['mode_%d' % mode_i]))
        return params

    def get_reg(self, params):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)

        reg = self.coordinate_transformation.get_reg(params_by_name['coordinate_transformation'])
        for mode_i, mode in enumerate(self.modes):
            reg += mode.get_reg(params_by_name['mode_%d' % mode_i])
        return reg

    def get_value_nodes(self, params, z_i):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)
        shifted = self.coordinate_transformation.transform(self.pts, self.data_z[z_i],
                                                           params_by_name['coordinate_transformation'])

        values = self.modes[0].get_mode(self.data_z[z_i], params_by_name['mode_0'])
        for mode_i in range(1, len(self.modes)):
            values += self.modes[mode_i].get_mode(self.data_z[z_i], params_by_name['mode_%d' % mode_i])

        # new!
        if self.fixed or type(self.coordinate_transformation) == IdentityCoordinateTransformation:
            return self.pts, values
        else:
            return shifted, values

    def calc_values(self, z, params_by_name):
        values = self.modes[0].get_mode(z, params_by_name['mode_0'])
        for mode_i in range(1, len(self.modes)):
            values += self.modes[mode_i].get_mode(z, params_by_name['mode_%d' % mode_i])
        return values

    def calc_shift(self, pts, z, params):
        params_by_name = self.split_params(params)
        pts_old = self.coordinate_transformation.inv_transform(pts, z, params_by_name['coordinate_transformation'])

        values = self.calc_values(z, params_by_name)
        H, W = self.y_crds.shape[0], self.x_crds.shape[0]
        interpol = jax.scipy.interpolate.RegularGridInterpolator((self.y_crds, self.x_crds), values.reshape((H, W)).T,
                                                                 method='linear', fill_value=0.)
        values_reconstruction = interpol(pts_old)

        return values_reconstruction
    
    def calc_shift_interpax(self, pts, z, params):
        params_by_name = self.split_params(params)
        pts_old = self.coordinate_transformation.inv_transform(pts, z, params_by_name['coordinate_transformation'])

        values = self.calc_values(z, params_by_name)
        n_x, n_y = self.x_crds.shape[0], self.y_crds.shape[0]
        values_reconstruction = interp2d_decay(pts_old[:, 0], pts_old[:, 1], self.x_crds, self.y_crds, values.reshape((n_y, n_x)).T, k=15.)

        return values_reconstruction

    def get_value_grid_arbitrary(self, params, z):
        assert (params.shape[0] == self.n_param())
        
        if self.fixed or type(self.coordinate_transformation) == IdentityCoordinateTransformation:
            return self.calc_values(z, self.split_params(params))
        else:
            return self.calc_shift_interpax(self.pts, z, params)

    def get_value_grid(self, params, z_i):
        return self.get_value_grid_arbitrary(params, self.data_z[z_i])

    def plot_coeff(self, vis, params, filename_prefix):
        params_by_name = self.split_params(params)

        self.coordinate_transformation.plot_coeff(params_by_name['coordinate_transformation'], self.data_z,
                                                  filename_prefix + '_coord.png')

        fig, ax = plt.subplots(len(self.modes), 2, figsize=(10., 5. * len(self.modes)))

        ax = numpy.atleast_2d(ax)

        for mode_i, mode in enumerate(self.modes):
            mode.plot_coeff(vis, params_by_name['mode_%d' % mode_i], ax[mode_i, :])
        plt.tight_layout()
        plt.savefig(filename_prefix + '_modes.png', dpi=300)


class Ansatz:

    def __init__(self, coordinate_frames: List[CoordinateFrame], data_z):
        self.coordinate_frames = coordinate_frames
        self.data_z = data_z

        delimiters = [0, ]
        self.param_names = []
        self.delimiters_per_param = {}

        for coord_frame_i, coord_frame in enumerate(self.coordinate_frames):
            delimiters.append(delimiters[-1] + coord_frame.n_param())

            self.param_names.append('coord_frame_%d' % coord_frame_i)
            self.delimiters_per_param[self.param_names[-1]] = (delimiters[-2], delimiters[-1])

    def n_param(self):
        tot = 0
        for coord_frame in self.coordinate_frames:
            tot += coord_frame.n_param()
        return tot

    def get_bounds(self):
        bounds = self.coordinate_frames[0].get_bounds()
        lb, ub = bounds.lb, bounds.ub

        for coord_frame in self.coordinate_frames[1:]:
            bounds_coord_frame = coord_frame.get_bounds()
            lb = jnp.concatenate((lb, bounds_coord_frame.lb))
            ub = jnp.concatenate((ub, bounds_coord_frame.ub))
        return scipy.optimize.Bounds(lb=lb, ub=ub)

    def split_params(self, params):
        params_split = {}

        for param_name in self.param_names:
            start_i, end_i = self.delimiters_per_param[param_name]
            params_split[param_name] = params[start_i: end_i]
        return params_split

    def split_params_recursive(self, params):
        params_split = {}

        for coord_frame_i, coord_frame in enumerate(self.coordinate_frames):
            start_i, end_i = self.delimiters_per_param['coord_frame_%d' % coord_frame_i]
            params_split['coord_frame_%d' % coord_frame_i] = coord_frame.split_params_recursive(params[start_i: end_i])
        return params_split

    def get_params(self, params_split):
        params = jnp.zeros(self.n_param())

        for param_name in self.param_names:
            start_i, end_i = self.delimiters_per_param[param_name]
            params = params.at[start_i: end_i].set(params_split[param_name])
        return params

    def get_params_recursive(self, params_split):
        params = jnp.zeros(self.n_param())

        for coord_frame_i, coord_frame in enumerate(self.coordinate_frames):
            start_i, end_i = self.delimiters_per_param['coord_frame_%d' % coord_frame_i]
            params = params.at[start_i: end_i].set(
                coord_frame.get_params_recursive(params_split['coord_frame_%d' % coord_frame_i]))
        return params

    def get_reg(self, params):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)

        reg = 0.
        for coord_tr_i, coord_frame in enumerate(self.coordinate_frames):
            reg += coord_frame.get_reg(params_by_name['coord_frame_%d' % coord_tr_i])
        return reg

    def get_value_nodes(self, params, z_i):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)

        pts, values = self.coordinate_frames[0].get_value_nodes(params_by_name['coord_frame_0'], z_i)

        for coord_frame_i in range(1, len(self.coordinate_frames)):
            pts_coord_frame, values_coord_frame = self.coordinate_frames[coord_frame_i].get_value_nodes(
                params_by_name['coord_frame_%d' % coord_frame_i], z_i)

            pts = jnp.concatenate((pts, pts_coord_frame), axis=0)
            values = jnp.concatenate((values, values_coord_frame), axis=0)
        return pts, values

    def get_value_grid_arbitrary(self, params, z):
        assert (params.shape[0] == self.n_param())

        params_by_name = self.split_params(params)
        values = self.coordinate_frames[0].get_value_grid_arbitrary(params_by_name['coord_frame_0'], z)
        for coord_frame_i in range(1, len(self.coordinate_frames)):
            values = values + self.coordinate_frames[coord_frame_i].get_value_grid_arbitrary(
                params_by_name['coord_frame_%d' % coord_frame_i], z)
        return values

    def get_value_grid(self, params, z_i):
        return self.get_value_grid_arbitrary(params, self.data_z[z_i])
