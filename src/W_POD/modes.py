from typing import Dict

import jax
import jax.numpy as jnp
from W_POD.util import frankenstein_softplus, RBF_interpol

from W_POD.interp import interp_decay, interp_decay_knots

class Mode():

    def __init__(self, x):
        raise NotImplementedError()

    def n_param(self):
        raise NotImplementedError()

    def n_coeff(self):
        return len(self.get_coeff_names())

    def n_pod(self):
        return len(self.get_pod_names())

    def n_param_sol(self):
        raise NotImplementedError()

    def get_mode_name(self):
        raise NotImplementedError()

    """
        Return the names of all parameters
        The below two functions return a subset of this one
    """

    def get_param_names(self):
        raise NotImplementedError()

    """
        Return the names of those parameters that specify coefficients w.r.t. z
    """

    def get_coeff_names(self):
        raise NotImplementedError()

    """
        Return mapping from coeff_name to corresponding RBF object
        This function is used to calculate the RKHS norm regularization
    """
    def get_RBF(self) -> Dict[str, RBF_interpol]:
        raise NotImplementedError()

    """
        Return the names of those parameters that specify the mode shape
    """

    def get_pod_names(self):
        raise NotImplementedError()

    # lb, ub
    def get_param_bounds_arr(self):
        raise NotImplementedError()

    def get_constraints(self):
        raise NotImplementedError()

    def calc_coeff(self, coeff_name: str, z: jax.typing.ArrayLike, coeff: jax.Array,
                   softplus: bool = True, variances: bool = False) -> jax.Array:
        raise NotImplementedError()

    def get_values_nodes(self, z, params_arr):
        raise NotImplementedError()

    def get_values_grid(self, z, params_arr):
        raise NotImplementedError()

    def get_values_grid_exact(self, sol_dict):
        raise NotImplementedError()

    def arr_to_dict(self, params_arr):
        raise NotImplementedError()

    # remark: output size must be exactly equal to n_param
    def dict_to_arr(self, params_dict):
        raise NotImplementedError()

    def write_hdf5(self, params_arr, hdf5_group):
        raise NotImplementedError()


class GlobalMode(Mode):

    def __init__(self, x, RBF_alpha: RBF_interpol):
        self._x = x
        self._RBF_alpha = RBF_alpha

        self.n_z = self._RBF_alpha.data_z.shape[0]

    def n_param(self):
        return self._x.shape[0] + self._RBF_alpha.n_param()

    def n_param_sol(self):
        return self._x.shape[0] + 1

    def get_mode_name(self):
        return 'GlobalMode'

    def get_param_names(self):
        return ['alpha_coeff', 'pod_global']

    def get_coeff_names(self):
        return ['alpha_coeff']

    def get_RBF(self):
        return {'alpha_coeff': self._RBF_alpha}

    def get_pod_names(self):
        return ['pod_global']

    def get_param_bounds_arr(self):
        return jnp.concatenate((1e-6 * jnp.ones(self._x.shape[0]), -jnp.inf*jnp.ones(self._RBF_alpha.n_param()))), \
                               jnp.concatenate((jnp.inf * jnp.ones(self._x.shape[0]), jnp.inf*jnp.ones(self._RBF_alpha.n_param())))

    def get_constraints(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        # constraints on RBF coeff to ensure positivity of interpolant
        A = jnp.zeros((self.n_z, self.n_param()))
        A = A.at[:, self._x.shape[0]:].set(self._RBF_alpha.get_linear_constraint_matrix())
        return A, 1e-6*jnp.ones(self.n_z), 100.*jnp.ones(self.n_z)

    def calc_coeff(self, coeff_name: str, z: jax.typing.ArrayLike, coeff: jax.Array,
                   softplus: bool=True, variances: bool = False) -> jax.Array:
        assert (coeff_name == 'alpha_coeff')

        values, var = self._RBF_alpha.predict(z, coeff, variances=True)

        if softplus:
            values = frankenstein_softplus(values)

        if variances:
            return values, var
        else:
            return values

    def get_values_nodes(self, z, params_arr):
        return self._x, self.get_values_grid(z, params_arr)

    def get_values_grid(self, z, params_arr):
        params_dict = self.arr_to_dict(params_arr)

        return self.calc_coeff('alpha_coeff', z, params_dict['alpha_coeff']) * params_dict['pod_global']
        return frankenstein_softplus(self._RBF_alpha.predict(z, params_dict['alpha_coeff'])) * params_dict['pod_global']

    def get_values_grid_exact(self, sol_dict):
        return frankenstein_softplus(sol_dict['alpha_value']) * sol_dict['pod_global']

    def arr_to_dict(self, params_arr):
        return {'pod_global': params_arr[:self._x.shape[0]], 'alpha_coeff': params_arr[self._x.shape[0]:]}

    def dict_to_arr(self, params_dict):
        return jnp.concatenate((params_dict['pod_global'], params_dict['alpha_coeff']))

    def write_hdf5(self, params_arr, hdf5_group):
        params_dict = self.arr_to_dict(params_arr)

        hdf5_group.create_dataset('pod_global', data=params_dict['pod_global'])
        hdf5_group.create_dataset('alpha_coeff', data=params_dict['alpha_coeff'])


class TransportedMode(Mode):

    def __init__(self, x, RBF_beta: RBF_interpol, RBF_gamma: RBF_interpol, gamma_increasing=False, gamma_decreasing=False):
        self._x = x

        self._RBF_beta = RBF_beta
        self._RBF_gamma = RBF_gamma

        self.n_z = self._RBF_beta.data_z.shape[0]

        self.gamma_increasing = gamma_increasing
        self.gamma_decreasing = gamma_decreasing

    def n_param(self):
        return self._x.shape[0] + self._RBF_beta.n_param() + self._RBF_gamma.n_param()

    def n_param_sol(self):
        return self._x.shape[0] + 2

    def get_mode_name(self):
        return 'TransportedMode'

    def get_param_names(self):
        return ['beta_coeff', 'gamma_coeff', 'pod_local']

    def get_coeff_names(self):
        return ['beta_coeff', 'gamma_coeff']

    def get_RBF(self):
        return {'beta_coeff': self._RBF_beta, 'gamma_coeff': self._RBF_gamma}

    def get_pod_names(self):
        return ['pod_local']

    """
        For alpha coeff + pod mode: (0, 10)
        For gamma coeff: (-10, 10)

        [TODO] This should be adjusted according to the test case
        Setting bounds on coeff doesn't work if theta is big :((
    """

    def get_param_bounds_arr(self):
        lb = jnp.concatenate((1e-6 * jnp.ones(self._x.shape[0]),
                              -jnp.inf * jnp.ones(self._RBF_beta.n_param()),
                              -jnp.inf * jnp.ones(self._RBF_gamma.n_param())))

        ub = jnp.concatenate((jnp.inf * jnp.ones(self._x.shape[0]),
                              jnp.inf * jnp.ones(self._RBF_beta.n_param()),
                              jnp.inf * jnp.ones(self._RBF_gamma.n_param())))
        return lb, ub

    def get_constraints(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        # constraints on RBF coeff to ensure positivity of interpolant
        A = jnp.zeros((self.n_z, self.n_param()))

        A = A.at[:self.n_z, self._x.shape[0]:self._x.shape[0] + self._RBF_beta.n_param()].set(self._RBF_beta.get_linear_constraint_matrix())

        lb = 1e-3 *jnp.ones(self.n_z)
        ub = jnp.inf * jnp.ones(self.n_z)

        if self.gamma_increasing or self.gamma_decreasing:
            # at most one
            assert (not self.gamma_decreasing) or (not self.gamma_increasing)

            A_gamma = jnp.zeros((self.n_z - 1, self.n_param()))
            A_gamma = A_gamma.at[:, -self._RBF_gamma.n_param():].set(self._RBF_gamma.get_monotonic_linear_constraints_matrix())

            lb_gamma, ub_gamma = None, None
            if self.gamma_increasing:
                lb_gamma = jnp.zeros(self.n_z-1)
                ub_gamma = jnp.inf * jnp.ones(self.n_z-1)
            if self.gamma_decreasing:
                lb_gamma = -jnp.inf * jnp.ones(self.n_z-1)
                ub_gamma = jnp.zeros(self.n_z-1)

            A = jnp.concatenate((A, A_gamma), axis=0)
            lb = jnp.concatenate((lb, lb_gamma), axis=0)
            ub = jnp.concatenate((ub, ub_gamma), axis=0)

        return A, lb, ub

    def calc_coeff(self, coeff_name: str, z: jax.typing.ArrayLike, coeff: jax.Array,
                   softplus=True, variances: bool = False) -> jax.Array | tuple[jax.Array, jax.Array]:
        assert (coeff_name in self.get_coeff_names())

        values, var = None, None
        if coeff_name == 'beta_coeff':
            values, var = self._RBF_beta.predict(z, coeff, variances=True)

            if softplus:
                values = frankenstein_softplus(values)
        elif coeff_name == 'gamma_coeff':
            values, var = self._RBF_gamma.predict(z, coeff, variances=True)

        if variances:
            return values, var
        else:
            return values

    def get_values_nodes(self, z, params_arr):
        params_dict = self.arr_to_dict(params_arr)
        beta_value = frankenstein_softplus(self._RBF_beta.predict(z, params_dict['beta_coeff']))

        return self._x + self._RBF_gamma.predict(z, params_dict['gamma_coeff']), beta_value * params_dict['pod_local']

    def get_values_grid(self, z, params_arr):
        params_dict = self.arr_to_dict(params_arr)

        beta_value = frankenstein_softplus(self._RBF_beta.predict(z, params_dict['beta_coeff']))

        old_x = self._x - self._RBF_gamma.predict(z, params_dict['gamma_coeff'])
        
        #values_interp = interp_decay_knots(old_x, self._x, params_dict['pod_local'], method='cubic2', padding_percentage=0.1, extrap=0.)
        values_interp = interp_decay(old_x, self._x, params_dict['pod_local'], method='catmull-rom', decay_rate=15.)

        return beta_value * values_interp

    def get_values_grid_exact(self, sol_dict):
        old_x = self._x - sol_dict['gamma_value']

        return frankenstein_softplus(sol_dict['beta_value']) * jnp.interp(old_x, self._x, sol_dict['pod_local'], 0., 0.)

    def arr_to_dict(self, params_arr):
        return {'pod_local': params_arr[: self._x.shape[0]],
                'beta_coeff': params_arr[self._x.shape[0]: self._x.shape[0] + self._RBF_beta.n_param()],
                'gamma_coeff': params_arr[-self._RBF_gamma.n_param():]}

    def dict_to_arr(self, params_dict):
        return jnp.concatenate((params_dict['pod_local'],
                                params_dict['beta_coeff'],
                                params_dict['gamma_coeff']))

    def write_hdf5(self, params_arr, hdf5_group):
        params_dict = self.arr_to_dict(params_arr)

        hdf5_group.create_dataset('pod_local', data=params_dict['pod_local'])
        hdf5_group.create_dataset('beta_coeff', data=params_dict['beta_coeff'])
        hdf5_group.create_dataset('gamma_coeff', data=params_dict['gamma_coeff'])


class StretchedMode(Mode):

    def __init__(self, x, RBF_beta: RBF_interpol, RBF_gamma_shift: RBF_interpol, RBF_gamma_stretch: RBF_interpol):
        self._x = x

        self._RBF_beta = RBF_beta
        self._RBF_gamma_shift = RBF_gamma_shift
        self._RBF_gamma_stretch = RBF_gamma_stretch

        self.scaling_axis = (self._x.max() + self._x.min()) / 2.

        self.n_z = self._RBF_beta.data_z.shape[0]

    def n_param(self):
        return self._x.shape[0] + self._RBF_beta.n_param() + self._RBF_gamma_shift.n_param() + self._RBF_gamma_stretch.n_param()

    def n_param_sol(self):
        return self._x.shape[0] + 3

    def get_mode_name(self):
        return 'StretchedMode'

    def get_param_names(self):
        return ['beta_coeff', 'gamma_shift_coeff', 'gamma_stretch_coeff', 'pod_local']

    def get_coeff_names(self):
        return ['beta_coeff', 'gamma_shift_coeff', 'gamma_stretch_coeff']

    def get_RBF(self):
        return {'beta_coeff': self._RBF_beta, 'gamma_shift_coeff': self._RBF_gamma_shift, 'gamma_stretch_coeff': self._RBF_gamma_stretch}

    def get_pod_names(self):
        return ['pod_local']

    def get_param_bounds_arr(self):
        lb = jnp.concatenate((1e-6 * jnp.ones(self._x.shape[0]),
                              -jnp.inf * jnp.ones(self._RBF_beta.n_param()),  # beta_coef
                              -jnp.inf * jnp.ones(self._RBF_gamma_shift.n_param()),  # shifts
                              -jnp.inf * jnp.ones(self._RBF_gamma_stretch.n_param())))  # scaling

        ub = jnp.concatenate((jnp.inf * jnp.ones(self._x.shape[0]),
                              jnp.inf * jnp.ones(self._RBF_beta.n_param()),
                              jnp.inf * jnp.ones(self._RBF_gamma_shift.n_param()),
                              jnp.inf * jnp.ones(self._RBF_gamma_stretch.n_param())))
        return lb, ub

    def get_constraints(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        # constraints on RBF coeff to ensure positivity of interpolant
        A = jnp.zeros((self.n_z, self.n_param()))
        A = A.at[:, self._x.shape[0]: self._x.shape[0] + self._RBF_beta.n_param()].set(self._RBF_beta.get_linear_constraint_matrix())

        return A, 0.5*jnp.ones(self.n_z), 100.*jnp.ones(self.n_z)

    def calc_coeff(self, coeff_name: str, z: jax.typing.ArrayLike, coeff: jax.Array,
                   softplus: bool=True, variances: bool = False) -> jax.Array:
        assert (coeff_name in self.get_coeff_names())

        values, var = None, None
        if coeff_name == 'beta_coeff':
            values, var = self._RBF_beta.predict(z, coeff, variances=True)
            if softplus:
                values = frankenstein_softplus(values)
        elif coeff_name == 'gamma_shift_coeff':
            values, var = self._RBF_gamma_shift.predict(z, coeff, variances=True)
        elif coeff_name == 'gamma_stretch_coeff':
            values, var = self._RBF_gamma_stretch.predict(z, coeff, variances=True)

        if variances:
            return values, var
        else:
            return values

    def get_values_nodes(self, z, params_arr):
        params_dict = self.arr_to_dict(params_arr)

        beta_value = frankenstein_softplus(self._RBF_beta.predict(z, params_dict['beta_coeff']))
        gamma_shift = self._RBF_gamma_shift.predict(z, params_dict['gamma_shift_coeff'])
        gamma_stretch = self._RBF_gamma_stretch.predict(z, params_dict['gamma_stretch_coeff'])
        return gamma_shift + self.scaling_axis + gamma_stretch * (self._x - self.scaling_axis), beta_value * \
               params_dict['pod_local']

    def get_values_grid(self, z, params_arr):
        params_dict = self.arr_to_dict(params_arr)

        gamma_shift = self._RBF_gamma_shift.predict(z, params_dict['gamma_shift_coeff'])
        gamma_scale = self._RBF_gamma_stretch.predict(z, params_dict['gamma_stretch_coeff'])

        beta_value = frankenstein_softplus(self._RBF_beta.predict(z, params_dict['beta_coeff']))

        new_x = gamma_shift + self.scaling_axis + gamma_scale * (self._x - self.scaling_axis)
        old_x = (self._x - gamma_shift - self.scaling_axis) / gamma_scale + self.scaling_axis

        values_interp = interp_decay_knots(old_x, self._x, params_dict['pod_local'], method='cubic2', padding_percentage=0.1, extrap=0.)

        return beta_value * values_interp

    def get_values_grid_exact(self, sol_dict):

        gamma_shift = sol_dict['gamma_shift_value']
        gamma_stretch = sol_dict['gamma_stretch_value']

        old_x = (self._x - gamma_shift - self.scaling_axis) / gamma_stretch + self.scaling_axis

        return frankenstein_softplus(sol_dict['beta_value']) * jnp.interp(old_x, self._x, sol_dict['pod_local'], 0., 0.)

    def arr_to_dict(self, param_arr):
        return {'pod_local': param_arr[: self._x.shape[0]],
                'beta_coeff': param_arr[self._x.shape[0]: self._x.shape[0] + self._RBF_beta.n_param()],
                'gamma_shift_coeff': param_arr[self._x.shape[0] + self._RBF_beta.n_param(): self._x.shape[0] + self._RBF_beta.n_param() + self._RBF_gamma_shift.n_param()],
                'gamma_stretch_coeff': param_arr[-self._RBF_gamma_stretch.n_param():]}

    def dict_to_arr(self, param_dict):
        return jnp.concatenate((param_dict['pod_local'],
                                param_dict['beta_coeff'],
                                param_dict['gamma_shift_coeff'],
                                param_dict['gamma_stretch_coeff']))

    def write_hdf5(self, params_arr, file):
        params_dict = self.arr_to_dict(params_arr)

        file.create_dataset('pod_local', data=params_dict['pod_local'])
        file.create_dataset('beta_coeff', data=params_dict['beta_coeff'])
        file.create_dataset('gamma_shift_coeff', data=params_dict['gamma_shift_coeff'])
        file.create_dataset('gamma_stretch_coeff', data=params_dict['gamma_stretch_coeff'])


