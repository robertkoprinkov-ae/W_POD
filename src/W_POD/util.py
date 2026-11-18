from typing import Dict, Any

import jax.numpy as jnp
import jax

import h5py

def frankenstein_softplus(x):
    cutoff_point = 0.01
    value_softplus = jnp.log(1.+jnp.exp(2.*cutoff_point))/2.
    scaling = cutoff_point / value_softplus
    return jnp.maximum(1e-7, x)  # was: jnp.where(x > cutoff_point, x, scaling * jnp.log(1.+jnp.exp(2.*x))/2.)

class DataCreation:
    
    # rk: goal of this class is not to blindly generate the data using the ansatz
    # but to generate data that can be used in the optimization, i.e. it must be
    # positive
    # can rewrite this to use the Ansatz class!!!!
    def __init__(self, x, ansatz, ansatz_values: list[Dict[str, Any]], data_z,
                 RBF: Dict[str, 'RBF_interpol'], noise=False):
        self.x = x
        self.data_z = data_z

        self.ansatz = ansatz
        self.ansatz_values = ansatz_values

        self.coeffs_exact = {}

        for mode_i, mode in enumerate(self.ansatz.modes):
            for coeff_name in mode.get_coeff_names():
                variable_name = coeff_name[:coeff_name.rfind('_')] + '_value'
                self.coeffs_exact[coeff_name] = RBF[coeff_name].get_coeff(self.ansatz_values[mode_i][variable_name](data_z))

        self.dx = self.x[1] - self.x[0]
        self._generate_data()

    def _generate_data(self):
        data_exact = []
        
        for i in range(self.data_z.shape[0]):
            data_exact.append(self.get_solution(self.data_z[i]))

        self.data_exact = jnp.array(data_exact)
        self.data_exact = jnp.maximum(self.data_exact, 1e-3)
        # TODO: normalize?

    # TODO can change ansatz_values to be a dictionary containing all the variables
    # but then need some functionality to split this dictionary into the part for each mode
    def get_solution(self, z):
        params_dict = []

        for mode_i, mode in enumerate(self.ansatz.modes):
            params_dict.append({})
            for param_name in self.ansatz_values[mode_i].keys():
                params_dict[-1][param_name] = self.ansatz_values[mode_i][param_name](z)
        return self.ansatz.get_values_grid_exact(params_dict)

class RBF_interpol:
    def __init__(self, theta, sigma, data_z, bias=False, nugget=1e-3, kernel='SE'):
        self.theta = theta
        self.sigma = sigma
        self.data_z = data_z
        
        if len(data_z.shape) == 1:
            self.dim_z = 1
        else:
            self.dim_z = self.data_z.shape[1]

        self.bias = bias
        self.nugget = nugget
        
        self.kernel = kernel

        if kernel == 'SE':
            self.rbf_kernel_scalars = self._rbf_kernel_scalars_SE
        elif kernel == 'matern52':
            self.rbf_kernel_scalars = self._rbf_kernel_scalars_matern52
        elif kernel == 'matern32':
            self.rbf_kernel_scalars = self._rbf_kernel_scalars_matern32
        else:
            raise TypeError('Kernel %s not implemented' % kernel)

        self._K = self._calc_K()

    def n_param(self):
        if self.bias:
            return self.data_z.shape[0] + 1
        else:
            return self.data_z.shape[0]

    """
        This matrix specifies the values at the data poinst self.data_z in terms of the coefficients
        and can be used to specify linear constraints
    """
    def get_linear_constraint_matrix(self):
        A = jnp.zeros((self.data_z.shape[0], self.n_param()))

        A = A.at[:, :self.data_z.shape[0]].set(self._K)

        if self.bias:
            A = A.at[:, -1].set(1.)

        return A

    def get_monotonic_linear_constraints_matrix(self):
        A = jnp.zeros((self.data_z.shape[0]-1, self.n_param()))
        A = A.at[:, :self.data_z.shape[0]].set(self._K[1:, :] - self._K[:-1, :])
        return A

    def _calc_K(self):
        # outer loop: give a column
        # inner loop: vectorize over this column
        K_matrix = jax.vmap(jax.vmap(lambda z_i, z_j: self.rbf_kernel(z_i, z_j), (None, 0)), (0, None))(self.data_z, self.data_z)
        return K_matrix

    def calc_RKHS_reg(self, coeff):
        K_matrix = self._K

        if self.bias:
            coeff = coeff[:-1]

        return coeff.T @ K_matrix @ coeff

    def get_coeff(self, values, nugget=None):
        if nugget is None:
            nugget = self.nugget

        K_matrix = self._K
        K_matrix = K_matrix.at[:, :].add(nugget*jnp.eye(K_matrix.shape[0]))

        if self.bias:
            bias_val = jnp.mean(values)
            values_minus_mean = values - bias_val
            coeff = jnp.linalg.solve(K_matrix, values_minus_mean)

            ret = jnp.zeros(self.n_param())
            ret = ret.at[:-1].set(coeff)
            ret = ret.at[-1].set(bias_val)
            return ret
        else:
            return jnp.linalg.solve(K_matrix, values)

    def _rbf_kernel_scalars_SE(self, z_1, z_2):
        assert(z_1.shape[0] == self.dim_z)
        assert(z_2.shape[0] == self.dim_z)

        r = jnp.linalg.norm(z_1 - z_2)
        return jnp.power(self.sigma, 2.) * jnp.exp(-0.5*jnp.power(r, 2.) / jnp.power(self.theta, 2.))
    
    def _rbf_kernel_scalars_matern52(self, z_1, z_2):
        assert(z_1.shape[0] == self.dim_z)
        assert(z_2.shape[0] == self.dim_z)

        r = jnp.linalg.norm(z_1 - z_2)
        return jnp.power(self.sigma, 2.) * (1 + jnp.sqrt(5) * r / self.theta + (5./3.)*jnp.power(r / self.theta, 2.)) * jnp.exp(- jnp.sqrt(5) * r / self.theta)
    
    def _rbf_kernel_scalars_matern32(self, z_1, z_2):
        assert(z_1.shape[0] == self.dim_z)
        assert(z_2.shape[0] == self.dim_z)

        r = jnp.linalg.norm(z_1 - z_2)
        return jnp.power(self.sigma, 2.) * (1 + jnp.sqrt(3) * r / self.theta) * jnp.exp(- jnp.sqrt(3) * r / self.theta)
    
    # z_1 can be a vector here, but not z_2
    def rbf_kernel(self, z_1, z_2):
        z_1 = jnp.array(z_1)
        z_2 = jnp.array(z_2)
        z_1_pad = z_1.reshape((-1, self.dim_z))
        z_2_pad = z_2.reshape((self.dim_z))
        
        assert(z_1_pad.shape[1] == self.dim_z)
        assert(z_2_pad.shape[0] == self.dim_z)
        
        ret = jax.vmap(self.rbf_kernel_scalars, (0, None))(z_1_pad, z_2_pad)

        if self.dim_z == 1:
            ret = ret.reshape(z_1.shape[:1] + z_2.shape[:1])
        else: # one bonus dimension always, 1D array means output must be 0D
            ret = ret.reshape(z_1.shape[:-1] + z_2.shape[:-1])
        return ret
    
    # do not accept vectorized z!
    def predict(self, z, coeff, variances=False):
        if self.bias:
            assert(coeff.shape[0] == self.data_z.shape[0] + 1)
        else:
            assert(coeff.shape[0] == self.data_z.shape[0])
        z = jnp.array(z)
        
        output_shape = None
        if self.dim_z == 1:
            output_shape = z.shape[:1]
        else:
            output_shape = z.shape[:-1]
        values = jnp.zeros(output_shape)

        if self.bias:
            values = coeff[-1] * jnp.ones(output_shape)
        values += jax.vmap(lambda coeff_i, data_z_i: coeff_i*self.rbf_kernel(z, data_z_i), (0, 0))(coeff[:self.data_z.shape[0]], self.data_z).sum(axis=0)
        
        if variances:
            K_matrix = self._K
            K_matrix = K_matrix.at[:, :].add(1e-6*jnp.eye(K_matrix.shape[0]))

            z_ = jnp.atleast_1d(z)
            var = jnp.zeros(z_.shape[0])
            
            def calc_one(z_val):
                k_star = self.rbf_kernel(self.data_z, z_val)
                return self.rbf_kernel(z_val, z_val) - k_star.T @ jnp.linalg.solve(K_matrix, k_star)
            var = jax.jit(jax.vmap(calc_one))(z_)
            var = var.reshape(output_shape)

            return values, var
        else: 
            return values

