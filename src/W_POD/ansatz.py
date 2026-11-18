from typing import Dict, Any, Set

import jax.numpy as jnp
import jax
import scipy.optimize

from ott.geometry import pointcloud
from ott.solvers import linear
import ott
from ott.tools import sinkhorn_divergence

from W_POD.modes import Mode

from jax.flatten_util import ravel_pytree

from W_POD.guarded_solver import GuardedNormalCG

from W_POD.hess_diag_probed import calc_hessian_diag_band

from tqdm import tqdm

import lineax as lx

def LUSolver(**kwargs):        # swallows rtol/atol from solve_lineax
    import lineax as lx
    return lx.LU()

class Ansatz():

    def __init__(self, modes: list[Mode]):
        self.modes = modes

        block_start = 0
        delimiters = [block_start]

        for mode in self.modes:
            delimiters.append(block_start + mode.n_param())
            block_start = delimiters[-1]
        self._delimiters = delimiters

    def n_coeff(self) -> int:
        """
        Returns number of coefficients over all modes in this ansatz
        :return: n_coeff
        """
        total = 0

        for mode in self.modes:
            total += mode.n_coeff()
        return total

    def n_param(self) -> int:
        """
        Returns shape of parameter vector describing a function from this ansatz class
        = #coefficients * data.data_z.shape[0] + #POD modes * self._x.shape[0]
        :return: #coefficients * data.data_z.shape[0] + #POD modes * self._x.shape[0]
        """
        total = 0

        for mode in self.modes:
            total += mode.n_param()
        return total

    def get_param_pytree(self) -> list[Set[str]]:
        """
        :return: List of sets containing names of parameters for each mode
        """
        ret = []
        for mode in self.modes:
            ret.append(set(mode.get_param_names()))
        return ret

    def get_coeff_pytree(self) -> list[Set[str]]:
        """
        :return: List of sets containing names of coefficients for each mode
        """
        ret = []
        for mode in self.modes:
            ret.append(set(mode.get_coeff_names()))
        return ret

    def get_bounds_arr(self) -> tuple[jax.Array, jax.Array]:
        """
        Returns bounds for each entry of the vector describing a function from this ansatz class
        :return: lb and ub arrays, each of shape (self.n_param(),)
        """
        lb_arr, ub_arr = jnp.empty(shape=(0,)), jnp.empty(shape=(0,))

        for mode in self.modes:
            lb, ub = mode.get_param_bounds_arr()

            lb_arr = jnp.concatenate((lb_arr, lb))
            ub_arr = jnp.concatenate((ub_arr, ub))

        return lb_arr, ub_arr

    # add test
    def get_constraints(self):
        """
        Return linear constraints for the vector describing a function from this ansatz class
        :return: scipy.optimize.LinearConstraint defining the linear constraints on the parameter vector
        """
        A = jnp.zeros((0, self.n_param()))
        lb = jnp.zeros(0)
        ub = jnp.zeros(0)

        for mode_i, mode in enumerate(self.modes):
            A_mode, lb_mode, ub_mode = mode.get_constraints()

            A_mode_padded = jnp.zeros((A_mode.shape[0], A.shape[1]))
            A_mode_padded = A_mode_padded.at[:, self._delimiters[mode_i]: self._delimiters[mode_i + 1]].set(A_mode)

            A = jnp.concatenate((A, A_mode_padded), axis=0)
            lb = jnp.concatenate((lb, lb_mode), axis=0)
            ub = jnp.concatenate((ub, ub_mode), axis=0)
        return scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible=False)

    def split_params_arr_by_mode(self, params_arr: jax.Array) -> list[jax.Array]:
        """
        Split parameter vector into a list of vectors corresponding to each mode.
        :param params_arr: parameter vector
        :return: parameter vector split by mode
        """
        ret = []

        for mode_i, mode in enumerate(self.modes):
            ret.append(params_arr[self._delimiters[mode_i]: self._delimiters[mode_i + 1]])
        return ret

    def arr_to_dict(self, params_arr: jax.Array) -> list[dict[str, jax.Array]]:
        params_per_mode = self.split_params_arr_by_mode(params_arr)
        ret = []

        for mode_i, mode in enumerate(self.modes):
            ret.append(mode.arr_to_dict(params_per_mode[mode_i]))

        return ret

    def dict_to_arr(self, params_dict_per_mode: list[dict[str, jax.Array]]) -> jax.Array:
        ret = []

        for mode_i, mode in enumerate(self.modes):
            ret.append(mode.dict_to_arr(params_dict_per_mode[mode_i]))
        return jnp.concatenate(ret)

    def get_values_nodes(self, z, params_arr):
        x, values = jnp.empty(shape=(0,)), jnp.empty(shape=(0,))
        params_per_mode = self.split_params_arr_by_mode(params_arr)

        for mode_i, mode in enumerate(self.modes):
            x_mode, values_mode = mode.get_values_nodes(z, params_per_mode[mode_i])

            x = jnp.concatenate((x, x_mode))
            values = jnp.concatenate((values, values_mode))
        return x, values

    def get_values_grid(self, z, params_arr):
        values = jnp.zeros(self.modes[0]._x.shape[0])
        params_per_mode = self.split_params_arr_by_mode(params_arr)

        for mode_i, mode in enumerate(self.modes):
            # [TODO] rename to mode.predict_grid?
            values_mode = mode.get_values_grid(z, params_per_mode[mode_i])

            values = values + values_mode
        return values

    # rename to get_values_grid_from_values?
    def get_values_grid_exact(self, sol_dict_per_mode: list[dict[str, jax.Array]]) -> jax.Array:
        sol = jnp.zeros(self.modes[0]._x.shape[0])

        for mode_i, mode in enumerate(self.modes):
            sol += mode.get_values_grid_exact(sol_dict_per_mode[mode_i])

        return sol

    def write_hdf5(self, params_arr, hdf5_group):
        hdf5_group.create_dataset('params_arr', data=params_arr)
        params_per_mode = self.split_params_arr_by_mode(params_arr)
        for mode_i, mode in enumerate(self.modes):
            grp_mode = hdf5_group.create_group('mode_%d' % mode_i)
            mode.write_hdf5(params_per_mode[mode_i], grp_mode)

class LossWeights:
    def __init__(self, l2, w2, penalty_exceed, reg_norm_pod, normalize, frobenius_or_l2_squared=2.):
        self.l2 = l2
        self.w2 = w2
        self.penalty_exceed = penalty_exceed
        self.reg_norm_pod = reg_norm_pod
        self.normalize = normalize

        self.frobenius_or_l2_squared = frobenius_or_l2_squared

    def write_hdf5(self, grp):
        grp['weight_l2'] = self.l2
        grp['weight_w2'] = self.w2
        grp['weight_penalty_exceed'] = self.penalty_exceed
        grp['weight_reg_norm_pod'] = self.reg_norm_pod
        grp['weight_normalize'] = self.normalize
        grp['weight_frobenius_or_l2_squared'] = self.frobenius_or_l2_squared


class W2Config:
    def __init__(self, tau_a, tau_b, rescale):
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.rescale = rescale

class OptimPhaseGradConfig:
    def __init__(self, ansatz: Ansatz, grad_cfg: list[Dict['str', Dict['str', bool]]]) -> None:
        self.ansatz = ansatz
        self.grad_cfg = grad_cfg

        # verify that "pytrees" coincide
        # TODO add this as a unit test
        grad_cfg_pytree = [mode_cfg.keys() for mode_cfg in grad_cfg]
        assert grad_cfg_pytree == ansatz.get_param_pytree()

    def write_hdf5(self, grp):
        for mode_i, mode in enumerate(self.ansatz.modes):
            for param_name in self.grad_cfg[mode_i].keys():
                for loss_name in self.grad_cfg[mode_i][param_name].keys():
                    grp['%d_%s_%s' % (mode_i, param_name, loss_name)] = self.grad_cfg[mode_i][param_name][loss_name]

class OptimPhaseCoeffConfig:
    def __init__(self, ansatz: Ansatz, coeff_cfg: list[Dict['str', float]]) -> None:
        self.ansatz = ansatz
        self.coeff_cfg = coeff_cfg

        # verify that "pytrees" coincide
        # TODO add this as a unit test
        coeff_cfg_pytree = [mode_cfg.keys() for mode_cfg in coeff_cfg]
        assert coeff_cfg_pytree == ansatz.get_coeff_pytree()

    def write_hdf5(self, grp):
        for mode_i, mode in enumerate(self.ansatz.modes):
            for coeff_name in self.coeff_cfg[mode_i].keys():
                grp['%d_%s' % (mode_i, coeff_name)] = self.coeff_cfg[mode_i][coeff_name]


class ParameterRescaling:

    def __init__(self, ansatz_cfg, cost, calc_l2_training, calc_w1_training, method='identity'):
        self.ansatz = ansatz_cfg.ansatz
        self.ansatz_cfg = ansatz_cfg # only need this for debugging purposes
        self.cost = cost
        
        self.calc_l2_training = calc_l2_training
        self.calc_w1_training = calc_w1_training
        
        self.method = method
        
        self.scaling_x = None
    
    def _hess_diag_exact(self, cost, params, norm_l2, norm_w1): # kaduk
        print(norm_l2, norm_w1)
        def hvp(f, params, v):
            value = jax.jvp(jax.grad(f, argnums=0), (params,), (v,))[1]
            return value @ v
        f = lambda p: cost(p, norm_l2, norm_w1, use_danskin=False)
        print('Cost', f(params))                      # cost magnitude
        print(self.ansatz_cfg.calc_reg(params, verbose=True)[1])
        g = jax.grad(f)(params)
        print('Grad', jnp.abs(g).max())  
        #print('Grad', self.ansatz.arr_to_dict(g))  
        # or trade memory for speed with chunks (JAX >= 0.4.31)
        
        def calc_hvp_one(v):
            return hvp(f, params, v)
        H_diag = jnp.zeros(params.shape[0])
        hvp_jit = jax.jit(calc_hvp_one)
        for i in tqdm(range(params.shape[0])):
            v = jnp.zeros(params.shape[0])
            v = v.at[i].set(1.)
            H_diag = H_diag.at[i].set(hvp_jit(v))

        return H_diag
    
    def _hess_exact(self, cost, params, norm_l2, norm_w1): # kaduk
        def hvp(f, params, v):
            return jax.jvp(jax.grad(f, argnums=0), (params,), (v,))[1]
        
        f = lambda p: cost(p, norm_l2, norm_w1, use_danskin=False)
        
        print('Cost', f(params))                      # cost magnitude
        print(self.ansatz_cfg.calc_reg(params, verbose=True)[1])
        g = jax.grad(f)(params)
        print('Grad', jnp.abs(g).max())  

        hvp_jit = jax.jit(hvp)
        H = jnp.zeros((params.shape[0], params.shape[0]))
        for i in tqdm(range(params.shape[0])):
            v = jnp.zeros(params.shape[0])
            v = v.at[i].set(1.)
            H = H.at[i, :].set(hvp(f, params, v))

        return H

    # implements 10.1016/j.apnum.2007.01.003; thanks claude
    def _hess_diag(self, f, *args, key, argnums=0, n_samples=256, chunk=8):
        idx = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        x, unflatten = ravel_pytree([args[i] for i in idx])

        def restricted(z):
            full = list(args)
            for i, v in zip(idx, unflatten(z)):
                full[i] = v
            return f(*full)

        g = jax.grad(restricted)
        hvp = lambda v: jax.jvp(g, (x,), (v,))[1]

        def one(k):
            v = jax.random.rademacher(k, x.shape, dtype=x.dtype)
            return v * hvp(v)

        def step(acc, ks):
            return acc + jax.vmap(one)(ks).sum(0), None

        keys = jax.random.split(key, n_samples).reshape(-1, chunk)
        acc, _ = jax.lax.scan(step, jnp.zeros_like(x), keys)
        diag = acc / n_samples
        out = unflatten(diag)
        return (out[0] if isinstance(argnums, int) else tuple(out)) 

    def reset_scaling(self, x0):
        self.x0 = x0
        norm_l2 = self.calc_l2_training(x0)
        norm_w1 = self.calc_w1_training(x0)
        
        if self.method == 'identity':
            self.scaling_x = jnp.ones(x0.shape[0])
        elif self.method == 'hessian':
            # do hessian rescaling
            #hess = self._hess_diag(self.cost, x0, norm_l2, norm_w1, key=jax.random.key(37), n_samples=4096, chunk=16)
            #print('Size: ', x0.shape[0])
            #hess = self._hess_diag_exact(self.cost, x0, norm_l2, norm_w1)
            print('Hi')
            hess = calc_hessian_diag_band(self.ansatz_cfg, x0)
            #print('Hessian: ', self.ansatz.arr_to_dict(hess))
            print('Hessian: min: ', hess.min(), ' max: ', hess.max())
            scaling = 1./jnp.sqrt(jnp.maximum(jnp.abs(hess), 1e-6 * jnp.max(jnp.abs(hess))))  # floor 1e-8 -> 1e-6 (s_max/s_min <= 1e3), see MODIFICATIONS.md
            
            print('Scaling (pre-norm): min: ', scaling.min(), ' max: ', scaling.max())
            #print('Scaling (pre-norm): ', self.ansatz.arr_to_dict(scaling))
            
            normalization_geometric_mean = jnp.exp(jnp.mean(jnp.log(scaling)))
            #scaling /= normalization_geometric_mean
            
            #print('Scaling (post-norm): min: ', scaling.min(), ' max: ', scaling.max())
            hess_diag_new_approx = jnp.diag(hess) * jnp.power(scaling, 2.)
            print('New diag (approx) min/max:', hess_diag_new_approx.min(), hess_diag_new_approx.max())
            
            #normalization_geometric_mean_new = jnp.exp(jnp.mean(jnp.log(scaling)))
            #print('New geometric mean', normalization_geometric_mean_new)
            
            self.scaling_x = scaling

            #def f_new(p, norm_l2, norm_w1):
            #    return self.cost(self.scaling_x*p, norm_l2, norm_w1)
            #hess_new = self._hess_diag_exact(f_new, x0/self.scaling_x, norm_l2, norm_w1)
            #print('Hessian_new: ', self.ansatz.arr_to_dict(hess_new))
            #print('Hesian: min: ', hess_new.min(), ' max: ', hess_new.max())
        else:
            raise NameError()
    
    def scaled_to_unscaled(self, x):
        # convert from scaled to unscaled
        return x * self.scaling_x
    
    def unscaled_to_scaled(self, x):
        # convert from unscaled to scaled
        return x / self.scaling_x
    
    def unscaled_to_scaled_bounds(self, lb, ub):
        return lb / self.scaling_x, ub / self.scaling_x

    def unscaled_to_scaled_constraints(self, constraints):
        lb, ub, A, keep_feasible = constraints.lb, constraints.ub, constraints.A, constraints.keep_feasible

        A = jax.vmap(lambda row, scaling: scaling*row, in_axes=(1, 0), out_axes=(1))(A, self.scaling_x)
        return scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible)

class AnsatzConfig:

    def __init__(self, data: 'DataCreation', weights_loss: LossWeights, w2_config: W2Config,
                 ansatz: Ansatz,
                 ansatz_grad: OptimPhaseGradConfig, ansatz_coeff_reg: OptimPhaseCoeffConfig, variable_scaling_method: str = 'identity'):
        self.data = data

        # config related to loss
        self.weights_loss = weights_loss
        self.w2_config = w2_config

        # config related to coeff
        self.ansatz = ansatz
        self.ansatz_grad = ansatz_grad
        self.ansatz_coeff_reg = ansatz_coeff_reg

        self._cost_grad = jax.grad(self._cost)
        self._cost_grad_scaled = jax.grad(self._cost_scaled)
        
        # assuming L_2 of each snapshot squared is used, and not frobenius of snapshot matrix
        self._l2_standard_min = 0.0001 * self.rel_error_scaling_factor()
        self._w2_standard_min = 0.001
        print('Minimum l2 normalization', self._l2_standard_min)
        
        self.l2_standard = None
        self.w2_standard = None
        
        self.rescale_params = ParameterRescaling(self, self._cost, self.calc_l2_training, self.calc_w1_training, method=variable_scaling_method)

        assert(self.ansatz_grad.ansatz == ansatz)
        assert(self.ansatz_coeff_reg.ansatz == ansatz)
    
    """
        Returns the term with which we scale the relative L_2 error. This is typically the square of the frobenius norm of the snapshot matrix.
    """
    def rel_error_scaling_factor(self):
        return self.data.dx * jnp.sum(jnp.power(jnp.sum(jnp.power(self.data.data_exact, 2.), axis=1), self.weights_loss.frobenius_or_l2_squared))

    def update_scaling(self, params_arr: jax.Array):
        self.l2_standard = jnp.maximum(self.calc_l2_training(params_arr), self._l2_standard_min)
        #self.l2_standard = self.calc_l2_training(params_arr)
        self.w2_standard = jnp.maximum(self.calc_w1_training(params_arr), self._w2_standard_min)
        
        self.rescale_params.reset_scaling(params_arr)
        
    """
        Calculate regularization of the coefficients
    """

    def calc_reg(self, params_arr: jax.Array, verbose=False) -> float:
        params_dict_per_mode = self.ansatz.arr_to_dict(params_arr)
        reg = 0.
        
        regs_per_mode = []
        for mode_i, mode in enumerate(self.ansatz.modes):
            RBFs = mode.get_RBF()

            regs_per_mode.append({})

            for coeff_name in mode.get_coeff_names():
                    reg += self.ansatz_coeff_reg.coeff_cfg[mode_i][coeff_name] * RBFs[coeff_name].calc_RKHS_reg(params_dict_per_mode[mode_i][coeff_name])
                    if verbose:
                        regs_per_mode[-1][coeff_name] = self.ansatz_coeff_reg.coeff_cfg[mode_i][coeff_name] * RBFs[coeff_name].calc_RKHS_reg(params_dict_per_mode[mode_i][coeff_name])
        
        if verbose:
            return reg, regs_per_mode
        else:
            return reg
    
    def calc_pod_norms_reg(self, params_arr: jax.Array) -> float:
        params_dict_per_mode = self.ansatz.arr_to_dict(params_arr)
        reg = 0.

        for mode_i, mode in enumerate(self.ansatz.modes):
            for pod_name in mode.get_pod_names(): # TODO since we only have one mode, we can change this to get_pod_name
                reg += jnp.power(jnp.sum(self.data.dx*jnp.power(params_dict_per_mode[mode_i][pod_name], 2.)) - 1., 2.)
        return reg
    
    def calc_exceed(self, z: jax.Array, params_arr: jax.Array, sol_exact: jax.Array) -> float:
        rec = self.ansatz.get_values_grid(z, params_arr)
        err = jnp.maximum(rec - sol_exact, 0.)

        return self.data.dx * jnp.sum(jnp.power(err, 2.))
    
    def calc_exceed_training(self, params_arr: jax.Array) -> float:
        exceed_tot = jax.vmap(lambda z_i, params_arr, data_exact_i: jnp.power(self.calc_exceed(z_i, params_arr, data_exact_i), 2.), (0, None, 0))(self.data.data_z, params_arr, self.data.data_exact).sum()
        return exceed_tot

    def cut_grad(self, params_arr: jax.Array, loss_term_name: str) -> jax.Array:
        params_dict_per_mode = self.ansatz.arr_to_dict(params_arr)
        ret = []

        for mode_i, mode in enumerate(self.ansatz.modes):
            ret.append({})
            for param_name in mode.get_param_names():
                if self.ansatz_grad.grad_cfg[mode_i][param_name][loss_term_name]:
                    ret[-1][param_name] = params_dict_per_mode[mode_i][param_name]
                else:
                    ret[-1][param_name] = jax.lax.stop_gradient(params_dict_per_mode[mode_i][param_name])
        return self.ansatz.dict_to_arr(ret)

    def calc_l2(self, z: jax.Array, params_arr: jax.Array, sol_exact: jax.Array) -> float:
        rec = self.ansatz.get_values_grid(z, params_arr)

        return self.data.dx * jnp.sum(jnp.power(rec - sol_exact, 2.))

    def calc_l2_training(self, params_arr: jax.Array) -> float:
        l2_vmapped = jax.jit(jax.vmap(lambda z, params_arr, data_exact_i: jnp.power(self.calc_l2(z, params_arr, data_exact_i), self.weights_loss.frobenius_or_l2_squared), (0, None, 0)))
        l2_tot = l2_vmapped(self.data.data_z, params_arr, self.data.data_exact).sum()

        return l2_tot


    """
        Calc W_1 error between predicted data and exact data
    """
    @jax.jit(static_argnames=['self', 'return_full'])
    def calc_w1_div(self, z: float, params_arr: jax.Array, sol_exact: jax.Array, return_full=False) -> float:
        x_pred, y_pred = self.ansatz.get_values_nodes(z, params_arr)
        y_pred = jax.lax.cond(self.w2_config.rescale, lambda x: x / x.sum(), lambda x: x,
                              y_pred)
        sol_exact = sol_exact / jax.lax.stop_gradient(sol_exact.sum())
        
        div, result_objects = sinkhorn_divergence.sinkhorn_divergence(
            pointcloud.PointCloud,
            x_pred.reshape((-1, 1)),
            self.data.x.reshape((-1, 1)),
            a=y_pred,
            b=sol_exact,
            epsilon=0.1,
            solve_kwargs={'threshold': 0.001, 'max_iterations': 100000}
        )
        if return_full:
            return div, result_objects
        else:
            return div


    @jax.jit(static_argnames=['self', 'return_full', 'use_danskin'])
    def calc_w1(self, z: float, params_arr: jax.Array, sol_exact: jax.Array, return_full=False, use_danskin=None) -> float:
        x_pred, y_pred = self.ansatz.get_values_nodes(z, params_arr)
        geom = pointcloud.PointCloud(
            x_pred.reshape((-1, 1)), self.data.x.reshape((-1, 1)), cost_fn=None,
            epsilon=0.1,
        )

        y_pred = jax.lax.cond(self.w2_config.rescale, lambda x: x / x.sum(), lambda x: x,
                              y_pred)
        sol_exact = sol_exact / jax.lax.stop_gradient(sol_exact.sum())
        
        if use_danskin == False:
            # LU for fp32, GuardedNormalCG for fp64; dtype is static under jit, so this is decided at trace time
            dt = jnp.result_type(x_pred, y_pred, sol_exact)
            nonsym = LUSolver if jnp.finfo(dt).bits <= 32 else GuardedNormalCG
            print('calc_w1: dtype %s -> implicit-diff solver %s' % (dt, nonsym.__name__))  # prints once per compilation
            implicit_solver = linear.implicit_differentiation.ImplicitDiff(solver_kwargs={'nonsym_solver': nonsym, 'ridge_kernel': 1e-6})
            result_w1 = linear.solve(geom, a=y_pred, b=sol_exact, threshold=0.001, tau_a=self.w2_config.tau_a,
                                 tau_b=self.w2_config.tau_b, max_iterations=100000, recenter_potentials=True, use_danskin=False, implicit_diff=implicit_solver)
        else:
            result_w1 = linear.solve(geom, a=y_pred, b=sol_exact, threshold=0.001, tau_a=self.w2_config.tau_a,
                                 tau_b=self.w2_config.tau_b, max_iterations=100000, recenter_potentials=True)

        w1 = jnp.where(result_w1.reg_ot_cost < 0., jnp.inf, result_w1.reg_ot_cost)  # W1INF (2026-11-12): +inf when the entropic cost is negative (diverged Sinkhorn), so SLSQP backtracks instead of accepting -inf
        if return_full:
            return w1, result_w1
        else:
            return w1

    """
        Calc W_1 error between predicted for this param set and training set
    """

    @jax.jit(static_argnames=['self', 'use_danskin'])
    def calc_w1_training(self, params_result: jax.Array, use_danskin=None) -> float:
        w1_vmapped = jax.jit(jax.vmap(lambda z, params_result, data_exact_i: jnp.power(self.calc_w1(z, params_result, data_exact_i, use_danskin=use_danskin), 1.), (0, None, 0)))
        w1_tot = w1_vmapped(self.data.data_z, params_result, self.data.data_exact).sum()
        
        #w1_tot_old = 0.
        #for z_i, z in enumerate(self.data.data_z):  # is this correct?
        #    w1_tot_old += jnp.power(self.calc_w1(z, params_result, self.data.data_exact[z_i]), 2.)
        #jax.debug.print('w1_tot old/new {0} {1}', w1_tot_old, w1_tot)
        return w1_tot  # THERE SHOULD NOT BE A SQRT HERE!!!!!!

    @jax.jit(static_argnames=['self', 'use_danskin'])
    def _cost(self, params_arr: jax.Array, normalize_l2: float, normalize_w2: float, use_danskin: bool = None) -> float:

        if self.weights_loss.w2 > 0:
            l2_loss = self.calc_l2_training(self.cut_grad(params_arr, 'l2'))
            w2_loss = self.calc_w1_training(self.cut_grad(params_arr, 'w1'), use_danskin = use_danskin)
            
            if self.weights_loss.normalize:
                l2_loss /= normalize_l2
                w2_loss /= normalize_w2
            

            return self.weights_loss.l2 *  l2_loss +\
                self.weights_loss.w2 * w2_loss + \
                self.calc_reg(params_arr) + self.weights_loss.penalty_exceed * self.calc_exceed_training(params_arr) + self.weights_loss.reg_norm_pod * self.calc_pod_norms_reg(params_arr)
        else:
            l2_loss = self.calc_l2_training(self.cut_grad(params_arr, 'l2'))

            if self.weights_loss.normalize:
                l2_loss /= self.l2_standard

            return self.weights_loss.l2 * l2_loss + \
                self.calc_reg(params_arr) + self.weights_loss.penalty_exceed * self.calc_exceed_training(params_arr) + self.weights_loss.reg_norm_pod * self.calc_pod_norms_reg(params_arr)
    
    @jax.jit(static_argnames=['self'])
    def _cost_scaled(self, params_arr: jax.Array, norm_l2, norm_w1, scaling_x: jax.Array):
        return self._cost(params_arr * scaling_x, norm_l2, norm_w1)

    def cost_jit_scaled(self, params_arr: jax.Array):
        return self._cost_scaled(params_arr, self.l2_standard, self.w2_standard, self.rescale_params.scaling_x)

    def cost_jit_scaled_grad(self, params_arr: jax.Array):
        return self._cost_grad_scaled(params_arr, self.l2_standard, self.w2_standard, self.rescale_params.scaling_x)
    
    def get_bounds_arr(self):
        lb, ub = self.ansatz.get_bounds_arr()

        return self.rescale_params.unscaled_to_scaled_bounds(lb, ub)
    
    def get_constraints(self):
        constraints = self.ansatz.get_constraints()

        return self.rescale_params.unscaled_to_scaled_constraints(constraints)

    def write_hdf5(self, params_arr, file):
        self.ansatz.write_hdf5(params_arr, file)

        grp_cfg = file.create_group('ansatz_cfg')

        self.weights_loss.write_hdf5(grp_cfg)

        grad = grp_cfg.create_group('grad')

        self.ansatz_grad.write_hdf5(grad)

        reg = grp_cfg.create_group('reg')
        self.ansatz_coeff_reg.write_hdf5(reg)
