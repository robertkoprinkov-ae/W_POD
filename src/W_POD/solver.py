import jax.numpy as jnp
import jax

from jax.experimental import checkify

from typing import Dict, Any, List

import jaxopt

from W_POD.ansatz import AnsatzConfig

# only for using the scipy optimizers, which use numpy arrays
import numpy
import scipy

# for writing to hdf5
import h5py
from pathlib import Path

class OptimCfg:
    def __init__(self, stop_if_linesearch_fails, verbose, log_w1, maxiter, scipy_optim_method, callback_prefix=''):
        self.stop_if_linesearch_fails = stop_if_linesearch_fails
        self.verbose = verbose
        self.log_w1 = log_w1
        self.maxiter = maxiter
        self.scipy_optim_method = scipy_optim_method
        self.callback_prefix = callback_prefix

class L2_W1_optimizer():

    def __init__(self, data, ansatz_cfg: AnsatzConfig, optim_cfg: OptimCfg):
        self._data = data
        self.ansatz_cfg = ansatz_cfg
        self.optim_cfg = optim_cfg

        self.ansatz = self.ansatz_cfg.ansatz

    def renormalize_losses(self, params_arr):
        self.ansatz_cfg.update_scaling(params_arr)

    # uses scipy minimize
    def solve_scipy(self, initial):

        # these have already been jitted
        objective_jit = self.ansatz_cfg.cost_jit_scaled
        objective_jit_grad = self.ansatz_cfg.cost_jit_scaled_grad
        
        # renormalize
        self.renormalize_losses(initial)
        
        def objective_numpy(x_optim):
            params_jnp = jnp.array(x_optim)

            return numpy.array(objective_jit(params_jnp), dtype=numpy.float64)
        
        def grad_numpy(x_optim):
            params_jnp = jnp.array(x_optim)
            return numpy.array(objective_jit_grad(params_jnp), dtype=numpy.float64)
        
        # not ideal
        x0 = self.ansatz_cfg.rescale_params.unscaled_to_scaled(initial)

        lb, ub = self.ansatz_cfg.get_bounds_arr()
        lb = numpy.array(lb)
        ub = numpy.array(ub)
        
        bounds = scipy.optimize.Bounds(lb=lb, ub=ub)
        constraints = self.ansatz_cfg.get_constraints()

        _log_l2 = []
        _log_w1 = []
        _log_tot = []
        
        def callback_log_error(intermediate_result: scipy.optimize.OptimizeResult):

            if isinstance(intermediate_result, numpy.ndarray):
                class DummyResult:
                    def __init__(self, x, fun):
                        self.x = x
                        self.fun = fun
                intermediate_result = DummyResult(intermediate_result, -1.)
            
            params_arr_unscaled = self.ansatz_cfg.rescale_params.scaled_to_unscaled(intermediate_result.x)

            l2_tot = self.ansatz_cfg.calc_l2_training(params_arr_unscaled)
            w1_tot = None
            reg, regs_per_mode = self.ansatz_cfg.calc_reg(params_arr_unscaled, verbose=True)

            if self.optim_cfg.log_w1:
                w1_tot = self.ansatz_cfg.calc_w1_training(params_arr_unscaled)
                _log_w1.append(self.ansatz_cfg.weights_loss.w2 * w1_tot)

            if self.optim_cfg.verbose:
                norm_l2, norm_w2 = self.ansatz_cfg.l2_standard, self.ansatz_cfg.w2_standard
                if norm_l2 is None:
                    norm_l2 = -1.
                if norm_w2 is None:
                    norm_w2 = -1.

                if self.optim_cfg.log_w1:
                    print('%sIter %d; error: %.4f; l2: %.4f (norm: %.4f; normalized: %.4f); w1: %.4f (norm: %.4f; normalized: %.4f;) reg: %.2f; rel: %.3f' % (self.optim_cfg.callback_prefix, self._iter, intermediate_result.fun,
                            l2_tot, norm_l2, l2_tot/norm_l2, w1_tot, norm_w2, w1_tot/norm_w2, reg, jnp.sqrt(l2_tot / self.ansatz_cfg.rel_error_scaling_factor())))
                    if self._iter % 50 == 0:
                        print(regs_per_mode)
                else:
                    print('%sIter %d; error: %.4f; l2: %.4f (norm: %.4f); w1: ----' % (self.optim_cfg.callback_prefix, self._iter, intermediate_result.fun,
                                                        l2_tot, norm_l2))
            self._iter += 1
                #if w1_tot > 5e3:
                #    print('Huge W1: %.2f' % w1_tot)
                #    f = h5py.File('data/debug_huge_w1.hdf5', 'w')
                #    f.create_dataset('params_arr', data=intermediate_result.x)
                #    f.close()
            _log_l2.append(self.ansatz_cfg.weights_loss.l2 * l2_tot)
            _log_tot.append(intermediate_result.fun)
        
        self._iter = 0
        result = scipy.optimize.minimize(
            objective_numpy,
            x0,
            method=self.optim_cfg.scipy_optim_method,
            jac = grad_numpy,
            bounds=bounds,
            constraints=constraints,
            callback=callback_log_error,
            options={'maxiter': self.optim_cfg.maxiter, 'maxcor': 50}
        )

        print('nfev, njev', result.nfev, result.njev)
        self._log_l2 = _log_l2
        self._log_w1 = _log_w1
        self._log_tot = _log_tot

        self.params_init = initial
        
        self.result = result
        self.params_result = self.ansatz_cfg.rescale_params.scaled_to_unscaled(result.x)
        
    def filename(self):
        name = '(l2='

        for mode_i, mode in enumerate(self.ansatz_cfg.ansatz.modes):
            for var_name, loss_enabled in self.ansatz_cfg.ansatz_grad[mode_i].items():
                if loss_enabled['l2']:
                    name += var_name + ','

        name += ';w1='

        for mode_i, mode in enumerate(self.ansatz_cfg.ansatz.modes):
            for var_name, loss_enabled in self.ansatz_cfg.ansatz_grad[mode_i].items():
                if loss_enabled['w1']:
                    name += var_name + ','
        name += ')'

        return name
    
    def description_pretty(self):
        name = '({\n l2: '

        for mode_i, mode in enumerate(self.ansatz_cfg.ansatz.modes):
            for var_name, loss_enabled in self.ansatz_cfg.ansatz_grad.grad_cfg[mode_i].items():
                if loss_enabled['l2']:
                    name += var_name + ', '

        name += '\n w1: '

        for mode_i, mode in enumerate(self.ansatz_cfg.ansatz.modes):
            for var_name, loss_enabled in self.ansatz_cfg.ansatz_grad.grad_cfg[mode_i].items():
                if loss_enabled['w1']:
                    name += var_name + ', '
        name += '\n}'

        return name
    
    def write_hdf5(self, params=None, filename='data/output/params.hdf5'):
        if params is None:
            params = self.params_result

        f = h5py.File(filename, 'w')
        
        f.attrs['description'] = self.description_pretty()
        grp_loss = f.create_group('loss')

        grp_loss['l2'] = jnp.array(self._log_l2)
        grp_loss['w1'] = jnp.array(self._log_w1)
        grp_loss['tot'] = jnp.array(self._log_tot)
        
        norm_l2, norm_w2 = self.ansatz_cfg.l2_standard, self.ansatz_cfg.w2_standard
        grp_loss['norm_l2'] = norm_l2
        grp_loss['norm_w2'] = norm_w2

        self.ansatz_cfg.write_hdf5(params, f)

        f.close()

    def calc_gradients_per_param(self, params_arr: jax.Array) -> list[Dict[str, jax.Array]]:

        gradients = [self.ansatz_cfg.ansatz.arr_to_dict(self.ansatz_cfg.cost_grad(params_arr))]

        gradients_per_param_per_mode = []

        for mode_i, mode in enumerate(self.ansatz_cfg.ansatz.modes):
            gradients_per_param_per_mode.append({})
            for param_name in mode.get_param_names():
                gradients_per_param_per_mode[-1][param_name] = []
                for grad in gradients:
                    gradients_per_param_per_mode[-1][param_name].append(grad[mode_i][param_name])
        return gradients_per_param_per_mode

class IteratedSolver():
    """
        Applies an array of solvers one by one a few times
        The solvers are given by their config files and are
        instansiated in this class
    """
    def __init__(self, data, ansatz_cfgs: List[AnsatzConfig], optim_cfgs: List[OptimCfg], n_iter: int):
        self.optim_cfgs = optim_cfgs
        self.n_iter = n_iter

        # TODO add unit test that running L2_W1_optimize twice returns the same result
        solvers = []
        for j in range(len(optim_cfgs)):
            solvers.append(L2_W1_optimizer(data, ansatz_cfgs[j], optim_cfgs[j]))

        self.ansatz_cfgs = ansatz_cfgs
        self.solvers = solvers

        self.ansatz = self.ansatz_cfgs[0].ansatz

        for ansatz_cfg in self.ansatz_cfgs:
            assert(ansatz_cfg.ansatz == self.ansatz)

    def solve(self, initial=None):
        _log_tot = []
        _log_l2 = []
        _log_w1 = []
        params_hist = []
        params = initial
        
        stage_i = 0
        for repeat_i in range(self.n_iter):
            for solver_i, solver in enumerate(self.solvers):
                print('Stage %d' % stage_i)
                solver.solve(params)

                params = solver.params_result
                
                _log_tot = _log_tot + solver._log_tot
                _log_l2 = _log_l2 + solver._log_l2
                _log_w1 = _log_w1 + solver._log_w1
                params_hist.append(params)
                stage_i += 1


        self._log_tot = _log_tot
        self._log_l2 = _log_l2
        self._log_w1 = _log_w1

        self.params_hist = params_hist
        
        self.params_init = initial
        
        self.params_result = self.solvers[-1].params_result
    
    def solve_scipy(self, initial=None, directory_out=None):
        _log_tot = []
        _log_l2 = []
        _log_w1 = []
        params_hist = []
        params = initial
        
        if directory_out is not None:
            Path(directory_out).mkdir(parents=True, exist_ok=True)
            
            # a bit manual since calling solver.write_hdf5 was causing problems
            f = h5py.File(directory_out + 'params_0.hdf5', 'w')
            f.create_dataset('params_arr', data=params)
            f.close()

        stage_i = 0
        for repeat_i in range(self.n_iter):
            for solver_i, solver in enumerate(self.solvers):
                print('Stage %d' % stage_i)

                solver.solve_scipy(params)

                params = solver.params_result
                
                _log_tot = _log_tot + solver._log_tot
                _log_l2 = _log_l2 + solver._log_l2
                _log_w1 = _log_w1 + solver._log_w1
                params_hist.append(params)

                print('Stage %d complete' % stage_i)
                print('Converged', solver.result.success)
                print('Message', solver.result.message)
                print('Status', solver.result.status)
                print('n_iter', solver.result.nit)
                
                if directory_out is not None:
                    solver.write_hdf5(params, filename=directory_out + 'params_%d.hdf5' % (stage_i+1))
                
                stage_i += 1

        self._log_tot = _log_tot
        self._log_l2 = _log_l2
        self._log_w1 = _log_w1
        
        self.params_hist = params_hist

        self.params_init = initial
        
        self.params_result = self.solvers[-1].params_result

    def filename(self):
        name = '%d*[' % self.n_iter

        for j in range(len(self.solvers_cfg)):
            name += self.solvers[j].filename() + ','
        return name + ']'
    
    """
        To be included in the hdf5 file
    """
    def description_pretty(self):
        name = '%d * [\n' % self.n_iter
        for j in range(len(self.solvers_cfg)):
            name += self.solvers[j].description_pretty() + ', '
        name += ']\n'
        return name
    

    def write_hdf5_every(self, directory='data/output/params_every/'):
        Path(directory).mkdir(parents=True, exist_ok=True)

        stage_i = 0

        pars_init = self.params_init
        # can make it static tbh
        self.solvers[-1].write_hdf5(pars_init, filename=directory + 'params_0.hdf5')
            
        for _ in range(self.n_iter):
            for _ in self.solvers:
                pars = self.params_hist[stage_i]
                self.solvers[-1].write_hdf5(pars, filename=directory + 'params_%d.hdf5' % (stage_i+1))
                stage_i += 1

    def calc_gradients_per_param(self, params_arr: jax.Array) -> list[Dict[str, jax.Array]]:

        gradients = []
        for solver_i, ansatz_cfg in enumerate(self.ansatz_cfgs):
            gradients.append(ansatz_cfg.ansatz.arr_to_dict(ansatz_cfg.cost_jit_scaled_grad(params_arr)))

        gradients_per_param_per_mode = []

        for mode_i, mode in enumerate(self.ansatz_cfgs[0].ansatz.modes):
            gradients_per_param_per_mode.append({})
            for param_name in mode.get_param_names():
                gradients_per_param_per_mode[-1][param_name] = []
                for grad in gradients:
                    gradients_per_param_per_mode[-1][param_name].append(grad[mode_i][param_name])
        return gradients_per_param_per_mode
