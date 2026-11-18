from typing import Dict

import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
from W_POD.solver import IteratedSolver, L2_W1_optimizer

from pathlib import Path

# to cycle through linestyles
from cycler import cycler

#jax.config.update("jax_enable_x64", True)

key = jax.random.key(42)

class Visualizer():

    def __init__(self, data, solver):
        self._data = data
        self._solver = solver
        self.ansatz = solver.ansatz

        # TODO
        # should ONLY be used to calculate l2 and w1, and do things that depend on the weights of the ansatz_cfg
        # we pick the ansatz_cfg corresponding to the first solver, but it represents all of them and should not be
        # used in a way that is specific to the first solver
        self.ansatz_cfg = None

        if isinstance(self._solver, IteratedSolver):
            self.ansatz_cfg = self._solver.solvers[0].ansatz_cfg
        else:
            self.ansatz_cfg = self._solver.ansatz_cfg

        self.random_key = jax.random.key(78)
    

    def _plot_pod(self, pod_name, params_dict, gradients_per_param_dict, ax):
        pod = params_dict[pod_name]
        gradients_pod = gradients_per_param_dict[pod_name]

        ax.plot(self._data.x, pod, label=pod_name)

        try: # TODO
            ax.plot(self._data.x, self._data.global_mode, label='Exact', linestyle='--')
        except:
            pass

        linestyle_cycler = (cycler(color=['r', 'g', 'b']) + cycler(linestyle=['--', ':', '-.']))
        ax.set_prop_cycle(linestyle_cycler)
        for solver_i, grad_pod in enumerate(gradients_pod):
            # 1e-3 is necessary here for the case that grad_pod_global = 0
            scaling = 0.1 * jnp.max(jnp.abs(pod)) / jnp.max(1e-3 + jnp.abs(grad_pod))
            # do not scale up
            scaling = jnp.minimum(scaling, 1.)
            ax.plot(self._data.x, pod - scaling * grad_pod, label='Solver %d' % solver_i)

        ax.set_title(pod_name)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.legend()
        ax.grid()

    def _plot_coeff(self, z_discret, calc_values, coeff, label, ax, gradients_coeff=None, plot_variances=False):
        values, variances = calc_values(z_discret, coeff, variances=True)

        ax.plot(z_discret, values, label=label)
        if plot_variances:
            ax.plot(z_discret, values + 2*jnp.sqrt(variances), linestyle='--', color='gray')
            ax.plot(z_discret, values - 2*jnp.sqrt(variances), linestyle='--', color='gray')
        ax.scatter(self._data.data_z, calc_values(self._data.data_z, coeff), marker='x', label='current')

        # skip first two colors
        linestyle_cycler = cycler(linestyle=['--', ':', '-.']) + cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2:5])
        ax.set_prop_cycle(linestyle_cycler)
        if gradients_coeff is not None:
            for solver_i, grad_coeff in enumerate(gradients_coeff):
                # 1e-3 is necessary here for the case that grad_coeff = 0
                scaling = 0.1*jnp.max(jnp.abs(coeff))/jnp.max(1e-3 + jnp.abs(grad_coeff))
                # do not scale up
                scaling = jnp.minimum(scaling, 1.)

                ax.scatter(self._data.data_z, calc_values(self._data.data_z, coeff-scaling*grad_coeff, softplus=False), marker='+',
                        label='solver %d' % solver_i)
                values = calc_values(z_discret, coeff-scaling*grad_coeff)
                ax.plot(z_discret, values, label='solver %d' % solver_i)

        
        ax.set_title('%s' % label)

        ax.set_xlabel('z')
        ax.set_ylabel(label)
        
        ax.legend()
        ax.grid()

    def plot_pod_and_coeff(self, params_arr, filename='img/pod_gamma.png'):
        params_dict_per_mode = self.ansatz.arr_to_dict(params_arr)

        gradients_per_param_per_mode = self._solver.calc_gradients_per_param(params_arr)

        n_modes = len(self.ansatz.modes)
        n_coeff = self.ansatz.n_coeff()

        fig, ax = plt.subplots(2, n_coeff, figsize=((n_coeff) * 5, 10))

        if n_coeff == 1: # so it is at least 2d
            ax = ax.reshape((-1, 1))

        z_start = self._data.data_z.min()
        z_end = self._data.data_z.max()
        z_interval_length = z_end - z_start
        z_start = z_start - 0.1 * z_interval_length
        z_end = z_end + 0.1 * z_interval_length
        z_discret = jnp.linspace(z_start, z_end, 10000)

        ax_mode_start_i = 0
        for mode_i, mode in enumerate(self.ansatz.modes):

            pod_names = mode.get_pod_names()
            coeff_names = mode.get_coeff_names()
            for j in range(mode.n_coeff()):
                if j < mode.n_pod():
                    self._plot_pod(pod_names[j], params_dict_per_mode[mode_i], gradients_per_param_per_mode[mode_i],
                                   ax[0, ax_mode_start_i + j])
                else:
                    ax[0, ax_mode_start_i + j].axis('off')

            for j in range(mode.n_coeff()):
                coeff_name = coeff_names[j]
                coeff = params_dict_per_mode[mode_i][coeff_name]
                self._plot_coeff(z_discret,
                                 lambda z, coeff, softplus=False, variances=False: mode.calc_coeff(coeff_name, z, coeff,
                                                                                softplus=softplus, variances=variances),
                                 coeff, coeff_name, ax[1, ax_mode_start_i+j],
                                 )
                                 #gradients_coeff=gradients_per_param_per_mode[mode_i][coeff_name], plot_variances=False)
            ax_mode_start_i += mode.n_coeff()

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    # TODO should we include x and data_z.shape[0] = n_z in the ansatz class for easy access and less clumsy code?
    def _plot_samples(self, z, params_dict_per_mode, ax, n_samples=100):
        mu = jnp.zeros(self.ansatz.n_coeff())
        cov = jnp.zeros((mu.shape[0], mu.shape[0]))

        coeff_i = 0
        for mode_i, mode in enumerate(self.ansatz.modes):
            for coeff_name in mode.get_coeff_names():
                mean_val, var_val = mode.calc_coeff(coeff_name, z, params_dict_per_mode[mode_i][coeff_name], variances=True)
                mu = mu.at[coeff_i].set(mean_val)
                cov = cov.at[coeff_i, coeff_i].set(var_val)

                coeff_i += 1

        subkey_samples, self.random_key = jax.random.split(self.random_key)
        samples_coeff_values = jax.random.multivariate_normal(subkey_samples, mu, cov, shape=(n_samples,))

        samples = []
        for sample_i in range(n_samples):
            dict_values = []
            coeff_i = 0

            for mode_i, mode in enumerate(self.ansatz.modes):
                dict_values.append({})

                for coeff_name in mode.get_coeff_names():
                    value_name = coeff_name[:coeff_name.rfind('_')+1] + 'value'
                    dict_values[-1][value_name] = samples_coeff_values[sample_i, coeff_i]
                    coeff_i += 1

                for pod_name in mode.get_pod_names():
                    dict_values[-1][pod_name] = params_dict_per_mode[mode_i][pod_name]

            samples.append(self.ansatz.get_values_grid_exact(dict_values))

            ax.plot(self._data.x, samples[-1], linestyle='-', color='gray', linewidth=1.)

    def compare_exact(self, z, params_arr, sol_exact=None, ax = None, samples=True, legend=True, filename='img/compare.png'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 2))

        if samples:
            # plot some samples
            self._plot_samples(z, self.ansatz.arr_to_dict(params_arr), ax, n_samples=100)

        if sol_exact is None:
            sol_exact = self._data.get_solution(z)
        
        #try:
        #    x_local_mode_exact = self._data._center_init + self._data._get_gamma(z)
        #    ax.axvline(x_local_mode_exact, color='red')
        #except:
        #    pass


        sol_predicted = self.ansatz.get_values_grid(z, params_arr)
         
        cost_l2 = self.ansatz_cfg.calc_l2(z, params_arr, sol_exact)
        cost_w1, result_w1 = self.ansatz_cfg.calc_w1(z, params_arr, sol_exact, return_full=True)
        

        ax.plot(self._data.x, sol_predicted, label='Approx.', linestyle='-', linewidth=5.)
        ax.plot(self._data.x, sol_exact, label='Exact', linestyle='--', linewidth=5.)

        # TODO uncomment
        params_per_mode = self.ansatz.split_params_arr_by_mode(params_arr)
        for mode_i, mode in enumerate(self.ansatz.modes):
            values = mode.get_values_grid(z, params_per_mode[mode_i])
            ax.plot(self._data.x, values, label='%s' % mode.get_mode_name(), linestyle='--', linewidth=2.5)


        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_title((r'$z=%.2f;$' % z) + '\n' + ('$L_2=%.2f; W_1=%.2f$ (conv: %s)' % (cost_l2, cost_w1, result_w1.converged)),
                     fontdict = {'fontsize' : 20})

        ax.grid()

        if legend:
            ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close(fig)

    def plot_training(self, params_arr, samples=None, filename='img/training.png'):
        n_data = int(jnp.minimum(self._data.data_z.shape[0], 10))
        fig, ax = plt.subplots(1, n_data, figsize=(5*n_data, 5))
        
        data_idx = jnp.round(jnp.linspace(0, self._data.data_z.shape[0]-1, n_data)).astype(int)
        for ax_i, data_i in enumerate(data_idx):
            legend = False
            if ax_i == 0:
                legend = True
            self.compare_exact(self._data.data_z[data_i], params_arr, sol_exact=self._data.data_exact[data_i, :], samples=samples, ax=ax[ax_i], legend=legend)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_test(self, params_arr, test_z, test_data, samples=True, filename='img/test.png'):
        fig, ax = plt.subplots(1, test_z.shape[0], figsize=(5 * test_z.shape[0], 5))

        for i in range(test_z.shape[0]):
            legend = False
            if i == 0:
                legend = True
            self.compare_exact(test_z[i], params_arr, sol_exact=test_data[i, :], samples=samples, ax=ax[i],
                               legend=legend)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    # TODO untested
    def plot_hist(self, filename='img/hist.png'):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        hist_l2 = self._solver._log_l2
        hist_w1 = self._solver._log_w1
        hist_tot = self._solver._log_tot
        
        line_tot = ax.plot(hist_tot, label=r'$L_2 + \eta W_1$')

        line_l2 = ax.plot(hist_l2, label=r'$L_2$')
        
        ax_w1 = ax.twinx()
        line_w1 = ax_w1.plot(jnp.array(hist_tot) - jnp.array(hist_l2), label=r'$\eta W_1$', color='green')
        
        ax.set_ylabel(r'$L_2$')
        ax_w1.set_ylabel(r'$W_1$')
        
        ax.set_xlabel('iter')

        #ax_w1.legend(loc='upper left')
        lines = [line_tot[0], line_l2[0], line_w1[0]]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    # TODO untested
    def plot_every(self, directory='img/plot_every/'):
        assert(isinstance(self._solver, IteratedSolver))

        Path(directory).mkdir(parents=True, exist_ok=True)

        stage_i = 0

        # don't use different visualizers? does the visualzier depend on the solver
        # or on the anstaz?

        # all the visualizations, all the reconstructions we do here depend on the AnsatzData and not on the
        # OptimPhase, OptimConfig, never mind the optimizer. Hence, one visualizer will do.

        params_init_arr = self._solver.params_init
        self.plot_training(params_init_arr, filename=directory + 'training_0.png')
        # why were we plotting the same gradients for all of them?
        # gradients_per_param = self._solver.calc_gradients_per_param(params_init_arr)
        self.plot_pod_and_coeff(params_init_arr, filename=directory + 'pod_gamma_0.png')
        #visualizers[0].plot_OT_training(pars_init, filename=directory + 'OT_training_0.png')

        for repeat_i in range(self._solver.n_iter):
            for solver_i, solver in enumerate(self._solver.solvers):
                params_arr = self._solver.params_hist[stage_i]
                self.plot_training(params_arr, filename=directory + 'training_%d.png' % (stage_i+1))

                stage_i += 1

        stage_i = 0
        for repeat_i in range(self._solver.n_iter):
            for solver_i, solver in enumerate(self._solver.solvers):
                params_arr = self._solver.params_hist[stage_i]
                # why were we plotting the same gradients for all of them?
                # gradients_per_param = self._calc_gradients_per_param(params_arr)
                self.plot_pod_and_coeff(params_arr, filename=directory + 'pod_gamma_%d.png' % (stage_i + 1))

                stage_i += 1

    # TODO untested
    def plot_OT_local(self, mat, pod_local, beta_value, exact, ax):
        ax.imshow(mat[self._data.x.shape[0]:, :], cmap='Grays', origin='lower')
        part_local = beta_value*pod_local

        ax2 = ax.twinx()
        ax2 = ax2.twiny()

        ax2.plot(self._data.x, exact / exact.max(), color='gray', label='Data (full)')
        ax2.plot(self._data.x, mat.sum(axis=0) / exact.max(), label='Data (full (?))', linestyle='--')
        ax2.plot(self._data.x, mat[self._data.x.shape[0]:, :].sum(axis=0) / exact.max(), label='Data (transported)')
        ax2.plot(part_local / part_local.max(), self._data.x, label='Reconstructed (local)')
        

        ax2.set_xlim(self._data.x.min(), self._data.x.max())
        ax2.set_ylim(self._data.x.min(), self._data.x.max())

        ax.set_xlim(0., 100.)

        # Major ticks every 20, minor ticks every 5
        major_ticks = jnp.arange(0, 100, 10)
        minor_ticks = jnp.arange(0, 100, 5)
            
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax2.legend()

    # TODO untested
    def plot_global(mat, z, ax):
        ax.imshow(mat[:self._data.x.shape[0], :], cmap='Grays', origin='lower')
        _, part_global, part_local = solver_w1.get_prediction(z, pod_global, pod_local, alpha_coeff, beta, gamma_coeff, partial=True)

        ax2 = ax.twinx()
        ax2 = ax2.twiny()

        ax2.plot(self._data.x, exact)
        ax2.plot(part_global, self._data.x)

        ax2.set_xlim(self._data.x.min(), self._data.x.max())
        ax2.set_ylim(self._data.x.min(), self._data.x.max())

        ax.set_xlim(0., 100.)
        ax.set_ylim(0., 100.)

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 100, 10)
        minor_ticks = np.arange(0, 100, 5)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    # TODO untested
    def _plot_OT(self, z, sol_exact=None, pod_global=None, pod_local=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, ax=None):
        pod_globla, pod_local, alpha_coeff, beta_coeff, gamma_coeff = self._fill_none(pod_global, pod_local, alpha_coeff, beta_coeff, gamma_coeff)
        
        if sol_exact is None:
            sol_exact = self._data.get_solution(z)
        sol_predicted, global_part, local_part = self._solver.get_prediction(z, pod_global, pod_local, alpha_coeff, beta_coeff, gamma_coeff, partial=True)
        
        cost_w1, result_w1 = self._solver.calc_w_1(pod_global, pod_local, alpha_coeff, beta_coeff, gamma_coeff, z, sol_exact, return_full=True)
        mat = result_w1.matrix
        
        beta_value = self._solver._calc_beta(z, beta_coeff)
        self.plot_OT_local(mat, pod_local, beta_value, sol_exact, ax)

    # TODO untested
    def plot_OT_training(self, pod_global=None, pod_local=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, filename='img/OT_training.png'):
        fig, ax = plt.subplots(1, self._data.data_z.shape[0], figsize=(5*self._data.data_z.shape[0], 5))
        
        for i in range(self._data.data_z.shape[0]):
            self._plot_OT(self._data.data_z[i], self._data.data_exact[i], pod_global, pod_local, alpha_coeff, beta_coeff, gamma_coeff, ax=ax[i])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)
