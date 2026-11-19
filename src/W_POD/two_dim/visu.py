import matplotlib.pyplot as plt
import jax.numpy as jnp

class Visualizer():
    def __init__(self, pts, x_crds, y_crds, data_z, data_exact, ansatz):

        self.pts = pts

        xx, yy = jnp.meshgrid(x_crds, y_crds)

        self.xx = xx
        self.yy = yy

        self.data_z = data_z
        self.data_exact = data_exact
        self.ansatz = ansatz

        self.dx = x_crds[1] - x_crds[0]

    def plot_data(self, data, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.pcolormesh(self.xx, self.yy, data.reshape((self.xx.shape[0], self.xx.shape[1])))
        return ax


    def plot_recon(self, params, filename_prefix, losses):
        fig, ax = plt.subplots(1, self.data_z.shape[0], figsize=(5. * self.data_z.shape[0], 5.))

        params_by_name = self.ansatz.split_params(params)

        losses_l2 = losses.loss_l2(params, full=True)

        losses_l2_rel = losses_l2 / losses.get_l2_scaling_factor()

        losses_w2 = losses.loss_w2(params, downscaled_size=losses.DOWNSCALED_SIZE, full=True)

        for i in range(self.data_z.shape[0]):
            self.plot_data(self.ansatz.get_value_grid_arbitrary(params, self.data_z[i]), ax[i])

            params_by_name_cf = self.ansatz.coordinate_frames[0].split_params(params_by_name['coord_frame_0'])
            ax[i].scatter(*self.ansatz.coordinate_frames[0].coordinate_transformation.get_shift_center(self.pts, self.data_z[i],
                                                                                                  params_by_name_cf[
                                                                                                      'coordinate_transformation']),
                          marker='x', color='red')
            ax[i].set_title('z=%.2f; l2_rel=%.2f; w2=%.2f' % (self.data_z[i], losses_l2_rel[i], losses_w2[i]))
        plt.colorbar(ax[-1].collections[-1])
        plt.savefig(filename_prefix + '_recon.png', dpi=300)

        self.ansatz.coordinate_frames[0].plot_coeff(self, params_by_name['coord_frame_0'], filename_prefix + '_frame0')
        self.ansatz.coordinate_frames[1].plot_coeff(self, params_by_name['coord_frame_1'], filename_prefix + '_frame1')

    def plot_recon_2d(self, params, filename_prefix, losses, n_temp, n_time):
        assert(self.data_z.shape[0] == n_time * n_temp)
        fig, ax = plt.subplots(n_temp, n_time, figsize=(5. * n_time, 5. * n_temp))

        params_by_name = self.ansatz.split_params(params)

        losses_l2 = losses.loss_l2(params, full=True)

        losses_l2_rel = losses_l2 / losses.get_l2_scaling_factor()

        losses_w2 = losses.loss_w2(params, downscaled_size=losses.DOWNSCALED_SIZE, full=True)

        for z_i in range(n_temp):
            for z_j in range(n_time):
                i = z_i * n_time + z_j
                self.plot_data(self.ansatz.get_value_grid_arbitrary(params, self.data_z[i]), ax[z_i, z_j])
                ax[z_i, z_j].set_title('z=%s; l2_rel=%.2f; w2=%.2f' % (str(self.data_z[i]), losses_l2_rel[i], losses_w2[i]))

            plt.colorbar(ax[z_i, -1].collections[-1])
        plt.savefig(filename_prefix + '_recon.png', dpi=300)

