import jax
import jax.numpy as jnp
from ott.solvers import linear
from ott.geometry import pointcloud
import numpy
import scipy

import time

import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src_wildfire.hess_diag_probed_2d import calc_hessian_diag_2d

class Losses():
    def __init__(self, pts, dx, data_z, data_exact, ansatz, l2_or_l1=1, DOWNSCALED_SIZE=None, reg_lambda=0.):
        self.pts = pts
        self.dx = dx
        self.ansatz = ansatz
        self.data_z = data_z
        self.data_exact = data_exact
        self.bounds = ansatz.get_bounds()

        self.l2_or_l1 = l2_or_l1
        self.DOWNSCALED_SIZE = DOWNSCALED_SIZE
        self.reg_lambda = reg_lambda

        self.H, self.W = data_exact.shape[-2], data_exact.shape[-1]

        self.data_exact_flat = data_exact.reshape((-1, self.H * self.W))

        self.n_coord_frames = len(self.ansatz.coordinate_frames)

        self.w2_vmapped = jax.vmap(self.w2_per_z_vmappable, in_axes=(0, None, None, None))
        
        self._cost_sum_jitted = jax.jit(self._cost_sum)
        self._cost_sum_grad_jitted = jax.jit(jax.grad(self._cost_sum))

    def _downscale_array(self, arr, factor=2):
        """
        Downscale an array of 2D numpy arrays by averaging non-overlapping blocks.

        Parameters:
        arr : numpy.ndarray (N_arr, H, W)
        factor : int (block size, e.g., 2 for 64x64 -> 32x32)

        Returns:
        numpy.ndarray (N_arr, H//factor, W//factor)
        """
        _, h, w = arr.shape

        # Ensure dimensions are divisible by the factor
        if h % factor != 0 or w % factor != 0:
            raise ValueError("Array dimensions must be divisible by the downscale factor.")

        # Reshape to separate the blocks, then average over the block axes
        # Shape becomes: (h//factor, factor, w//factor, factor)
        # Mean over axes 1 and 3 (the factor dimensions)
        downsampled = arr.reshape(arr.shape[0], h // factor, factor, w // factor, factor).mean(axis=(2, 4))

        return downsampled

    @jax.jit(static_argnums=(0,), static_argnames=('downscaled_size'))
    def loss_w2_z(self, params, z_i, data_exact_flat, downscaled_size=None):
        pts_2, vals_2 = self.ansatz.get_value_nodes(params, z_i)

        a = jnp.abs(data_exact_flat[z_i, :]) + 1e-6  # TODO
        b = jnp.maximum(vals_2, 0.) + 1e-6

        a_norm = a / a.sum()
        b_norm = b / b.sum()

        result_w2 = None
        if downscaled_size is not None:
            factor = int(self.H // downscaled_size)
            # [modification 3_modes]
            a_norm_small = self._downscale_array(a_norm.reshape((1, self.H, self.W)), factor=factor).flatten()
            b_norm_small = self._downscale_array(b_norm.reshape((self.n_coord_frames, self.H, self.W)), factor=factor).flatten()

            a_norm_small = a_norm_small / a_norm_small.sum()
            b_norm_small = b_norm_small / b_norm_small.sum()

            pts_square_a = self.pts.reshape((1, self.H, self.W, 2))
            x_small_a = self._downscale_array(pts_square_a[:, :, :, 0], factor=factor).reshape((-1, 1))
            y_small_a = self._downscale_array(pts_square_a[:, :, :, 1], factor=factor).reshape((-1, 1))
            pts_small_a = jnp.concatenate((x_small_a, y_small_a), axis=1)

            # [modification 3_modes]
            pts_square_b = pts_2.reshape((self.n_coord_frames, self.H, self.W, 2))
            x_small_b = self._downscale_array(pts_square_b[:, :, :, 0], factor=factor).reshape((-1, 1))
            y_small_b = self._downscale_array(pts_square_b[:, :, :, 1], factor=factor).reshape((-1, 1))
            pts_small_b = jnp.concatenate((x_small_b, y_small_b), axis=1)

            geom = pointcloud.PointCloud(
                pts_small_a,
                pts_small_b,
                cost_fn=None,
                epsilon=.1
            )

            result_w2 = linear.solve(geom, a=a_norm_small, b=b_norm_small, threshold=0.001, max_iterations=100000, use_danskin=False)

        else:
            geom = pointcloud.PointCloud(
                self.pts,
                pts_2,
                cost_fn=None,
                epsilon=.1
            )

            result_w2 = linear.solve(geom, a=a_norm, b=b_norm, threshold=0.01, max_iterations=100000)
        return result_w2

    # this rewritten function needs some work when it comes to naming the variables
    def w2_per_z_vmappable(self, z_i, params, data_exact_flat, downscaled_size):
        c = self.loss_w2_z(params, z_i, data_exact_flat, downscaled_size=downscaled_size).reg_ot_cost
        return jnp.power(jnp.where(jnp.isfinite(c) & (c >= 0.), c, 1e10), 1)  # a diverged Sinkhorn can give a negative / -inf / NaN entropic cost; 1e10 so L-BFGS-B backtracks

    def _loss_w2(self, params, data_exact_flat, downscaled_size=None, full=False):
        cost_hist = self.w2_vmapped(jnp.arange(self.data_z.shape[0]), params, data_exact_flat, downscaled_size)
        cost_tot = jnp.sum(cost_hist)

        if full:
            return cost_hist
        else:
            return cost_tot
    
    def loss_w2(self, params, downscaled_size=None, full=False):
        return self._loss_w2(params, self.data_exact_flat, downscaled_size, full)

    def compute_l2_per_z_vmappable(self, z_i, params, data_exact_flat_row):
        """Compute cost for a single z_i slice."""
        values_reconstruction = self.ansatz.get_value_grid(params, z_i)
        hat_values = data_exact_flat_row - values_reconstruction
        cost = jnp.sum(jnp.power(hat_values, 2.)) * self.dx * self.dx  # Assuming dx is scalar
        return cost

    def _loss_l2(self, params, data_exact_flat, full=False):
        cost_hist = jax.lax.map(lambda args: self.compute_l2_per_z_vmappable(args[0], params, args[1]), (jnp.arange(self.data_z.shape[0]), data_exact_flat), batch_size=50)
        cost_tot = jnp.sum(jnp.power(cost_hist, self.l2_or_l1))
        if full:
            return jnp.array(cost_hist)
        else:
            return cost_tot
    
    def loss_l2(self, params, full=False):
        return self._loss_l2(params, self.data_exact_flat, full)

    def get_l2_scaling_factor(self):
        if self.l2_or_l1 == 1:
            return jnp.sum(jnp.power(self.data_exact, 2.)) * self.dx * self.dx
        elif self.l2_or_l1 == 2:
            return jnp.sum(jnp.power(jnp.sum(jnp.power(self.data_exact, 2.), axis=1) * self.dx * self.dx, 2.))


    def _cost_sum(self, params, norm_l2, norm_w1, data_exact_flat, full=False):
        l2 = self._loss_l2(params, data_exact_flat, full=full)
        w2 = self._loss_w2(params, data_exact_flat, full=full, downscaled_size=self.DOWNSCALED_SIZE)

        if full:
            return l2, l2/norm_l2, w2, w2/norm_w1
        else:
            return w2 / norm_w1 + l2 / norm_l2 + self.reg_lambda * self.ansatz.get_reg(params)
    
    def cost_sum(self, params, norm_l2, norm_w1, full=False):
        return self._cost_sum_jitted(params, norm_l2, norm_w1, self.data_exact_flat)
    
    def cost_sum_grad(self, params, norm_l2, norm_w1, full=False):
        return self._cost_sum_grad_jitted(params, norm_l2, norm_w1, self.data_exact_flat)

class Losses_scaled():
    def __init__(self, losses: Losses, params_init: jax.typing.ArrayLike, data_exact: jax.typing.ArrayLike, scaling='gradient'):
        self.ansatz = losses.ansatz
        self.losses = losses
        
        n_y, n_x = data_exact.shape[-2], data_exact.shape[-1]
        self.data_exact_flat = data_exact.reshape((-1, n_y * n_x))
        
        self.scaling = scaling

        self.n_iter = 0

        self.hist = []
        self.hist_grad_modes = []
        self.hist_grad_remainder = []

        self.hist_params_rescaling = []
        self.iter_rescaling = []
        
        self.reset_scaling(params_init)

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
        return (out[0] if isinstance(argnums, int) else tuple(out)), diag.sum() 

    def reset_scaling(self, params):
        self.hist_params_rescaling.append(params)
        self.iter_rescaling.append(self.n_iter)

        self.norm_l2 = self.losses.loss_l2(params)
        self.norm_w1 = self.losses.loss_w2(params, downscaled_size=self.losses.DOWNSCALED_SIZE)


        if self.scaling == 'identity':
            self.scaling_x_modes = 1.
            self.scaling_x_remainder = 1.
        elif self.scaling == 'gradient':
            grad_params = self.losses.cost_sum_grad(params, self.norm_l2, self.norm_w1)

            modes_grad_norm = jnp.linalg.norm(self.mask_modal_variables(grad_params))
            remainder_grad_norm = jnp.linalg.norm(self.mask_remainder(grad_params))

            # want both gradients to be equally significant
            scaling_grad_modes = 1./modes_grad_norm
            scaling_grad_remainder = 1./remainder_grad_norm

            # z = 5x => f'(z) = f(5x) = f'(x)*5
            # to get gradient to change by scaling_gr_mode, must multiply respective x coord by scaling_gr_mode
            
            self.scaling_x_modes = scaling_grad_modes
            self.scaling_x_remainder = scaling_grad_remainder
        elif self.scaling == 'hessian':
            
            hess = calc_hessian_diag_2d(self.losses._cost_sum, params, self.norm_l2, self.norm_w1, self.losses.data_exact_flat, ansatz=self.ansatz, argnums=0, downscale_factor=20, floor=False)
            
            print('Hessian: min: ', hess.min(), ' max: ', hess.max())
            
            scaling = 1./jnp.sqrt(jnp.maximum(jnp.abs(hess), 1e-6 * jnp.max(jnp.abs(hess))))
            
            normalization_geometric_mean = jnp.exp(jnp.mean(jnp.log(scaling)))
            print('Hessian scaling (pre-normalization): min: ', scaling.min(), ' max: ', scaling.max())
            scaling /= normalization_geometric_mean
            print('Hessian scaling (post-normalization): min: ', scaling.min(), ' max: ', scaling.max())
            
            self.scaling_x = scaling
        else:
            raise NameError()

        self._cost_grad = jax.jit(jax.grad(self._cost_graddable))
        self._cost = jax.jit(self._cost_graddable)

    def mask_remainder(self, x):
        xs = self.ansatz.split_params_recursive(x)

        for coord_frame in xs.keys():
            for keyname in xs[coord_frame].keys():
                if keyname[:4] == 'mode':
                    xs[coord_frame][keyname]['mode'] = 0.
        return self.ansatz.get_params_recursive(xs)

    def mask_modal_variables(self, x):
        return x - self.mask_remainder(x)

    def convert_x(self, x):
        if self.scaling == 'hessian':
            return x * self.scaling_x
        else:
            mode_copy = self.mask_modal_variables(x)

            x_scaled = self.scaling_x_remainder * self.mask_remainder(x)
            x_scaled = x_scaled + mode_copy * self.scaling_x_modes

            return x_scaled

    def convert_x_inv(self, x):
        if self.scaling == 'hessian':
            return x / self.scaling_x
        else:
            mode_copy = self.mask_modal_variables(x)

            x_unscaled = self.mask_remainder(x) / self.scaling_x_remainder
            x_unscaled = x_unscaled + mode_copy / self.scaling_x_modes

            return x_unscaled
    
    def _cost_graddable(self, x, data_exact_flat): # TODO does this update? 
        x_scaled = self.convert_x(x)

        return self.losses._cost_sum(x_scaled, self.norm_l2, self.norm_w1, data_exact_flat)
    
    def _cost_graddable_certainly_working(self, x, norm_l2, norm_w1, data_exact_flat): # TODO does this update? 
        x_scaled = self.convert_x(x)

        return self.losses._cost_sum(x_scaled, norm_l2, norm_w1, data_exact_flat)
        
    def cost_grad(self, x):
        return self._cost_grad(x, self.data_exact_flat)
    
    def cost_grad_np(self, x):
        g = numpy.asarray(self.cost_grad(jnp.array(x)))
        if not numpy.all(numpy.isfinite(g)): # non-finite gradient -> log crash in solution + zeros so the L-BFGS-B line search backtracks instead of stopping
            print("W2PATCH: non-finite gradient (%d of %d entries) -> zeros" % (int((~numpy.isfinite(g)).sum()), g.size), flush=True)
            g = numpy.zeros_like(g)
        return g
    
    def cost(self, x):
        return self._cost(x, self.data_exact_flat)

    def cost_np(self, x):
        f = numpy.asarray(self.cost(jnp.array(x)))
        if not numpy.isfinite(f) or f >= 1e6:  # W2PATCH (2026-11-12): log when the per-snapshot W2 guard (1e10, divided by the W2 normalisation) fired
            print("W2PATCH: guarded cost %.3e" % float(f), flush=True)
        return f

    def optimize(self, params_init, maxiter, debug=False):
        self.reset_scaling(params_init)

        def callback(intermediate_result):
            params = self.convert_x(intermediate_result.x)
            l2 = self.losses.loss_l2(params)
            # need sqrt: otherwise get frobenius squared instead of frobenius
            l2_rel = jnp.sqrt(l2 / self.losses.get_l2_scaling_factor())
            reg = self.losses.reg_lambda * self.ansatz.get_reg(params)

            print('Iter %d; loss: %.4f: ; frobenius/l2: %.4f; reg: %.4f; w2: %.4f; rel frobenius/l2' % (self.n_iter,
                                                                                                        intermediate_result.fun,
                                                                                                        l2 * self.norm_l2,
                                                                                                        reg,
                                                                                                        (intermediate_result.fun - l2 / self.norm_l2 - reg) * self.norm_w1),
                  l2_rel)
            self.n_iter += 1
            
            self.hist.append(intermediate_result.fun)

            grad_params = self.cost_grad_np(intermediate_result.x)
            
            modes_grad_norm = jnp.linalg.norm(self.mask_modal_variables(grad_params))
            remainder_grad_norm = jnp.linalg.norm(self.mask_remainder(grad_params))
            
            self.hist_grad_modes.append(modes_grad_norm)
            self.hist_grad_remainder.append(remainder_grad_norm)

        time_start = time.time()
        
        bounds_unscaled = self.losses.bounds
        lb_s = numpy.where(numpy.isfinite(bounds_unscaled.lb), numpy.asarray(self.convert_x_inv(jnp.asarray(bounds_unscaled.lb))), bounds_unscaled.lb)
        ub_s = numpy.where(numpy.isfinite(bounds_unscaled.ub), numpy.asarray(self.convert_x_inv(jnp.asarray(bounds_unscaled.ub))), bounds_unscaled.ub)
        bounds_scaled = scipy.optimize.Bounds(lb_s, ub_s)
        
        res = scipy.optimize.minimize(self.cost_np, self.convert_x_inv(params_init), method='L-BFGS-B', jac=self.cost_grad_np, bounds=bounds_scaled,
                options={'maxiter': maxiter, 'maxcor': 50}, callback=callback) # [0609] modified!
        time_end = time.time()

        if res.nit == 0:
            print('Converged instantly')
        else:
            print('Time per iter: %.2fs' % ((time_end - time_start)/res.nit))
        print(res.message)

        if debug:
            return self.convert_x(res.x), res
        else:
            return self.convert_x(res.x)
