import jax.numpy as jnp
import jax

from W_POD.util import RBF_interpol
from W_POD import util

import matplotlib.pyplot as plt

import openturns as ot

import smt 

import numpy

import pytest

jax.config.update("jax_enable_x64", True)


class Test_RBF_interpol():
    
    def setup_method(self):
        self.data_z_1d = jnp.array([0.1, 1., 2.3])
        self.data_z_2d = jnp.array([[0.1, 0.5], [1., 1.], [1., 0.]])
        
        self.f_1d = lambda z: jnp.power(z, 2.) + jnp.cos(2*z)
        self.f_2d = lambda z: jnp.power(z[:, 0], 2.) + jnp.cos(2*z[:, 0]) + jnp.sin(2*z[:, 1])

        self.params_1d = {'theta': 0.5, 'sigma': 1.}
        self.params_2d = {'theta': 0.5, 'sigma': 1.}
        
        self.RBF_1d = RBF_interpol(self.params_1d['theta'], self.params_1d['sigma'], self.data_z_1d)
        self.RBF_2d = RBF_interpol(self.params_2d['theta'], self.params_2d['sigma'], self.data_z_2d)
    
        self.RBF_1d_bias = RBF_interpol(self.params_1d['theta'], self.params_1d['sigma'], self.data_z_1d, bias=True)
        self.RBF_2d_bias = RBF_interpol(self.params_2d['theta'], self.params_2d['sigma'], self.data_z_2d, bias=True)
    
    def rbf_1d(self, z_1, z_2):
        return jnp.power(self.params_1d['sigma'], 2.) * jnp.exp(-0.5 * jnp.power((z_1-z_2)/self.params_1d['theta'], 2.))
    
    def predict_1d(self, z, data_z, coeff):
        ret = 0.

        for i in range(data_z.shape[0]):
            ret += coeff[i] * self.rbf_1d(z, data_z[i])
        return ret
    
    def predict_2d(self, z, data_z, coeff):
        ret = 0.

        for i in range(data_z.shape[0]):
            ret += coeff[i] * self.rbf_2d(z, data_z[i])
        return ret

    def rbf_2d(self, z_1, z_2):
        assert(z_1.shape[0] == 2)
        assert(z_2.shape[0] == 2)

        return jnp.power(self.params_2d['sigma'], 2.) * jnp.exp(-0.5 * jnp.power(jnp.linalg.norm(z_1-z_2)/self.params_1d['theta'], 2.))
    
    def test_kernel(self):
        K_1d = self.RBF_1d._calc_K()
        
        K_ex = jnp.zeros((self.data_z_1d.shape[0], self.data_z_1d.shape[0]))
        for i in range(self.data_z_1d.shape[0]):
            for j in range(self.data_z_1d.shape[0]):
                K_ex = K_ex.at[i, j].set(self.rbf_1d(self.data_z_1d[i], self.data_z_1d[j]))

        assert pytest.approx(K_ex, abs=1e-9) == K_1d
        
        K_2d = self.RBF_2d._calc_K()
        
        K_ex = jnp.zeros((self.data_z_2d.shape[0], self.data_z_2d.shape[0]))
        for i in range(self.data_z_2d.shape[0]):
            for j in range(self.data_z_2d.shape[0]):
                K_ex = K_ex.at[i, j].set(self.rbf_2d(self.data_z_2d[i], self.data_z_2d[j]))

        assert pytest.approx(K_ex, abs=1e-9) == K_2d

    def test_get_coeff(self):
        # ensure coefficients actually interpolate data
        y_1d = self.f_1d(self.data_z_1d)
        coeff_1d = self.RBF_1d.get_coeff(y_1d)
        
        predictions = jnp.array([self.predict_1d(z, self.data_z_1d, coeff_1d) for z in self.data_z_1d])
        
        assert pytest.approx(y_1d, abs=1e-2) == predictions
        
        y_2d = self.f_2d(self.data_z_2d)
        coeff_2d = self.RBF_2d.get_coeff(y_2d)
        predictions = jnp.array([self.predict_2d(z, self.data_z_2d, coeff_2d) for z in self.data_z_2d])
        assert pytest.approx(y_2d, abs=1e-2, rel=1e-2) == predictions
        
        coeff_2d_bias = self.RBF_2d_bias.get_coeff(y_2d)
        predictions = coeff_2d_bias[-1] + jnp.array([self.predict_2d(z, self.data_z_2d, coeff_2d_bias[:-1]) for z in self.data_z_2d])
        assert pytest.approx(y_2d, abs=1e-2, rel=1e-2) == predictions

    def test__predict_z(self):
        pass 
    
    def test_predict_1d(self):
        data_z = jnp.linspace(0., 2.5, 4)
        y_train = self.f_1d(data_z)
        
        cov_model = ot.SquaredExponential(1)
        cov_model.activateAmplitude(True)
        cov_model.activateScale(True)
        print(type(y_train))
        fitter = ot.GaussianProcessFitter(ot.Sample(numpy.array(data_z.reshape((-1, 1)))), ot.Sample(numpy.array(y_train.reshape(-1, 1))), cov_model) 
        fitter.run()
        cov_result = fitter.getResult().getCovarianceModel()
        print(cov_result.getScale(), cov_result.getAmplitude())

        RBF_1d = RBF_interpol(cov_result.getScale()[0], cov_result.getAmplitude()[0], data_z)
        coeff = RBF_1d.get_coeff(y_train)

        z_test = jnp.linspace(0., 2.5, 1000)

        y_test = RBF_1d.predict(z_test[573], coeff)
        
        assert pytest.approx(self.f_1d(z_test[573]), abs=1e-2, rel=2e-1) == y_test
        
        y_test = RBF_1d.predict(z_test, coeff)
        y_exact = self.f_1d(z_test)

        assert pytest.approx(y_exact, abs=1., rel=2e-1) == y_test
        
        y_test_mean, y_test_var = RBF_1d.predict(z_test, coeff, variances=True)

        assert (y_test_mean == y_test).all()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(z_test, y_exact, label='exact')
        ax.scatter(data_z, self.f_1d(data_z), marker='x', label='training')
        
        ax.plot(z_test, y_test_mean, label='prediction')
        ax.plot(z_test, y_test_mean - 2*jnp.sqrt(y_test_var), linestyle='--', color='gray')
        ax.plot(z_test, y_test_mean + 2*jnp.sqrt(y_test_var), linestyle='--', color='gray')
        
        
        ax.legend()
        ax.grid()

        plt.savefig('img/util/plot_RBF_predict.png', dpi=300)
        
        assert (y_exact > y_test_mean - 2*jnp.sqrt(y_test_var)).all()
        assert (y_exact < y_test_mean + 2*jnp.sqrt(y_test_var)).all()

    def test_predict_2d(self):
        experiment = ot.LHSExperiment(ot.Normal(2), 50, False, True)
        data_z = jnp.array(experiment.generate())
        y_train = self.f_2d(data_z)
        
        cov_model = ot.SquaredExponential(2)
        cov_model.activateAmplitude(True)
        cov_model.activateScale(True)
        fitter = ot.GaussianProcessFitter(ot.Sample(numpy.array(data_z.reshape((-1, 2)))), ot.Sample(numpy.array(y_train.reshape(-1, 1))), cov_model) 
        fitter.run()
        cov_result = fitter.getResult().getCovarianceModel()
        print(cov_result.getScale(), cov_result.getAmplitude())

        RBF_2d = RBF_interpol(cov_result.getScale()[0], cov_result.getAmplitude()[0], data_z)
        coeff = RBF_2d.get_coeff(y_train)

        z_test = jnp.array(experiment.generate())

        y_test = RBF_2d.predict(z_test[5], coeff)
        
        #assert pytest.approx(self.f_2d(z_test[5:6])[0], abs=1e-2, rel=5e-2) == y_test
        
        y_test = RBF_2d.predict(z_test, coeff)
        y_exact = self.f_2d(z_test)

        #assert pytest.approx(y_exact, abs=3e-2, rel=5e-2) == y_test
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        x_grid = jnp.linspace(data_z[:, 0].min(), data_z[:, 0].max(), 100)
        y_grid = jnp.linspace(data_z[:, 1].min(), data_z[:, 1].max(), 100)
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i_i, i in enumerate(jnp.linspace(0, 99, 5).round().astype(int)):
            z_test = jnp.concatenate((x_grid[i]*jnp.ones((y_grid.shape[0], 1)), y_grid.reshape((-1, 1))), axis=1)
            ax[1].plot(y_grid, self.f_2d(z_test), color=colors[i_i], linestyle='--', label='exact')
            ax[1].plot(y_grid, RBF_2d.predict(z_test, coeff), color=colors[i_i], linestyle='-', label='predict')
        ax[1].set_xlabel(r'$x_2$')
        ax[1].set_title(r'$x_1$ fixed')
        ax[1].grid()
        ax[1].legend()

        for i_i, i in enumerate(jnp.linspace(0, 99, 5).round().astype(int)):
            z_test = jnp.concatenate((x_grid.reshape((-1, 1)), y_grid[i]*jnp.ones((x_grid.shape[0], 1))), axis=1)
            ax[0].plot(y_grid, self.f_2d(z_test), color=colors[i_i], linestyle='--', label=r'exact $x_2=%.2f$' % y_grid[i])
            ax[0].plot(y_grid, RBF_2d.predict(z_test, coeff), color=colors[i_i], linestyle='-', label='predict')
        ax[0].set_xlabel(r'$x_1$')
        ax[0].set_title(r'$x_2$ fixed')
        ax[0].grid()

        plt.savefig('img/util/plot_RBF_predict_2d.png', dpi=300)


def test_RBF_predict_shape():
    data_z = jnp.array([0.1, 1., 2.3])
    
    RBF = RBF_interpol(1., 1., data_z)
    coeff = jnp.ones_like(data_z)

    predict = jax.jit(lambda z: RBF.predict(z, coeff))
    predict_vars = jax.jit(lambda z: RBF.predict(z, coeff, variances=True))
    # at one point
    out = predict(0.5)
    out_var, var = predict_vars(0.5)

    assert out.shape == ()
    assert(var.shape == tuple())

    # at many points
    z_discret = jnp.linspace(0., 10., 100)
    out_vec = predict(z_discret)
    out_vec_var, var_vec = predict_vars(z_discret)

    assert out_vec.shape == z_discret.shape
    assert var_vec.shape == (z_discret.shape[0], )

def test_RBF_predict():
    data_z = jnp.array([0.1, 0.25, 0.7, 1., 1.5, 2.8])
    data_y = 1. + 2*jnp.cos(data_z)

    RBF = RBF_interpol(1., 1., data_z)
    coeff = RBF.get_coeff(data_y)

    assert coeff.shape == data_z.shape

    predict = jax.jit(lambda z: RBF.predict(z, coeff))

    z_test = jnp.linspace(data_z.min(), data_z.max(), 100)
    y_test = 1. + 2.*jnp.cos(z_test)
    y_pred = predict(z_test)

    assert pytest.approx(y_test, abs=0.1) == y_pred

def test_RBF_predict_vars():
    data_z = jnp.array([0.1, 0.25, 0.7, 1., 1.5, 2.8])
    data_y = 1. + 2 * jnp.cos(data_z)

    RBF = RBF_interpol(0.6, 1., data_z, nugget=1e-6)
    coeff = RBF.get_coeff(data_y)

    assert coeff.shape == data_z.shape

    predict = jax.jit(lambda z: RBF.predict(z, coeff))
    predict_vars = jax.jit(lambda z: RBF.predict(z, coeff, variances=True))

    z_test = jnp.linspace(data_z.min(), data_z.max(), 100)
    y_test = 1. + 2. * jnp.cos(z_test)
    y_pred = predict(z_test)
    y_pred_var, var_y_pred_var = predict_vars(z_test)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(z_test, y_pred)
    ax.plot(z_test, y_test, marker='x')
    ax.plot(z_test, y_pred_var - 2*jnp.sqrt(var_y_pred_var), linestyle='--', color='gray')
    ax.plot(z_test, y_pred_var + 2*jnp.sqrt(var_y_pred_var), linestyle='--', color='gray')
    plt.savefig('img/util/plot_RBF_predict.png', dpi=300)

    assert((y_pred == y_pred_var).all())
    assert pytest.approx(y_test, abs=0.15) == y_pred
    assert (var_y_pred_var > 0.).all()
    assert (y_pred_var - 2*jnp.sqrt(var_y_pred_var) < y_test).all()
    assert (y_pred_var + 2*jnp.sqrt(var_y_pred_var) > y_test).all()

    y_train_pred = predict(data_z)
    y_train_pred_var, var_y_train_pred_var = predict_vars(data_z)

    assert((y_train_pred == y_train_pred_var).all())
    assert pytest.approx(data_y, abs=0.1) == y_train_pred
    assert(pytest.approx(0., abs=1e-6)) == var_y_train_pred_var



def test_frankenstein_softplus():
    
    x = jnp.linspace(-10., 10., 101)
    x_neg = jnp.linspace(-10., 0., 100)
    x_pos = jnp.linspace(0., 10., 100)

    frankenstein_der = jax.vmap(jax.grad(util.frankenstein_softplus))
    der = frankenstein_der(x)
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].plot(x, util.frankenstein_softplus(x), label='frankenstein')
    ax[0].plot(x, x, label='y=x', linestyle='--')
    ax[1].plot(x_neg, frankenstein_der(x_neg))
    ax[2].plot(x_pos, frankenstein_der(x_pos))
    
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')

    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')

    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')

    ax[0].legend()
    
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    ax[0].set_title('Frankstein Softplus value')
    ax[1].set_title('Frankstein Softplus derivative')
    ax[2].set_title('Frankstein Softplus derivative')

    plt.tight_layout()
    plt.savefig('img/frankenstein_softplus.png', dpi=300)
    
    assert (der > 0).all()
    

    values_neg = util.frankenstein_softplus(x_neg)
    print(values_neg, frankenstein_der(x_neg))
    assert pytest.approx(values_neg, abs=0.1) == 0.

    values_pos = util.frankenstein_softplus(x_pos)

    assert pytest.approx(values_pos, abs=0.1) == x_pos

