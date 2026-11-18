"""Pure-plumbing unit tests for ``W_POD.util.DataCreation`` (group H of
``unit_tests.md``).

Two tiny 3-snapshot synthetic setups are built by hand:

* ``global+transported`` -- exercises coeff names ``alpha_coeff`` / ``beta_coeff``
  / ``gamma_coeff`` (the last with a bias RBF).
* ``global+stretched``   -- exercises ``alpha_coeff`` / ``beta_coeff`` /
  ``gamma_shift_coeff`` / ``gamma_stretch_coeff``.

Between them all five ``*_coeff -> *_value`` name manglings are exercised
end-to-end.  ``TransportedMode`` + ``StretchedMode`` are deliberately *not*
combined in one ansatz: both own a ``beta_coeff`` and ``DataCreation`` keys
``coeffs_exact`` / the ``RBF`` dict by bare coeff name, so they would collide.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from W_POD.modes import GlobalMode, TransportedMode, StretchedMode
from W_POD.util import RBF_interpol, DataCreation
from W_POD.ansatz import Ansatz


_X = jnp.linspace(0.0, 1.0, 12)
_DATA_Z = jnp.array([0.2, 0.6, 1.1])


def _bump(center, width):
    return jnp.exp(-0.5 * ((_X - center) / width) ** 2)


class DataCase:
    def __init__(self, dc, rbf):
        self.dc = dc
        self.rbf = rbf                       # {coeff_name: RBF_interpol}
        self.n_z = _DATA_Z.shape[0]
        self.x_len = _X.shape[0]


def _global_transported():
    rbf = {
        "alpha_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),   # n_param = n_z + 1
        "gamma_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
    }
    ansatz = Ansatz([
        GlobalMode(_X, rbf["alpha_coeff"]),
        TransportedMode(_X, rbf["beta_coeff"], rbf["gamma_coeff"]),
    ])
    values = [
        {"pod_global": lambda z: _bump(0.30, 0.20),
         "alpha_value": lambda z: 1.0 + 0.5 * jnp.sin(z)},
        {"pod_local": lambda z: _bump(0.50, 0.10),
         "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
         "gamma_value": lambda z: 0.15 * jnp.sin(z)},
    ]
    return DataCase(DataCreation(_X, ansatz, values, _DATA_Z, rbf), rbf)


def _global_stretched():
    rbf = {
        "alpha_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "beta_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "gamma_shift_coeff": RBF_interpol(0.5, 1.0, _DATA_Z),
        "gamma_stretch_coeff": RBF_interpol(0.5, 1.0, _DATA_Z, bias=True),
    }
    ansatz = Ansatz([
        GlobalMode(_X, rbf["alpha_coeff"]),
        StretchedMode(_X, rbf["beta_coeff"], rbf["gamma_shift_coeff"],
                      rbf["gamma_stretch_coeff"]),
    ])
    values = [
        {"pod_global": lambda z: _bump(0.30, 0.20),
         "alpha_value": lambda z: 1.0 + 0.5 * jnp.sin(z)},
        {"pod_local": lambda z: _bump(0.50, 0.10),
         "beta_value": lambda z: 1.2 + 0.3 * jnp.cos(z),
         "gamma_shift_value": lambda z: 0.05 * jnp.sin(z),
         "gamma_stretch_value": lambda z: 1.0 + 0.1 * jnp.sin(z)},   # never ~0
    ]
    return DataCase(DataCreation(_X, ansatz, values, _DATA_Z, rbf), rbf)


_CASES = {"global+transported": _global_transported(),
          "global+stretched": _global_stretched()}


@pytest.fixture(params=list(_CASES), ids=list(_CASES))
def data_case(request):
    return _CASES[request.param]


# ---------------------------------------------------------------------------
# H48 -- coeff-name -> value-name mangling
# ---------------------------------------------------------------------------

def test_H48_coeff_name_to_value_name_mangling():
    """Replicates the inline expression in DataCreation.__init__
    (``coeff_name[:coeff_name.rfind('_')] + '_value'``) and pins all five cases.
    An incorrect mangling would make DataCreation.__init__ raise KeyError on
    ``ansatz_values[mode_i][variable_name]`` -- see the end-to-end fixture."""
    def mangle(coeff_name):
        return coeff_name[:coeff_name.rfind("_")] + "_value"

    assert mangle("alpha_coeff") == "alpha_value"
    assert mangle("beta_coeff") == "beta_value"
    assert mangle("gamma_coeff") == "gamma_value"
    assert mangle("gamma_shift_coeff") == "gamma_shift_value"
    assert mangle("gamma_stretch_coeff") == "gamma_stretch_value"


# ---------------------------------------------------------------------------
# H49 -- data_exact shape, dx
# ---------------------------------------------------------------------------

def test_H49_data_exact_shape_and_dx(data_case):
    dc = data_case.dc
    assert dc.data_exact.shape == (data_case.n_z, data_case.x_len)
    assert float(dc.dx) == float(_X[1] - _X[0])


# ---------------------------------------------------------------------------
# H50 -- coeffs_exact structure
# ---------------------------------------------------------------------------

def test_H50_coeffs_exact_structure(data_case):
    dc = data_case.dc
    coeff_names = [cn for m in dc.ansatz.modes for cn in m.get_coeff_names()]

    assert set(dc.coeffs_exact.keys()) == set(coeff_names)
    for cn in coeff_names:
        assert dc.coeffs_exact[cn].shape == (data_case.rbf[cn].n_param(),)


# ---------------------------------------------------------------------------
# H51 -- get_solution shape (and consistency with the stored rows)
# ---------------------------------------------------------------------------

def test_H51_get_solution_shape(data_case):
    dc = data_case.dc

    for z in (0.1, 0.6, 1.3):
        assert dc.get_solution(z).shape == (data_case.x_len,)

    # _generate_data() stacks get_solution(data_z[i]) row by row
    for i, z in enumerate(np.asarray(dc.data_z)):
        np.testing.assert_allclose(
            np.asarray(dc.get_solution(float(z))),
            np.asarray(dc.data_exact[i]),
            rtol=1e-5, atol=1e-6,
        )
