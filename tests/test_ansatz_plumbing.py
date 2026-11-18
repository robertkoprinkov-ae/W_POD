"""Pure-plumbing unit tests for the ``Ansatz`` aggregation layer.

Covers group D of ``unit_tests.md``: an ``Ansatz`` is just a list of modes plus
the block-structured parameter vector that stitches them together
(``_delimiters``, ``split_params_arr_by_mode``, ``arr_to_dict`` / ``dict_to_arr``,
``get_bounds_arr``, ``get_constraints``).  Structural checks only -- the per-mode
behaviour is exercised by ``test_modes_plumbing.py``.

Every ``Ansatz`` here is built from modes whose ``n_param()`` values are pairwise
distinct, so a wrong delimiter or a mis-ordered block changes a length and fails.
"""

import h5py
import numpy as np
import jax.numpy as jnp
import pytest
import scipy.optimize

from W_POD.modes import GlobalMode, TransportedMode, StretchedMode
from W_POD.util import RBF_interpol
from W_POD.ansatz import Ansatz, OptimPhaseGradConfig, OptimPhaseCoeffConfig


_X = jnp.linspace(0.0, 1.0, 10)  # shared spatial mesh for all modes in a case


def _rbf(n_z, bias=False):
    return RBF_interpol(1.0, 1.0, jnp.linspace(0.0, 1.0, n_z), bias=bias)


def _global():
    return GlobalMode(_X, _rbf(4))                 # n_param = 10 + 4  = 14


def _global_big():
    return GlobalMode(_X, _rbf(9))                 # n_param = 10 + 9  = 19


def _transported():
    return TransportedMode(_X, _rbf(5), _rbf(6))   # n_param = 10 + 5 + 6 = 21


def _stretched():
    return StretchedMode(_X, _rbf(3), _rbf(4), _rbf(7))  # n_param = 10+3+4+7 = 24


_PARAM_G = {"alpha_coeff", "pod_global"}
_PARAM_T = {"beta_coeff", "gamma_coeff", "pod_local"}
_PARAM_S = {"beta_coeff", "gamma_shift_coeff", "gamma_stretch_coeff", "pod_local"}
_COEFF_G = {"alpha_coeff"}
_COEFF_T = {"beta_coeff", "gamma_coeff"}
_COEFF_S = {"beta_coeff", "gamma_shift_coeff", "gamma_stretch_coeff"}


class AnsatzCase:
    def __init__(self, ansatz, n_param, n_coeff, param_pytree, coeff_pytree):
        self.ansatz = ansatz
        self.n_param = n_param
        self.n_coeff = n_coeff
        self.param_pytree = param_pytree
        self.coeff_pytree = coeff_pytree


def _build_cases():
    return {
        "Global+Transported": AnsatzCase(
            Ansatz([_global(), _transported()]),
            n_param=14 + 21, n_coeff=1 + 2,
            param_pytree=[_PARAM_G, _PARAM_T],
            coeff_pytree=[_COEFF_G, _COEFF_T],
        ),
        "Transported+Global": AnsatzCase(  # reversed order
            Ansatz([_transported(), _global()]),
            n_param=21 + 14, n_coeff=2 + 1,
            param_pytree=[_PARAM_T, _PARAM_G],
            coeff_pytree=[_COEFF_T, _COEFF_G],
        ),
        "Global+Transported+Stretched": AnsatzCase(
            Ansatz([_global(), _transported(), _stretched()]),
            n_param=14 + 21 + 24, n_coeff=1 + 2 + 3,
            param_pytree=[_PARAM_G, _PARAM_T, _PARAM_S],
            coeff_pytree=[_COEFF_G, _COEFF_T, _COEFF_S],
        ),
        "single-Transported": AnsatzCase(
            Ansatz([_transported()]),
            n_param=21, n_coeff=2,
            param_pytree=[_PARAM_T],
            coeff_pytree=[_COEFF_T],
        ),
        "two-Global": AnsatzCase(  # repeated mode type
            Ansatz([_global(), _global_big()]),
            n_param=14 + 19, n_coeff=1 + 1,
            param_pytree=[_PARAM_G, _PARAM_G],
            coeff_pytree=[_COEFF_G, _COEFF_G],
        ),
    }


_CASES = _build_cases()


@pytest.fixture(params=list(_CASES), ids=list(_CASES))
def ans_case(request):
    return _CASES[request.param]


def _random_mode_dict(mode, rng):
    """A dict with the mode's parameter names and correct-length random values.
    Names/lengths are discovered via the (already tested) mode API."""
    ref = mode.arr_to_dict(jnp.arange(mode.n_param()))
    return {k: jnp.asarray(rng.standard_normal(int(v.shape[0])))
            for k, v in ref.items()}


# ---------------------------------------------------------------------------
# D17 -- n_param() / n_coeff() are sums over modes
# ---------------------------------------------------------------------------

def test_D17_n_param_and_n_coeff_are_sums(ans_case):
    ans = ans_case.ansatz
    assert ans.n_param() == sum(m.n_param() for m in ans.modes)
    assert ans.n_coeff() == sum(m.n_coeff() for m in ans.modes)
    # against the independently declared totals
    assert ans.n_param() == ans_case.n_param
    assert ans.n_coeff() == ans_case.n_coeff


# ---------------------------------------------------------------------------
# D18 -- _delimiters structure
# ---------------------------------------------------------------------------

def test_D18_delimiters(ans_case):
    ans = ans_case.ansatz
    d = ans._delimiters

    assert len(d) == len(ans.modes) + 1
    assert d[0] == 0
    assert d[-1] == ans.n_param()
    assert all(d[i] < d[i + 1] for i in range(len(d) - 1))  # strictly increasing
    for i, m in enumerate(ans.modes):
        assert d[i + 1] - d[i] == m.n_param()


# ---------------------------------------------------------------------------
# D19 -- split_params_arr_by_mode
# ---------------------------------------------------------------------------

def test_D19_split_params_arr_by_mode(ans_case):
    ans = ans_case.ansatz
    n = ans.n_param()
    pieces = ans.split_params_arr_by_mode(jnp.arange(n))

    assert len(pieces) == len(ans.modes)
    assert [int(p.shape[0]) for p in pieces] == [m.n_param() for m in ans.modes]

    np.testing.assert_array_equal(
        np.concatenate([np.asarray(p) for p in pieces]), np.arange(n)
    )

    offset = 0
    for m, p in zip(ans.modes, pieces):
        np.testing.assert_array_equal(
            np.asarray(p), np.arange(offset, offset + m.n_param())
        )
        offset += m.n_param()


# ---------------------------------------------------------------------------
# D20 -- arr_to_dict / dict_to_arr round-trips (Ansatz level)
# ---------------------------------------------------------------------------

def test_D20_roundtrip_array_dict_array(ans_case):
    ans = ans_case.ansatz
    arr = jnp.arange(ans.n_param())
    round_trip = ans.dict_to_arr(ans.arr_to_dict(arr))
    np.testing.assert_array_equal(np.asarray(round_trip), np.asarray(arr))


def test_D20_roundtrip_dict_array_dict(ans_case):
    ans = ans_case.ansatz
    rng = np.random.default_rng(20)
    dicts = [_random_mode_dict(m, rng) for m in ans.modes]

    arr = ans.dict_to_arr(dicts)
    assert arr.shape == (ans.n_param(),)

    dicts2 = ans.arr_to_dict(arr)
    assert len(dicts2) == len(dicts)
    for d, d2 in zip(dicts, dicts2):
        assert set(d2) == set(d)
        for k in d:
            np.testing.assert_array_equal(np.asarray(d2[k]), np.asarray(d[k]))


# ---------------------------------------------------------------------------
# D21 -- pytrees
# ---------------------------------------------------------------------------

def test_D21_param_and_coeff_pytrees(ans_case):
    ans = ans_case.ansatz

    assert ans.get_param_pytree() == [set(m.get_param_names()) for m in ans.modes]
    assert ans.get_coeff_pytree() == [set(m.get_coeff_names()) for m in ans.modes]

    # against the independently declared pytrees
    assert ans.get_param_pytree() == ans_case.param_pytree
    assert ans.get_coeff_pytree() == ans_case.coeff_pytree


# ---------------------------------------------------------------------------
# D22 -- get_bounds_arr
# ---------------------------------------------------------------------------

def test_D22_bounds_arr_is_per_mode_bounds_concatenated(ans_case):
    ans = ans_case.ansatz
    lb, ub = ans.get_bounds_arr()
    n = ans.n_param()

    assert lb.shape == (n,)
    assert ub.shape == (n,)

    exp_lb = np.concatenate(
        [np.asarray(m.get_param_bounds_arr()[0]) for m in ans.modes]
    )
    exp_ub = np.concatenate(
        [np.asarray(m.get_param_bounds_arr()[1]) for m in ans.modes]
    )
    np.testing.assert_array_equal(np.asarray(lb), exp_lb)
    np.testing.assert_array_equal(np.asarray(ub), exp_ub)
    assert np.all(np.asarray(lb) <= np.asarray(ub))


# ---------------------------------------------------------------------------
# D23 -- get_constraints assembly
# ---------------------------------------------------------------------------

def test_D23_constraints_assembly(ans_case):
    ans = ans_case.ansatz
    n = ans.n_param()

    lc = ans.get_constraints()
    assert isinstance(lc, scipy.optimize.LinearConstraint)

    A = np.asarray(lc.A)
    lb = np.broadcast_to(np.asarray(lc.lb), (A.shape[0],))
    ub = np.broadcast_to(np.asarray(lc.ub), (A.shape[0],))

    assert A.ndim == 2
    assert A.shape[1] == n

    per_mode = [m.get_constraints() for m in ans.modes]
    assert A.shape[0] == sum(Am.shape[0] for Am, _, _ in per_mode)
    assert lb.shape[0] == ub.shape[0] == A.shape[0]

    row = 0
    for i, (Am, lbm, ubm) in enumerate(per_mode):
        r = int(Am.shape[0])
        c0, c1 = ans._delimiters[i], ans._delimiters[i + 1]
        block = A[row:row + r, :]

        # the mode's own constraint matrix sits in its delimiter column range ...
        np.testing.assert_array_equal(block[:, c0:c1], np.asarray(Am))
        # ... and every other column is zero for these rows
        mask = np.ones(n, dtype=bool)
        mask[c0:c1] = False
        assert np.all(block[:, mask] == 0.0)

        np.testing.assert_array_equal(lb[row:row + r], np.asarray(lbm))
        np.testing.assert_array_equal(ub[row:row + r], np.asarray(ubm))
        row += r

    assert row == A.shape[0]


# ---------------------------------------------------------------------------
# D24 -- characterization: modes on different x meshes (see future.md sec 1)
# ---------------------------------------------------------------------------

def test_D24_mismatched_x_meshes_characterization():
    """Pins the current *unguarded* behaviour.  If a mesh-consistency guard is
    added to Ansatz.__init__, rewrite this as a pytest.raises on the
    constructor."""
    xa = jnp.linspace(0.0, 1.0, 10)
    xb = jnp.linspace(0.0, 1.0, 7)      # different length
    xc = jnp.linspace(0.0, 2.0, 10)     # same length, different span

    ma = GlobalMode(xa, _rbf(4))
    mb = GlobalMode(xb, _rbf(4))
    mc = GlobalMode(xc, _rbf(4))

    # (1) different-length meshes: constructor is silent ...
    ans = Ansatz([ma, mb])
    assert ans.n_param() == ma.n_param() + mb.n_param()
    assert ans._delimiters == [0, ma.n_param(), ma.n_param() + mb.n_param()]

    # ... the failure only surfaces later, inside get_values_grid
    with pytest.raises((TypeError, ValueError)):
        ans.get_values_grid(0.5, jnp.ones(ans.n_param()))

    # (2) same length, different span: no error anywhere -- silently wrong
    ans2 = Ansatz([ma, mc])
    out = ans2.get_values_grid(0.5, jnp.ones(ans2.n_param()))
    assert out.shape == (xa.shape[0],)


# ---------------------------------------------------------------------------
# Group E -- Ansatz value-output shapes (shape / plumbing only)
# ---------------------------------------------------------------------------
# The modes' value paths run division and interpolation, so a degenerate
# parameter vector (all zeros -> zero gamma_stretch -> 1/0) can blow up.  These
# helpers build a non-degenerate parameter vector / value dict: pod == 1,
# amplitude (alpha/beta) and stretch interpolants ~ 1, shift/gamma ~ 0.

_Z = 0.5  # scalar evaluation point


def _sane_mode_params(mode):
    ref = mode.arr_to_dict(jnp.zeros(mode.n_param()))
    rbfs = mode.get_RBF()
    out = {}
    for name in ref:
        if name in mode.get_pod_names():
            out[name] = jnp.ones(int(ref[name].shape[0]))
        else:
            rbf = rbfs[name]
            n_z = int(rbf.data_z.shape[0])
            near_one = ("stretch" in name) or ("beta" in name) or ("alpha" in name)
            out[name] = rbf.get_coeff(jnp.ones(n_z) if near_one else jnp.zeros(n_z))
    return mode.dict_to_arr(out)


def _sane_params_arr(ans):
    return jnp.concatenate([_sane_mode_params(m) for m in ans.modes])


def _sane_value_dict(mode):
    out = {name: jnp.ones(int(mode._x.shape[0])) for name in mode.get_pod_names()}
    for coeff_name in mode.get_coeff_names():
        value_name = coeff_name[:coeff_name.rfind("_")] + "_value"
        near_one = ("stretch" in coeff_name) or ("beta" in coeff_name) \
            or ("alpha" in coeff_name)
        out[value_name] = jnp.asarray(1.0 if near_one else 0.0)
    return out


def test_E25_get_values_grid_shape(ans_case):
    ans = ans_case.ansatz
    out = ans.get_values_grid(_Z, _sane_params_arr(ans))

    assert out.shape == (_X.shape[0],)
    assert np.all(np.isfinite(np.asarray(out)))  # sane params -> finite result


def test_E26_get_values_nodes_shapes_and_mode_order(ans_case):
    ans = ans_case.ansatz
    params = _sane_params_arr(ans)

    x_out, v_out = ans.get_values_nodes(_Z, params)

    total_nodes = sum(int(m._x.shape[0]) for m in ans.modes)
    assert x_out.shape == (total_nodes,)
    assert v_out.shape == x_out.shape
    assert x_out.shape == (len(ans.modes) * _X.shape[0],)  # shared mesh

    # the Ansatz concatenates the per-mode node clouds in mode order
    pieces = ans.split_params_arr_by_mode(params)
    exp_x = np.concatenate(
        [np.asarray(m.get_values_nodes(_Z, p)[0]) for m, p in zip(ans.modes, pieces)]
    )
    exp_v = np.concatenate(
        [np.asarray(m.get_values_nodes(_Z, p)[1]) for m, p in zip(ans.modes, pieces)]
    )
    np.testing.assert_array_equal(np.asarray(x_out), exp_x)
    np.testing.assert_array_equal(np.asarray(v_out), exp_v)


def test_E27_get_values_grid_exact_shape(ans_case):
    ans = ans_case.ansatz
    out = ans.get_values_grid_exact([_sane_value_dict(m) for m in ans.modes])

    assert out.shape == (_X.shape[0],)
    assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# Group F -- OptimPhaseGradConfig / OptimPhaseCoeffConfig validation
# ---------------------------------------------------------------------------
# Both classes assert in __init__ that
#   [mode_cfg.keys() for mode_cfg in cfg] == ansatz.get_param_pytree()   (grad)
#   [mode_cfg.keys() for mode_cfg in cfg] == ansatz.get_coeff_pytree()   (coeff)
# i.e. one dict per mode, keyed by exactly that mode's param / coeff names.
# NOTE: the check is a bare `assert`, so these pytest.raises(AssertionError)
# tests require pytest to run WITHOUT `python -O` (the default).


def _valid_grad_cfg(ans):
    return [{name: {"l2": True, "w1": True} for name in m.get_param_names()}
            for m in ans.modes]


def _valid_coeff_cfg(ans):
    return [{name: 1e-3 for name in m.get_coeff_names()} for m in ans.modes]


def _rename_one_key(mode_cfg):
    """Return a copy of mode_cfg with its first key misspelled."""
    first = next(iter(mode_cfg))
    return {(k + "_typo" if k == first else k): v for k, v in mode_cfg.items()}


def _drop_one_key(mode_cfg):
    first = next(iter(mode_cfg))
    return {k: v for k, v in mode_cfg.items() if k != first}


def _add_bogus_key(mode_cfg, value):
    return {**mode_cfg, "bogus_param": value}


# --- grad config ----------------------------------------------------------

def test_F28_grad_config_valid_construction(ans_case):
    ans = ans_case.ansatz
    cfg = _valid_grad_cfg(ans)

    obj = OptimPhaseGradConfig(ans, cfg)
    assert obj.ansatz is ans
    assert obj.grad_cfg is cfg


def test_F29_grad_config_wrong_mode_count(ans_case):
    ans = ans_case.ansatz
    cfg = _valid_grad_cfg(ans)

    with pytest.raises(AssertionError):
        OptimPhaseGradConfig(ans, cfg[:-1])          # too few
    with pytest.raises(AssertionError):
        OptimPhaseGradConfig(ans, cfg + [cfg[-1]])   # too many


def test_F30_grad_config_wrong_keys(ans_case):
    ans = ans_case.ansatz
    inner = {"l2": True, "w1": True}

    for corrupt in (_rename_one_key, _drop_one_key,
                    lambda mc: _add_bogus_key(mc, inner)):
        cfg = _valid_grad_cfg(ans)
        cfg[-1] = corrupt(cfg[-1])                   # corrupt the last mode
        with pytest.raises(AssertionError):
            OptimPhaseGradConfig(ans, cfg)


# --- coeff config -------------------------------------------------------

def test_F31_coeff_config_valid_construction(ans_case):
    ans = ans_case.ansatz
    cfg = _valid_coeff_cfg(ans)

    obj = OptimPhaseCoeffConfig(ans, cfg)
    assert obj.ansatz is ans
    assert obj.coeff_cfg is cfg


def test_F32_coeff_config_wrong_mode_count(ans_case):
    ans = ans_case.ansatz
    cfg = _valid_coeff_cfg(ans)

    with pytest.raises(AssertionError):
        OptimPhaseCoeffConfig(ans, cfg[:-1])
    with pytest.raises(AssertionError):
        OptimPhaseCoeffConfig(ans, cfg + [cfg[-1]])


def test_F33_coeff_config_wrong_keys(ans_case):
    ans = ans_case.ansatz

    for corrupt in (_rename_one_key, _drop_one_key,
                    lambda mc: _add_bogus_key(mc, 1e-3)):
        cfg = _valid_coeff_cfg(ans)
        cfg[-1] = corrupt(cfg[-1])
        with pytest.raises(AssertionError):
            OptimPhaseCoeffConfig(ans, cfg)


# ---------------------------------------------------------------------------
# Group J (ansatz half) -- hdf5 write / read round-trip
# ---------------------------------------------------------------------------

def test_J56_ansatz_write_hdf5_roundtrip(ans_case, tmp_path):
    ans = ans_case.ansatz
    arr = np.arange(ans.n_param(), dtype=np.float64)

    path = tmp_path / "ansatz.h5"
    with h5py.File(path, "w") as f:
        ans.write_hdf5(arr, f)

    with h5py.File(path, "r") as f:
        # a flat params_arr dataset ...
        assert "params_arr" in f
        np.testing.assert_array_equal(np.asarray(f["params_arr"][()]), arr)

        # ... plus exactly one group per mode, named mode_0 .. mode_{n-1}
        assert {k for k in f.keys() if k != "params_arr"} == \
            {"mode_%d" % i for i in range(len(ans.modes))}

        dicts = []
        for i, mode in enumerate(ans.modes):
            grp = f["mode_%d" % i]
            assert set(grp.keys()) == set(mode.get_param_names())
            dicts.append({name: jnp.asarray(grp[name][()]) for name in grp.keys()})

    reassembled = ans.dict_to_arr(dicts)
    np.testing.assert_array_equal(np.asarray(reassembled), arr)
