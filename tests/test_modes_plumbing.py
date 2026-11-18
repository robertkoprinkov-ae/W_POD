"""Pure-plumbing unit tests for the mode classes.

Covers groups A (parameter-array <-> dict plumbing), B (metadata consistency),
C (bounds & constraint structure) and the per-mode half of J (hdf5 round-trip)
of ``unit_tests.md``.  Structural / mechanical correctness only -- no numerical or
algorithmic oracles.  Every case is built from trivial hand-made inputs.

The RBF interpolators inside each mode are deliberately given *different* numbers
of ``data_z`` points (and, in one case, a bias term) so that every block of the
parameter vector has a distinct length.  A mis-aligned slice in ``arr_to_dict`` /
``dict_to_arr`` therefore changes a length and is guaranteed to fail rather than
slip through unnoticed.
"""

import h5py
import numpy as np
import jax.numpy as jnp
import pytest

from W_POD.modes import GlobalMode, TransportedMode, StretchedMode
from W_POD.util import RBF_interpol


def _rbf(n_z, bias=False):
    return RBF_interpol(1.0, 1.0, jnp.linspace(0.0, 1.0, n_z), bias=bias)


class ModeCase:
    """A mode instance plus an *independently specified* description of it."""

    def __init__(self, mode, x_len, layout, mode_name, coeff_names, pod_names,
                 n_param_sol, constrained_coeff):
        self.mode = mode
        self.x_len = x_len
        # (name, length) for each block, in dict_to_arr / parameter-array order.
        self.layout = layout
        self.mode_name = mode_name
        self.coeff_names = coeff_names
        self.pod_names = pod_names
        self.n_param_sol = n_param_sol
        # coefficient whose RBF get_constraints() places a linear-constraint
        # block for (the amplitude coefficient: alpha for GlobalMode, beta else).
        self.constrained_coeff = constrained_coeff


def _build_cases():
    x_len = 11
    x = jnp.linspace(0.0, 1.0, x_len)
    cases = {}

    cases["GlobalMode"] = ModeCase(
        mode=GlobalMode(x, _rbf(5)),
        x_len=x_len,
        layout=[("pod_global", x_len), ("alpha_coeff", 5)],
        mode_name="GlobalMode",
        coeff_names=["alpha_coeff"],
        pod_names=["pod_global"],
        n_param_sol=x_len + 1,
        constrained_coeff="alpha_coeff",
    )
    cases["TransportedMode"] = ModeCase(
        mode=TransportedMode(x, _rbf(5), _rbf(7)),
        x_len=x_len,
        layout=[("pod_local", x_len), ("beta_coeff", 5), ("gamma_coeff", 7)],
        mode_name="TransportedMode",
        coeff_names=["beta_coeff", "gamma_coeff"],
        pod_names=["pod_local"],
        n_param_sol=x_len + 2,
        constrained_coeff="beta_coeff",
    )
    # RBFs with a bias term: n_param() = data_z + 1, so the blocks no longer
    # match the data_z sizes -- exercises the offset arithmetic harder.
    cases["TransportedMode-bias"] = ModeCase(
        mode=TransportedMode(x, _rbf(5, bias=True), _rbf(7, bias=True)),
        x_len=x_len,
        layout=[("pod_local", x_len), ("beta_coeff", 6), ("gamma_coeff", 8)],
        mode_name="TransportedMode",
        coeff_names=["beta_coeff", "gamma_coeff"],
        pod_names=["pod_local"],
        n_param_sol=x_len + 2,
        constrained_coeff="beta_coeff",
    )
    cases["StretchedMode"] = ModeCase(
        mode=StretchedMode(x, _rbf(4), _rbf(6), _rbf(3)),
        x_len=x_len,
        layout=[
            ("pod_local", x_len),
            ("beta_coeff", 4),
            ("gamma_shift_coeff", 6),
            ("gamma_stretch_coeff", 3),
        ],
        mode_name="StretchedMode",
        coeff_names=["beta_coeff", "gamma_shift_coeff", "gamma_stretch_coeff"],
        pod_names=["pod_local"],
        n_param_sol=x_len + 3,
        constrained_coeff="beta_coeff",
    )
    return cases


_CASES = _build_cases()


@pytest.fixture(params=list(_CASES), ids=list(_CASES))
def case(request):
    return _CASES[request.param]


def _random_dict(layout, seed):
    rng = np.random.default_rng(seed)
    return {name: jnp.asarray(rng.standard_normal(length))
            for name, length in layout}


# ---------------------------------------------------------------------------
# Group A -- parameter-array <-> dict plumbing
# ---------------------------------------------------------------------------

def test_A1_arr_to_dict_partitions_exactly(case):
    """A1: n_param() equals the summed block lengths and arr_to_dict partitions
    the array with no gap and no overlap."""
    n = case.mode.n_param()
    assert n == sum(length for _, length in case.layout)

    d = case.mode.arr_to_dict(jnp.arange(n))
    assert sum(int(v.shape[0]) for v in d.values()) == n

    pooled = np.sort(np.concatenate([np.asarray(v) for v in d.values()]))
    np.testing.assert_array_equal(pooled, np.arange(n))


def test_A2_roundtrip_array_dict_array(case):
    """A2: dict_to_arr(arr_to_dict(v)) == v for v = arange(n_param())."""
    arr = jnp.arange(case.mode.n_param())
    round_trip = case.mode.dict_to_arr(case.mode.arr_to_dict(arr))
    np.testing.assert_array_equal(np.asarray(round_trip), np.asarray(arr))


def test_A3_roundtrip_dict_array_dict(case):
    """A3: arr_to_dict(dict_to_arr(d)) == d for a dict with correct-length
    entries."""
    d = _random_dict(case.layout, seed=3)
    d2 = case.mode.arr_to_dict(case.mode.dict_to_arr(d))

    assert set(d2.keys()) == set(d.keys())
    for name in d:
        np.testing.assert_array_equal(np.asarray(d2[name]), np.asarray(d[name]))


def test_A4_arr_to_dict_uses_correct_offsets(case):
    """A4: each named slice sits at the documented offset in the parameter
    array."""
    n = case.mode.n_param()
    d = case.mode.arr_to_dict(jnp.arange(n))

    offset = 0
    for name, length in case.layout:
        np.testing.assert_array_equal(
            np.asarray(d[name]), np.arange(offset, offset + length)
        )
        offset += length
    assert offset == n


def test_A5_dict_to_arr_key_order_independent(case):
    """A5: dict_to_arr depends on key *values*, not on dict insertion order --
    in particular it is unaffected by get_param_names() ordering (which for
    GlobalMode is not the array order).  This guards AnsatzConfig.cut_grad."""
    d = _random_dict(case.layout, seed=5)

    canonical = np.asarray(case.mode.dict_to_arr(d))

    # rebuilt in get_param_names() order (the order cut_grad iterates in)
    d_names_order = {name: d[name] for name in case.mode.get_param_names()}
    np.testing.assert_array_equal(
        np.asarray(case.mode.dict_to_arr(d_names_order)), canonical
    )

    # rebuilt in reverse layout order
    d_reversed = {name: d[name] for name, _ in reversed(case.layout)}
    np.testing.assert_array_equal(
        np.asarray(case.mode.dict_to_arr(d_reversed)), canonical
    )

    # and the canonical result is exactly the layout-order concatenation
    expected = np.concatenate([np.asarray(d[name]) for name, _ in case.layout])
    np.testing.assert_array_equal(canonical, expected)


# ---------------------------------------------------------------------------
# Group B -- mode metadata consistency
# ---------------------------------------------------------------------------

def test_B6_param_names_partition_into_coeff_and_pod(case):
    """B6: get_param_names() is exactly the disjoint union of get_coeff_names()
    and get_pod_names(), with no duplicates."""
    pnames = case.mode.get_param_names()
    cnames = case.mode.get_coeff_names()
    podnames = case.mode.get_pod_names()

    assert len(pnames) == len(set(pnames))
    assert set(pnames) == set(cnames) | set(podnames)
    assert set(cnames).isdisjoint(set(podnames))


def test_B7_n_coeff_and_n_pod_match_name_lists(case):
    """B7: n_coeff() / n_pod() agree with the length of the name lists."""
    assert case.mode.n_coeff() == len(case.mode.get_coeff_names())
    assert case.mode.n_pod() == len(case.mode.get_pod_names())


def test_B8_get_RBF_keys_match_coeff_names(case):
    """B8: get_RBF() is keyed exactly by the coefficient names."""
    assert set(case.mode.get_RBF().keys()) == set(case.mode.get_coeff_names())


def test_B9_n_param_is_mesh_plus_rbf_params(case):
    """B9: n_param() == len(x) + sum of each coefficient RBF's n_param()."""
    rbf_total = sum(rbf.n_param() for rbf in case.mode.get_RBF().values())
    assert case.mode.n_param() == case.x_len + rbf_total


def test_B10_n_param_sol(case):
    """B10: n_param_sol() == len(x) + (1 / 2 / 3), i.e. one scalar value per
    coefficient trajectory."""
    assert case.mode.n_param_sol() == case.n_param_sol
    assert case.mode.n_param_sol() == case.x_len + len(case.mode.get_coeff_names())


def test_B11_mode_name(case):
    """B11: get_mode_name() returns the expected literal."""
    assert case.mode.get_mode_name() == case.mode_name


# ---------------------------------------------------------------------------
# Group C -- bounds & constraint-matrix structure
# ---------------------------------------------------------------------------

def _blocks(layout):
    """Yield (name, start, stop) for each parameter block, in array order."""
    offset = 0
    for name, length in layout:
        yield name, offset, offset + length
        offset += length


def test_C12_param_bounds_shapes_and_ordering(case):
    """C12: get_param_bounds_arr() returns two (n_param(),) arrays, NaN-free,
    with lb <= ub elementwise."""
    lb, ub = case.mode.get_param_bounds_arr()
    n = case.mode.n_param()

    assert lb.shape == (n,)
    assert ub.shape == (n,)

    lb_np, ub_np = np.asarray(lb), np.asarray(ub)
    assert not np.isnan(lb_np).any()
    assert not np.isnan(ub_np).any()
    assert np.all(lb_np <= ub_np)


def test_C13_param_bounds_values_per_block(case):
    """C13: the pod block has lb == 1e-6 and ub == +inf; every coefficient
    block has lb == -inf and ub == +inf."""
    lb, ub = case.mode.get_param_bounds_arr()
    lb_np, ub_np = np.asarray(lb), np.asarray(ub)

    pod = set(case.pod_names)
    coeff = set(case.coeff_names)

    for name, start, stop in _blocks(case.layout):
        lb_block, ub_block = lb_np[start:stop], ub_np[start:stop]
        if name in pod:
            assert np.allclose(lb_block, 1e-6), name
            assert np.isposinf(ub_block).all(), name
        elif name in coeff:
            assert np.isneginf(lb_block).all(), name
            assert np.isposinf(ub_block).all(), name
        else:  # pragma: no cover - guarded by test_B6
            raise AssertionError("block %r is neither pod nor coeff" % name)


def test_C14_constraints_shapes(case):
    """C14: get_constraints() returns (A, lb, ub) with
    A.shape == (lb.shape[0], n_param()), ub.shape == lb.shape, and -- for these
    non-monotonic modes -- lb.shape[0] == n_z.  lb <= ub, NaN-free."""
    A, lb, ub = case.mode.get_constraints()
    n = case.mode.n_param()

    assert A.ndim == 2
    assert lb.ndim == 1 and ub.ndim == 1
    assert A.shape == (lb.shape[0], n)
    assert ub.shape == lb.shape
    assert lb.shape[0] == case.mode.n_z

    lb_np, ub_np = np.asarray(lb), np.asarray(ub)
    assert not np.isnan(lb_np).any()
    assert not np.isnan(ub_np).any()
    assert np.all(lb_np <= ub_np)


def test_C16_constraint_block_placement(case):
    """C16: the amplitude RBF's linear-constraint matrix sits at column offset
    len(x); every other column of A is zero."""
    A, _, _ = case.mode.get_constraints()
    A_np = np.asarray(A)

    rbf = case.mode.get_RBF()[case.constrained_coeff]
    offset = case.x_len
    width = rbf.n_param()

    np.testing.assert_array_equal(
        A_np[:, offset:offset + width],
        np.asarray(rbf.get_linear_constraint_matrix()),
    )

    mask = np.ones(A_np.shape[1], dtype=bool)
    mask[offset:offset + width] = False
    assert np.all(A_np[:, mask] == 0.0)


# --- C15: monotonic-gamma variant of TransportedMode ------------------------
# The monotonic path sets an (n_z - 1, gamma_n_param) block, so it only works
# when the beta and gamma RBFs have the same number of data_z points.  _rbf()
# is deterministic, so passing the same size twice gives two identical RBFs.

def _transported_equal_rbf(n_pts, **kwargs):
    x = jnp.linspace(0.0, 1.0, 9)
    return TransportedMode(x, _rbf(n_pts), _rbf(n_pts), **kwargs)


def test_C15_monotonic_gamma_row_counts_and_bounds():
    """C15: gamma_increasing / gamma_decreasing each append exactly n_z - 1
    rows; both flags together raise AssertionError; neither keeps the base
    n_z rows.  The appended bounds encode the sign of the successive gamma
    differences."""
    n = 6

    base = _transported_equal_rbf(n)
    A0, lb0, _ = base.get_constraints()
    assert A0.shape[0] == n
    assert lb0.shape[0] == n

    inc = _transported_equal_rbf(n, gamma_increasing=True)
    A1, lb1, ub1 = inc.get_constraints()
    assert A1.shape == (n + (n - 1), inc.n_param())
    assert lb1.shape[0] == ub1.shape[0] == n + (n - 1)
    np.testing.assert_array_equal(np.asarray(lb1[n:]), np.zeros(n - 1))
    assert np.isposinf(np.asarray(ub1[n:])).all()

    dec = _transported_equal_rbf(n, gamma_decreasing=True)
    A2, lb2, ub2 = dec.get_constraints()
    assert A2.shape == (n + (n - 1), dec.n_param())
    np.testing.assert_array_equal(np.asarray(ub2[n:]), np.zeros(n - 1))
    assert np.isneginf(np.asarray(lb2[n:])).all()

    both = _transported_equal_rbf(n, gamma_increasing=True, gamma_decreasing=True)
    with pytest.raises(AssertionError):
        both.get_constraints()


def test_C15_monotonic_gamma_block_placement():
    """C15/C16: the appended rows carry the gamma RBF's monotonic-constraint
    matrix in its own (trailing) column range and zeros before it; the first
    n_z rows are byte-for-byte the non-monotonic matrix."""
    n = 6
    base = _transported_equal_rbf(n)
    inc = _transported_equal_rbf(n, gamma_increasing=True)

    A_base = np.asarray(base.get_constraints()[0])
    A_inc = np.asarray(inc.get_constraints()[0])

    np.testing.assert_array_equal(A_inc[:n, :], A_base)

    gamma_rbf = inc._RBF_gamma
    gw = gamma_rbf.n_param()
    np.testing.assert_array_equal(
        A_inc[n:, -gw:],
        np.asarray(gamma_rbf.get_monotonic_linear_constraints_matrix()),
    )
    assert np.all(A_inc[n:, :-gw] == 0.0)


# ---------------------------------------------------------------------------
# Group J (per-mode half) -- hdf5 write / read round-trip
# ---------------------------------------------------------------------------

def test_J55_mode_write_hdf5_roundtrip(case, tmp_path):
    """write_hdf5 stores one dataset per parameter name; reading them back and
    reassembling via dict_to_arr reproduces the original array."""
    mode = case.mode
    arr = np.arange(mode.n_param(), dtype=np.float64)

    path = tmp_path / "mode.h5"
    with h5py.File(path, "w") as f:
        mode.write_hdf5(arr, f)

    with h5py.File(path, "r") as f:
        assert set(f.keys()) == set(mode.get_param_names())
        for name, length in case.layout:            # each dataset the right size
            assert f[name].shape == (length,)
        d = {name: jnp.asarray(f[name][()]) for name in f.keys()}

    reassembled = mode.dict_to_arr(d)
    np.testing.assert_array_equal(np.asarray(reassembled), arr)
