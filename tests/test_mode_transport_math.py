"""Math-tier tests for mode-level transport math (group N in unit_tests.md).

This file currently covers N1, N2 and N9. N3-N8/N10 (amplitude scaling,
exact arithmetic in get_values_nodes, self-consistency, etc.) are still
pending.

N1 -- TransportedMode.get_values_grid shift, exact (see docstring on
test_N1_transported_shift_matches_analytical_gaussian below).

N2 -- StretchedMode.get_values_grid stretch, exact (see docstring on
test_N2_stretched_matches_analytical_gaussian below).

N9 -- get_values_grid()'s gradient w.r.t. the full parameter vector
(pod_local, beta_coeff, and gamma_coeff / gamma_shift_coeff +
gamma_stretch_coeff), for TransportedMode and StretchedMode, against central
finite differences. This exercises interpax.interp1d's (catmull-rom) gradient
composed with the coordinate transform and the RBF coefficient gradients -- a
code path nothing had checked before this.

Two different precision policies live in this file, and which one applies to
a given test was decided by MEASURING, not by a blanket rule:

  - N9 (gradient vs. finite differences) genuinely needs fp64: at fp32,
    central-difference noise floors around 1e-3 to 1e-4, which could mask a
    real gradient bug of similar size. Every N9 check therefore exists twice --
    a "_smoke" version (fp32-safe tolerance, always runs, so the default suite
    keeps *some* gradient coverage) and a precision-labelled version
    (`@pytest.mark.skipif(not jax.config.jax_enable_x64, ...)`, tight
    tolerance). Calibrated:
      - fp64: h swept 1e-4 down to 3e-7; error dropped ~100x per 10x in h
        (clean quadratic central-difference convergence) until bottoming out
        near h=1e-6 (max|diff|~6e-10), then roundoff took back over below
        that. h=1e-5 is a conservative step short of that optimum; worst
        measured max|diff| across both modes, z in {0.6, 0.8} was 2.15e-8 --
        atol=rtol=1e-6 keeps >45x headroom.
      - fp32: h=1e-3 was the sweet spot (unlike test_rbf_accuracy.py's
        pure-RBF case, where h=1e-2 was optimal -- this composed system
        accumulates roundoff faster, so the optimal step is smaller); worst
        measured max|diff| was ~2e-3-2.7e-3. atol=rtol=1e-2 keeps several x
        headroom while staying safe against per-run random-seed variability.
  - N1 (analytical shift check) does NOT need fp64, checked rather than
    assumed by symmetry with N9: measured max error at fp32 was 1.5445396e-6,
    at fp64 1.5445408e-6 -- identical to ~7 significant figures. That means
    the residual is a genuine (tiny) property of the catmull-rom
    interpolation itself, not floating-point roundoff, so fp64 buys nothing
    here. N1 is therefore a single test with one tolerance, run at whatever
    precision is active (verified to pass under both).
  - N2 (analytical stretch check) -- same conclusion, checked the same way:
    fp32 measured 7.033e-6 (pure stretch) / 6.914e-6 (shift+stretch); fp64
    measured 6.958e-6 for both. Same order of magnitude, not the ~1e5x gap
    N9's genuinely roundoff-limited case showed -- this residual is off-node
    cubic-interpolation truncation error (unlike N1, the stretch transform
    generally does not land query points exactly on mesh nodes), not
    roundoff. Single test, one tolerance, no fp64 gate.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import interpax

from W_POD.modes import TransportedMode, StretchedMode, GlobalMode
from W_POD.util import RBF_interpol, frankenstein_softplus


_REQUIRE_X64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="precision-labelled gradient check -- run with JAX_ENABLE_X64=1 for "
           "the tight (atol=1e-6) tolerance; see the _smoke sibling test for "
           "the always-on, fp32-safe version of the same check",
)

_REQUIRE_X64_TEST_ONLY = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="test-only numerics (np.interp resampling the codebase itself "
           "never does) -- fp64-only by scoping choice, not because fp32 is "
           "imprecise here (confirmed fp32 gives the same numbers); run with "
           "JAX_ENABLE_X64=1",
)


# ---------------------------------------------------------------------------
# N1 -- TransportedMode.get_values_grid shift, exact
# ---------------------------------------------------------------------------

# Mesh spacing 1/100 = 0.01, so a shift of exactly 0.5 (an integer multiple of
# the spacing) maps every mesh node exactly onto another mesh node -- this is
# what makes the residual below (1.5e-6) a property of catmull-rom itself
# rather than a "how close can two arbitrary points be" approximation error.
_X_N1 = jnp.linspace(0.0, 1.0, 101)
_POD_LOCAL_N1 = jnp.exp(-0.5 * jnp.power((_X_N1 - 0.5) / 0.1, 2.0))
_DATA_Z_N1 = jnp.array([0.2, 0.6, 1.0])

# Measured max error 1.5445396e-6 (fp32) / 1.5445408e-6 (fp64) -- agree to ~7
# significant figures, i.e. NOT roundoff (see module docstring). One
# tolerance, safe at either precision.
_N1_ATOL = 1e-5


def test_N1_transported_shift_matches_analytical_gaussian():
    """A shift of gamma=0.5 applied to a Gaussian bump centered at 0.5 (width
    0.1) should produce a Gaussian centered at 1.0 (same width) -- checked
    against the closed form, not just "shape looks plausible".

    beta/gamma are pinned EXACTLY via RBF_interpol(bias=True).get_coeff() on a
    constant target: for a constant target the residual after de-meaning is
    the exact zero vector, so the non-bias coefficients solve to exactly zero
    and predict(z, coeff) == the constant, for ANY z, with zero RBF-fitting
    noise (see unit_tests.md group N's "Exact-coefficient trick"). That
    isolates this test to "is the coordinate-transform + interpolation logic
    right", independent of RBF accuracy (already covered by groups G/M)."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(_DATA_Z_N1.shape[0]))       # beta == 1 exactly
    gamma_coeff = RBF_gamma.get_coeff(0.5 * jnp.ones(_DATA_Z_N1.shape[0]))  # gamma == 0.5 exactly
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    # z is arbitrary (off-node on purpose) since beta/gamma are pinned constant
    result = mode.get_values_grid(0.37, params)
    analytical = jnp.exp(-0.5 * jnp.power((_X_N1 - 1.0) / 0.1, 2.0))

    np.testing.assert_allclose(np.asarray(result), np.asarray(analytical), atol=_N1_ATOL)


# ---------------------------------------------------------------------------
# N3 -- amplitude scaling combined with transport
# ---------------------------------------------------------------------------

# Same setup as N1, but beta=2 instead of 1 -- catches a bug that gets the
# shift geometry right but drops or misapplies the amplitude multiply (e.g.
# multiplying before instead of after interpolation would still pass N1,
# since 1*x == x, but would show up here as soon as beta != 1).
# Measured max error 3.0890792e-6 (fp32) / 3.0890818e-6 (fp64) -- agrees to
# ~6 sig figs (same conclusion as N1: not roundoff), and is exactly 2x N1's
# 1.5445e-6, consistent with beta multiplying a fixed interpolation residual.
# atol=1e-5, matching N1 (~3.2x headroom over the measured worst case;
# deterministic fixture, no randomness to buffer against).
_N3_ATOL = 1e-5


def test_N3_transported_amplitude_scales_the_shifted_gaussian():
    """Same shift as N1 (gamma=0.5, bump at 0.5 width 0.1 -> bump at 1.0), but
    with beta=2 pinned instead of 1: the whole result should scale by exactly
    beta, not just get the location/width right."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(2.0 * jnp.ones(_DATA_Z_N1.shape[0]))    # beta == 2 exactly
    gamma_coeff = RBF_gamma.get_coeff(0.5 * jnp.ones(_DATA_Z_N1.shape[0]))  # gamma == 0.5 exactly
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    result = mode.get_values_grid(0.37, params)
    analytical = 2.0 * jnp.exp(-0.5 * jnp.power((_X_N1 - 1.0) / 0.1, 2.0))

    np.testing.assert_allclose(np.asarray(result), np.asarray(analytical), atol=_N3_ATOL)
    # the peak itself must be exactly beta (frankenstein_softplus is the
    # identity above its 0.01 cutoff, so this should hold to full precision)
    np.testing.assert_allclose(float(jnp.max(result)), 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# N2 -- StretchedMode.get_values_grid stretch, exact
# ---------------------------------------------------------------------------

# Same mesh and pod_local as N1, so `pod_local` is centered exactly on
# StretchedMode's scaling_axis (= (x.min()+x.max())/2 = 0.5) -- that's what
# makes "pure stretch doesn't move the center" hold in the first case below.
_X_N2 = _X_N1
_POD_LOCAL_N2 = _POD_LOCAL_N1
_DATA_Z_N2 = _DATA_Z_N1

# Measured max error: 7.033e-6 (pure stretch) / 6.914e-6 (shift+stretch) at
# fp32; 6.958e-6 (both) at fp64 -- same order of magnitude at both precisions
# (see module docstring), i.e. genuine off-node interpolation error, not
# roundoff. The whole fixture is deterministic (no randomness anywhere), so
# there is no run-to-run variability to buffer against -- atol=1e-5 (~1.4x
# headroom over the measured worst case, tightened from 1e-4 -> 3e-5 -> 2e-5
# -> 1e-5) leaves margin only for minor cross-platform/library-version
# floating-point differences. Deliberately tight; verified passing at both
# fp32 and fp64 at this value.
_N2_ATOL = 1e-5

_N2_CASES = [
    (0.0, 2.0, "pure_stretch"),   # center unchanged (pod_local sits on the axis), width x2
    (0.5, 2.0, "shift_and_stretch"),  # center +0.5, width x2
]


@pytest.mark.parametrize("shift_val,stretch_val,case_id", _N2_CASES,
                         ids=[c[2] for c in _N2_CASES])
def test_N2_stretched_matches_analytical_gaussian(shift_val, stretch_val, case_id):
    """StretchedMode's coordinate transform is
    x -> gamma_shift + axis + gamma_stretch*(x-axis). Applied to a Gaussian
    bump centered at 0.5 (width 0.1, exactly on the scaling axis), it should
    produce a Gaussian centered at 0.5+gamma_shift with width 0.1*gamma_stretch
    -- checked against the closed form. beta/gamma_shift/gamma_stretch are
    pinned exactly via the same bias=True + constant-target trick as N1 (see
    its docstring); unlike N1, the query points here generally do not land
    exactly on mesh nodes, so a small (~1e-5, see module docstring) genuine
    interpolation residual is expected and tolerated."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_shift = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_stretch = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    n_z = _DATA_Z_N2.shape[0]
    beta_coeff = RBF_beta.get_coeff(jnp.ones(n_z))
    shift_coeff = RBF_shift.get_coeff(shift_val * jnp.ones(n_z))
    stretch_coeff = RBF_stretch.get_coeff(stretch_val * jnp.ones(n_z))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N2, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    result = mode.get_values_grid(0.37, params)
    center = 0.5 + shift_val
    width = 0.1 * stretch_val
    analytical = jnp.exp(-0.5 * jnp.power((_X_N2 - center) / width, 2.0))

    np.testing.assert_allclose(np.asarray(result), np.asarray(analytical), atol=_N2_ATOL)


# ---------------------------------------------------------------------------
# N4 -- get_values_nodes is exact arithmetic, no interpolation at all
# ---------------------------------------------------------------------------

# get_values_nodes is the point-cloud representation used for the W2 term.
# Unlike get_values_grid, it never calls interpax -- it's literally a `+`/`*`
# away from the coefficients, so there is no interpolation error to tolerate
# at all. Confirmed (not assumed) bit-for-bit exact at both fp32 and fp64:
# jnp.all(x_out == expected) and jnp.all(v_out == expected) both True.
# Non-constant, non-round coefficients (beta=2.0/1.5, gamma=0.3,
# shift=0.2, stretch=1.7) are used deliberately -- with beta=1/gamma=0 a bug
# that drops a term entirely could still look right by accident.

def test_N4_transported_get_values_nodes_is_exact_arithmetic():
    """TransportedMode.get_values_nodes(z, params) == (x + gamma, beta*pod_local),
    exactly -- no tolerance, this is pure arithmetic."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(2.0 * jnp.ones(_DATA_Z_N1.shape[0]))
    gamma_coeff = RBF_gamma.get_coeff(0.3 * jnp.ones(_DATA_Z_N1.shape[0]))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    x_out, v_out = mode.get_values_nodes(0.37, params)
    expected_x = _X_N1 + 0.3
    expected_v = 2.0 * _POD_LOCAL_N1

    np.testing.assert_array_equal(np.asarray(x_out), np.asarray(expected_x))
    np.testing.assert_array_equal(np.asarray(v_out), np.asarray(expected_v))


def test_N4_stretched_get_values_nodes_is_exact_arithmetic():
    """StretchedMode.get_values_nodes(z, params) ==
    (gamma_shift + axis + gamma_stretch*(x-axis), beta*pod_local), exactly."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_shift = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_stretch = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    n_z = _DATA_Z_N2.shape[0]
    beta_coeff = RBF_beta.get_coeff(1.5 * jnp.ones(n_z))
    shift_coeff = RBF_shift.get_coeff(0.2 * jnp.ones(n_z))
    stretch_coeff = RBF_stretch.get_coeff(1.7 * jnp.ones(n_z))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N2, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    x_out, v_out = mode.get_values_nodes(0.37, params)
    axis = mode.scaling_axis
    expected_x = 0.2 + axis + 1.7 * (_X_N2 - axis)
    expected_v = 1.5 * _POD_LOCAL_N2

    np.testing.assert_array_equal(np.asarray(x_out), np.asarray(expected_x))
    np.testing.assert_array_equal(np.asarray(v_out), np.asarray(expected_v))


# --- N4 extended: cross-validate the point cloud against the N1/N2 analytical
# ground truth, not just against a hand-derived formula ------------------------
#
# The two tests above check get_values_nodes' raw arithmetic against a formula
# *I* typed -- that's an internal-consistency check, not an independent one: a
# sign-convention mismatch between get_values_grid and get_values_nodes could
# in principle slip past it if the same (wrong) assumption were baked into
# both the code and my formula. This closes that gap: resample the
# (x_out, v_out) point cloud back onto the original mesh with np.interp (a
# routine the codebase itself never calls -- independent of both interpax and
# jnp.interp) and compare against the SAME closed-form analytical Gaussian
# N1/N2 already validated `get_values_grid` against.
#
# Uses interpax.interp1d(method='catmull-rom') -- the SAME resampling scheme
# `get_values_grid` itself uses (mirroring its padding trick: extend the
# point cloud by one padding node on each side with value 0 before calling
# interp1d, so extrap=0. has enough neighbours to work with). This is a
# slightly weaker independence guarantee than the original np.interp version
# (a bug in interpax's catmull-rom implementation could in principle affect
# both this check and get_values_grid identically) -- but get_values_nodes
# itself contains no interpolation at all, so interpax here is not testing
# itself, only resampling its point cloud for comparison, and cubic
# convergence removes the dominant source of tolerance-widening error that
# linear np.interp had on the stretched (2x coarsened) point cloud.
#
# fp64-only, no fp32 smoke companion: this resampling step is pure test
# machinery, not something the production code ever does, so there is no
# scenario where it needs to be verified at fp32.
#
# Measured (fp64): Transported (gamma=0.5, same case as N1) 1.545e-6;
# Stretched (shift=0.5, stretch=2, same case as N2's second case) 6.958e-6 --
# both close to N1/N2's own catmull-rom-based errors, ~180x tighter than the
# old linear-np.interp Stretched number (1.2445e-3), confirming that number
# was dominated by np.interp's O(h^2) truncation error on the 2x-coarsened
# point cloud, not by get_values_nodes itself. Confirmed fp32 gives the same
# numbers (1.545e-6 / 6.914e-6) -- not roundoff, consistent with N1/N2's
# conclusion -- but gated to fp64 anyway per the "we don't do this in
# practice" scoping decision above, not a precision one. Negative controls
# (fp64): wrong shift value gives max err 0.559; a simulated sign-flipped
# stretch convention gives max err 1.0 -- both far above the tightened
# tolerances below.

def _resample_via_interpax(x_out, v_out, query):
    """Resample a (x_out, v_out) point cloud onto `query`, mirroring
    get_values_grid's own padding+catmull-rom pattern (see modes.py)."""
    x_out = jnp.asarray(x_out)
    v_out = jnp.asarray(v_out)
    padding = 0.1 * (x_out.max() - x_out.min())
    coords_extended = jnp.concatenate((jnp.array([x_out.min() - padding]), x_out,
                                       jnp.array([x_out.max() + padding])))
    values_extended = jnp.concatenate((jnp.array([0.]), v_out, jnp.array([0.])))
    return np.asarray(interpax.interp1d(query, coords_extended, values_extended,
                                        method='catmull-rom', extrap=0.))


@_REQUIRE_X64_TEST_ONLY
def test_N4_transported_get_values_nodes_resampled_matches_analytical():
    """Resample TransportedMode.get_values_nodes' point cloud (gamma=0.5, same
    case as N1) back onto the mesh with interpax and check it against the
    same analytically-shifted Gaussian N1 validated get_values_grid against."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(_DATA_Z_N1.shape[0]))
    gamma_coeff = RBF_gamma.get_coeff(0.5 * jnp.ones(_DATA_Z_N1.shape[0]))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    x_out, v_out = mode.get_values_nodes(0.37, params)
    resampled = _resample_via_interpax(x_out, v_out, _X_N1)
    analytical = np.asarray(jnp.exp(-0.5 * jnp.power((_X_N1 - 1.0) / 0.1, 2.0)))

    np.testing.assert_allclose(resampled, analytical, atol=5e-6)


@_REQUIRE_X64_TEST_ONLY
def test_N4_stretched_get_values_nodes_resampled_matches_analytical():
    """Same cross-check for StretchedMode (shift=0.5, stretch=2, same case as
    N2's second case)."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_shift = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_stretch = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    n_z = _DATA_Z_N2.shape[0]
    beta_coeff = RBF_beta.get_coeff(jnp.ones(n_z))
    shift_coeff = RBF_shift.get_coeff(0.5 * jnp.ones(n_z))
    stretch_coeff = RBF_stretch.get_coeff(2.0 * jnp.ones(n_z))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N2, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    x_out, v_out = mode.get_values_nodes(0.37, params)
    resampled = _resample_via_interpax(x_out, v_out, _X_N2)
    analytical = np.asarray(jnp.exp(-0.5 * jnp.power((_X_N2 - 1.0) / 0.2, 2.0)))

    np.testing.assert_allclose(resampled, analytical, atol=2e-5)


# ---------------------------------------------------------------------------
# N5 -- get_values_grid_exact analytical check (the jnp.interp linear pipeline
# used by DataCreation -- a separate code path from get_values_grid's
# catmull-rom one; a bug in one wouldn't necessarily show up in the other)
# ---------------------------------------------------------------------------
#
# get_values_grid_exact takes a `sol_dict` of plain scalars/arrays directly
# (gamma_value, beta_value, pod_local -- no RBF/z involved at all), so no
# exact-coefficient trick is needed here: the known shift/stretch/beta values
# are just written into the dict directly. Same shift/stretch cases and same
# closed-form analytical Gaussian as N1/N2.
#
# Measured (fp64): Transported (gamma=0.5) 2.249e-6 -- matches N1's own
# catmull-rom error almost exactly, because jnp.interp is exact at aligned
# nodes too (see N1's docstring) and what's left in both cases is the same
# boundary-truncation residual (the Gaussian's true tail leaking past
# [x.min(), x.max()], clamped to 0 by construction) rather than an
# interpolation-scheme difference. atol=1e-5, matching N1.
#
# Stretched (both N2 cases: pure_stretch and shift_and_stretch) 1.2445e-3 --
# same magnitude as N4-extended's old np.interp number, for the same reason:
# jnp.interp is linear, and here `old_x = (x-shift-axis)/stretch+axis` is
# sampled at the ORIGINAL mesh spacing (0.01) while querying `pod_local`
# (width 0.1, curvature 1/sigma^2=100) -- the same h^2/8*|f''| bound
# (0.01^2/8*100 = 1.25e-3) predicts the measured value to within 0.4%, same
# as N4-extended's derivation. atol=2e-3.
#
# Confirmed fp32 gives the same numbers (2.249e-6 / 1.24454e-3, agreeing to
# ~6 sig figs) -- not roundoff, so one tolerance is safe at both precisions
# (same conclusion as N1/N2). Detection power: a wrong gamma (0.3 instead of
# 0.5) gives max err 0.891; a wrong stretch (1.5 instead of 2.0) gives
# max err 0.209 -- both far above atol.

_N5_TRANSPORTED_ATOL = 1e-5
_N5_STRETCHED_ATOL = 2e-3


def test_N5_transported_get_values_grid_exact_matches_analytical_gaussian():
    """get_values_grid_exact's closed-form-scalar/jnp.interp path should
    reproduce the same shifted Gaussian as N1's get_values_grid (RBF/catmull-
    rom) path -- an independent code path, so a bug in one wouldn't
    necessarily show up in the other."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    sol_dict = {"gamma_value": 0.5, "beta_value": 1.0, "pod_local": _POD_LOCAL_N1}
    result = mode.get_values_grid_exact(sol_dict)
    analytical = jnp.exp(-0.5 * jnp.power((_X_N1 - 1.0) / 0.1, 2.0))

    np.testing.assert_allclose(np.asarray(result), np.asarray(analytical), atol=_N5_TRANSPORTED_ATOL)


@pytest.mark.parametrize("shift_val,stretch_val,case_id", _N2_CASES,
                         ids=[c[2] for c in _N2_CASES])
def test_N5_stretched_get_values_grid_exact_matches_analytical_gaussian(shift_val, stretch_val, case_id):
    """Same cross-check for StretchedMode.get_values_grid_exact, both N2
    cases (pure stretch and shift+stretch)."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_shift = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_stretch = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    sol_dict = {"gamma_shift_value": shift_val, "gamma_stretch_value": stretch_val,
                "beta_value": 1.0, "pod_local": _POD_LOCAL_N2}
    result = mode.get_values_grid_exact(sol_dict)
    center = 0.5 + shift_val
    width = 0.1 * stretch_val
    analytical = jnp.exp(-0.5 * jnp.power((_X_N2 - center) / width, 2.0))

    np.testing.assert_allclose(np.asarray(result), np.asarray(analytical), atol=_N5_STRETCHED_ATOL)


# ---------------------------------------------------------------------------
# N6 -- GlobalMode sanity (no transform to test, rounds out coverage
# symmetrically with the Transported/Stretched modes above)
# ---------------------------------------------------------------------------
#
# GlobalMode has no coordinate transform at all: get_values_grid is just
# frankenstein_softplus(alpha_value) * pod_global, elementwise, on the
# original mesh -- no interpolation anywhere. alpha=2.0 (not 1.0) is pinned
# via the same bias=True exact-coefficient trick as N1, deliberately
# non-round so a bug dropping the multiply entirely wouldn't accidentally
# look right. Since alpha=2.0 is above frankenstein_softplus's 0.01 cutoff,
# it is the identity there (see util.py), so the whole pipeline
# (RBF predict -> softplus -> multiply) is exact arithmetic, no floating-
# point interpolation residual to tolerate -- confirmed bit-exact
# (jnp.all(result == expected) True) at both fp32 and fp64, so
# assert_array_equal (zero tolerance) is used, same as N4.
#
# Also checks get_values_nodes (trivially `(self._x, get_values_grid(...))`
# per the source) and get_values_grid_exact (same arithmetic, scalar
# alpha_value/pod_global passed directly, no RBF) both agree exactly with
# get_values_grid -- rounding out the same three entry points N1-N5 covered
# for the other two mode types, even though none of them involve a transform
# here. Detection power: a result that's just alpha broadcast without the
# pod_global multiply differs by up to 2.0 -- easily caught by exact equality.

def test_N6_global_get_values_grid_is_exact_arithmetic():
    """GlobalMode.get_values_grid(z, params) == alpha * pod_global, exactly --
    no interpolation involved for this mode at all."""
    RBF_alpha = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = GlobalMode(_X_N1, RBF_alpha)

    alpha_coeff = RBF_alpha.get_coeff(2.0 * jnp.ones(_DATA_Z_N1.shape[0]))
    params = mode.dict_to_arr({"pod_global": _POD_LOCAL_N1, "alpha_coeff": alpha_coeff})

    result = mode.get_values_grid(0.37, params)
    expected = 2.0 * _POD_LOCAL_N1

    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_N6_global_get_values_nodes_and_get_values_grid_exact_agree():
    """get_values_nodes should return the untouched mesh plus the same value
    as get_values_grid; get_values_grid_exact (plain scalars, no RBF) should
    also match exactly -- all three entry points, no transform to disagree
    on."""
    RBF_alpha = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = GlobalMode(_X_N1, RBF_alpha)

    alpha_coeff = RBF_alpha.get_coeff(2.0 * jnp.ones(_DATA_Z_N1.shape[0]))
    params = mode.dict_to_arr({"pod_global": _POD_LOCAL_N1, "alpha_coeff": alpha_coeff})

    grid_result = mode.get_values_grid(0.37, params)
    x_out, v_out = mode.get_values_nodes(0.37, params)
    exact_result = mode.get_values_grid_exact({"alpha_value": 2.0, "pod_global": _POD_LOCAL_N1})

    np.testing.assert_array_equal(np.asarray(x_out), np.asarray(_X_N1))
    np.testing.assert_array_equal(np.asarray(v_out), np.asarray(grid_result))
    np.testing.assert_array_equal(np.asarray(exact_result), np.asarray(grid_result))


# ---------------------------------------------------------------------------
# N7 -- self-consistency: get_values_grid (RBF/catmull-rom) vs.
# get_values_grid_exact (exact scalars/linear-interp) shouldn't diverge when
# both represent the same underlying beta/gamma
# ---------------------------------------------------------------------------
#
# Unlike N1-N6, this deliberately does NOT use the bias=True exact-
# coefficient trick -- the whole point is to exercise REAL (approximate)
# RBF fitting, i.e. RBF_interpol's actual defaults (bias=False, nugget=1e-3),
# which is what the codebase uses unless bias=True is passed explicitly.
# get_coeff(1.0*ones(3)) does NOT reproduce beta==1.0 exactly even AT a
# training z (ridge-regression/nugget effect, see group G/M) -- measured
# predict() residuals off the constant target: beta ~3e-3 to 5e-3, gamma/
# shift ~1.5e-3 to 2.5e-3, same order at an on-node z (0.6, a training point)
# as an off-node one (0.37) -- the nugget doesn't vanish on-node without
# bias=True.
#
# get_values_grid uses these approximately-fitted RBF coefficients (the
# "ansatz reconstruction" pathway); get_values_grid_exact is fed the TRUE
# known scalar beta/gamma directly (the "ground truth" pathway, as
# DataCreation would use it). The two should agree up to the RBF's own
# fitting noise (which dominates -- N1/N5 already showed the interpolation-
# scheme difference alone is ~1e-6, three orders of magnitude smaller).
#
# Measured max|diff| (fp64), z=0.6 (on-node) / z=0.37 (off-node):
# Transported 1.1917e-2 / 6.710e-3; Stretched 5.411e-3 / 3.609e-3. Confirmed
# fp32 gives the same numbers (Transported z=0.6: 1.1917e-2) -- algorithmic
# RBF-fitting noise, not roundoff, so one tolerance is safe at both
# precisions. atol set from the worse (on-node) case with ~1.7-1.9x headroom:
# Transported 2e-2, Stretched 1e-2. Detection power: a wrong ground-truth
# gamma fed to get_values_grid_exact (0.3 instead of 0.5) gives max err
# 0.894 -- far above atol.

_N7_TRANSPORTED_ATOL = 2e-2
_N7_STRETCHED_ATOL = 1e-2

_N7_Z_CASES = [(0.6, "on_node"), (0.37, "off_node")]


@pytest.mark.parametrize("z,case_id", _N7_Z_CASES, ids=[c[1] for c in _N7_Z_CASES])
def test_N7_transported_grid_and_exact_paths_are_self_consistent(z, case_id):
    """get_values_grid (real, non-bias RBF fit + catmull-rom) and
    get_values_grid_exact (exact known beta/gamma + linear interp) should
    stay close when both represent the same nominal shift/beta -- two
    independent code paths for "evaluate this mode" that shouldn't diverge
    beyond the RBF's own (already-characterized) fitting noise."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N1)   # bias=False (production default)
    RBF_gamma = RBF_interpol(1.0, 1.0, _DATA_Z_N1)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(1.0 * jnp.ones(_DATA_Z_N1.shape[0]))
    gamma_coeff = RBF_gamma.get_coeff(0.5 * jnp.ones(_DATA_Z_N1.shape[0]))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    grid_result = mode.get_values_grid(z, params)
    exact_result = mode.get_values_grid_exact({
        "gamma_value": 0.5, "beta_value": 1.0, "pod_local": _POD_LOCAL_N1,
    })

    np.testing.assert_allclose(np.asarray(grid_result), np.asarray(exact_result),
                               atol=_N7_TRANSPORTED_ATOL)


@pytest.mark.parametrize("z,case_id", _N7_Z_CASES, ids=[c[1] for c in _N7_Z_CASES])
def test_N7_stretched_grid_and_exact_paths_are_self_consistent(z, case_id):
    """Same self-consistency check for StretchedMode (shift=0.5, stretch=2)."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N2)
    RBF_shift = RBF_interpol(1.0, 1.0, _DATA_Z_N2)
    RBF_stretch = RBF_interpol(1.0, 1.0, _DATA_Z_N2)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    n_z = _DATA_Z_N2.shape[0]
    beta_coeff = RBF_beta.get_coeff(1.0 * jnp.ones(n_z))
    shift_coeff = RBF_shift.get_coeff(0.5 * jnp.ones(n_z))
    stretch_coeff = RBF_stretch.get_coeff(2.0 * jnp.ones(n_z))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N2, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    grid_result = mode.get_values_grid(z, params)
    exact_result = mode.get_values_grid_exact({
        "gamma_shift_value": 0.5, "gamma_stretch_value": 2.0,
        "beta_value": 1.0, "pod_local": _POD_LOCAL_N2,
    })

    np.testing.assert_allclose(np.asarray(grid_result), np.asarray(exact_result),
                               atol=_N7_STRETCHED_ATOL)


# ---------------------------------------------------------------------------
# N8 -- boundary / degenerate case: shift/stretch large enough to push the
# bump's support entirely outside the domain should give a clean near-zero
# result, not NaN/Inf/wraparound nonsense
# ---------------------------------------------------------------------------
#
# Motivated by the test_integration_unregularized.py incident (a periodic
# gamma(z) sampled at its own zero-crossings silently produced a near-rank-1
# dataset) -- not the same bug, but the same family of concern: does the
# single-snapshot "mass leaves the domain" case degrade gracefully. interpax's
# `extrap` parameter, per its docstring, returns the SPECIFIED VALUE (here
# 0.) for any query point outside [x[0], x[-1]] -- a hard clamp, not a decaying
# tail -- so once every query point's transformed coordinate is outside the
# padded domain, the result should be EXACTLY 0., not just small.
#
# Two distinct mechanisms are covered:
#
# 1. Large SHIFT (gamma / gamma_shift): translates the whole query domain far
#    outside the padded mesh -- every old_x lands outside [x.min()-10%,
#    x.max()+10%], so every point should hit the extrap value exactly.
#    Measured (both fp32 and fp64): jnp.all(result == 0.) True, no NaN/Inf.
#
# 2. Large STRETCH with an off-axis pod_local (peak at 0.7, not at the
#    scaling_axis 0.5): unlike shift, stretch does not translate old_x
#    outward -- old_x = (x-shift-axis)/stretch + axis COMPRESSES toward the
#    axis as stretch grows, so what leaves the (in-domain) sampled region is
#    the SUPPORT itself, not the query points. A pod_local centered exactly
#    on the axis (as N1/N2's fixture is) would never show this -- pure
#    stretch never moves the axis point, so its peak stays fully sampled
#    however large the stretch. Measured max|result| as stretch grows,
#    confirming a real, monotonic, non-noise effect (not exact zero, since
#    old_x stays in-domain near the axis, just far from the off-axis peak):
#    stretch=1 -> 1.0 (full peak, sanity check the setup finds the peak when
#    NOT compressed); stretch=10 -> 1.109e-2; stretch=50 -> 7.318e-4.
#    Confirmed fp32 matches fp64 to 6 sig figs (7.318024e-4 / 7.318025e-4) --
#    algorithmic, not roundoff. atol=2e-3 (~2.7x headroom over the stretch=50
#    measurement); the stretch=1 sanity value (1.0) is far outside this atol,
#    confirming the threshold is meaningful and not trivially satisfied by
#    any stretch value.

_N8_STRETCH_OFFAXIS_ATOL = 2e-3


def test_N8_transported_large_shift_pushes_bump_out_of_domain():
    """gamma=5.0 (50x the bump's own width, far past the 10%-padded domain)
    should push the entire shifted bump out of [x.min()-10%, x.max()+10%],
    giving an exact all-zero result via interpax's extrap=0. clamp -- not
    NaN, not Inf, not a residual bump."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, _DATA_Z_N1, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(1.0 * jnp.ones(_DATA_Z_N1.shape[0]))
    gamma_coeff = RBF_gamma.get_coeff(5.0 * jnp.ones(_DATA_Z_N1.shape[0]))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    result = mode.get_values_grid(0.37, params)

    assert not bool(jnp.any(jnp.isnan(result)))
    assert not bool(jnp.any(jnp.isinf(result)))
    np.testing.assert_array_equal(np.asarray(result), np.zeros_like(np.asarray(result)))


def test_N8_stretched_large_shift_pushes_bump_out_of_domain():
    """Same shift-based boundary check for StretchedMode (gamma_shift=5.0,
    gamma_stretch=1.0 -- pure translation, isolating the shift mechanism)."""
    RBF_beta = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_shift = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    RBF_stretch = RBF_interpol(1.0, 1.0, _DATA_Z_N2, bias=True)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    n_z = _DATA_Z_N2.shape[0]
    beta_coeff = RBF_beta.get_coeff(1.0 * jnp.ones(n_z))
    shift_coeff = RBF_shift.get_coeff(5.0 * jnp.ones(n_z))
    stretch_coeff = RBF_stretch.get_coeff(1.0 * jnp.ones(n_z))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N2, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    result = mode.get_values_grid(0.37, params)

    assert not bool(jnp.any(jnp.isnan(result)))
    assert not bool(jnp.any(jnp.isinf(result)))
    np.testing.assert_array_equal(np.asarray(result), np.zeros_like(np.asarray(result)))


def test_N8_stretched_large_stretch_compresses_away_from_offaxis_support():
    """A large stretch with pod_local peaked OFF the scaling axis (0.7, not
    0.5): unlike the shift cases above, stretch compresses old_x toward the
    axis rather than translating it out, so the mechanism that leaves the
    sampled region here is the support itself, not the query domain -- a
    distinct failure mode from the two shift-based cases above, and one that
    a pod_local centered on the axis (as N1/N2's fixture is) could never
    surface."""
    x_offaxis = jnp.linspace(0.0, 1.0, 101)
    pod_offaxis = jnp.exp(-0.5 * jnp.power((x_offaxis - 0.7) / 0.05, 2.0))
    data_z = jnp.array([0.2, 0.6, 1.0])

    RBF_beta = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_shift = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_stretch = RBF_interpol(1.0, 1.0, data_z, bias=True)
    mode = StretchedMode(x_offaxis, RBF_beta, RBF_shift, RBF_stretch)

    beta_coeff = RBF_beta.get_coeff(1.0 * jnp.ones(3))
    shift_coeff = RBF_shift.get_coeff(0.0 * jnp.ones(3))
    stretch_coeff = RBF_stretch.get_coeff(50.0 * jnp.ones(3))
    params = mode.dict_to_arr({
        "pod_local": pod_offaxis, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    result = mode.get_values_grid(0.37, params)

    assert not bool(jnp.any(jnp.isnan(result)))
    assert not bool(jnp.any(jnp.isinf(result)))
    np.testing.assert_allclose(np.asarray(result), np.zeros_like(np.asarray(result)),
                               atol=_N8_STRETCH_OFFAXIS_ATOL)


# --- N8 extended: the padding-decay TRANSITION itself, not just its two
# limits (fully inside vs. fully clamped to 0) ------------------------------
#
# The two shift tests above show gamma=5.0 gives an all-zero result, and N1
# shows gamma=0.5 (mesh-aligned, fully interior) gives the undistorted
# Gaussian. Neither exercises the padding region [x.max(), x.max()+10%]
# itself, where catmull-rom has to interpolate from pod_local's own boundary
# value down to the padding knot's forced 0 -- a transition where a cubic
# (non-monotone-preserving) spline could in principle overshoot/oscillate.
#
# gamma=-1.0 routes old_x = x - gamma = x + 1.0 so that x in [0, 0.1] maps to
# old_x in [1.0, 1.1] -- exactly the padding-decay zone -- while pod_local is
# built with its peak AT x.max()=1.0 (not the usual mid-domain peak), so the
# zone starts from a clearly nonzero value (1.0) instead of an
# already-near-zero tail. Measured (fp32 and fp64 agree; fp64 gives
# -1.11e-16 at the padding knot vs. fp32's exact 0.0 -- roundoff-level
# either way): the 11 values from old_x=1.0 to old_x=1.1 are strictly
# monotonically decreasing (all 10 successive diffs negative, ~-0.09 to
# -0.10 each), landing within 1e-9 of exactly 0 at the padding knot itself
# (old_x=1.1). StretchedMode with stretch=1 reduces to the identical
# computation (verified bit-for-bit identical output) since stretch=1 makes
# its coordinate transform pure translation -- not retested separately, as
# it would exercise the exact same interpax call with zero added coverage.

def test_N8_transported_padding_decay_is_monotonic_and_reaches_zero():
    """The catmull-rom transition inside the 10% padding zone (from
    pod_local's own boundary value down to the padding knot's forced 0)
    should decrease monotonically and land near-exactly at 0 at the far
    edge -- not overshoot, oscillate, or undershoot negative by more than
    floating-point noise."""
    x_edge = jnp.linspace(0.0, 1.0, 101)
    pod_edge = jnp.exp(-0.5 * jnp.power((x_edge - 1.0) / 0.05, 2.0))  # peak at x.max()
    data_z = jnp.array([0.2, 0.6, 1.0])

    RBF_beta = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, data_z, bias=True)
    mode = TransportedMode(x_edge, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(3))
    gamma_coeff = RBF_gamma.get_coeff(-1.0 * jnp.ones(3))  # old_x = x + 1.0
    params = mode.dict_to_arr({
        "pod_local": pod_edge, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    result = mode.get_values_grid(0.37, params)
    decay_zone = result[:11]  # x in [0, 0.1] -> old_x in [1.0, 1.1]

    assert not bool(jnp.any(jnp.isnan(result)))
    np.testing.assert_array_less(np.diff(np.asarray(decay_zone)), 0.0)
    np.testing.assert_allclose(float(decay_zone[-1]), 0.0, atol=1e-9)
    np.testing.assert_allclose(float(decay_zone[0]), 1.0, atol=1e-9)


_FD_H_FP64 = 1e-5
_FD_ATOL_FP64 = 1e-6
_FD_RTOL_FP64 = 1e-6

_FD_H_SMOKE = 1e-3
_FD_ATOL_SMOKE = 1e-2
_FD_RTOL_SMOKE = 1e-2


def _central_fd_grad(f, params, h):
    """Elementwise central-difference gradient of a scalar-valued f(params)."""
    p = np.asarray(params, dtype=np.float64)
    grad = np.zeros_like(p)
    for i in range(p.shape[0]):
        plus, minus = p.copy(), p.copy()
        plus[i] += h
        minus[i] -= h
        f_plus = float(f(jnp.asarray(plus, dtype=params.dtype)))
        f_minus = float(f(jnp.asarray(minus, dtype=params.dtype)))
        grad[i] = (f_plus - f_minus) / (2.0 * h)
    return grad


# --- N8 extended: gradient accuracy strictly INSIDE the padding-decay zone --
#
# Same fixture as the value/monotonicity test above (gamma=-1.0 routes
# old_x = x+1.0, so x in [0,0.1] maps to old_x in [1.0,1.1] -- the padding
# transition). Checked per-index (isolating each of the 11 output points'
# own gradient) before writing this, and found a genuine, important
# subtlety: the two ENDPOINTS of this range are NOT ordinary interior points
# and are deliberately excluded here.
#
# - idx=10 (old_x=1.1 EXACTLY, the outer padding knot itself) is a true
#   corner: the VALUE decays smoothly to exactly 0 approaching from inside
#   (left one-sided derivative measured ~10, w.r.t. the bias gamma_coeff
#   component), but is pinned flat at exactly 0 for every old_x beyond it
#   (extrap=0. is a hard constant, right one-sided derivative = 0) -- a
#   value-continuous, derivative-DIScontinuous kink. Confirmed directly by
#   evaluating one-sided differences approaching old_x=1.1 from both sides
#   (left slope stably ~10, right slope stably 0, no convergence toward a
#   shared limit as the step shrinks -- unlike an ordinary smooth point).
#   Autodiff, evaluated exactly at the corner, deterministically returns the
#   LEFT-hand derivative (~10 for the bias component) -- a legitimate,
#   well-defined choice, but NOT what central FD converges to: central FD
#   straddles the corner and converges to the AVERAGE of the two one-sided
#   derivatives (measured: FD -> 5.0, autodiff -> 10.0, a clean, exact 2x
#   mismatch, reproducing at every h from 1e-4 to 1e-6 -- not roundoff, not
#   a bug, just the mathematical fact that a classical derivative doesn't
#   exist at a corner, only one-sided derivatives do, and FD and autodiff
#   resolve that ambiguity differently). This is a real property of the
#   `extrap=0.` boundary construction, not anything specific to
#   TransportedMode -- reproduced with a minimal `interpax.interp1d` call
#   with no W_POD code involved at all.
# - idx=0 (old_x=1.0 EXACTLY, i.e. x.max() itself) is NOT a corner (one-sided
#   slopes were confirmed to converge to the SAME limit, ~-8.91, as the step
#   shrinks toward 0) -- but the local FD landscape there is more sharply
#   curved than an ordinary interior point (being adjacent, one catmull-rom
#   segment away, to the padding node pinned at exactly 0), so it needs a
#   smaller FD step than idx=1-9 to converge to the same tightness; at the
#   same h=1e-5 used below it shows max|diff|~9.8e-3 instead of ~1e-8. Not
#   included in this check to keep a single h/atol usable for the whole
#   zone; N9 already establishes that get_values_grid's gradient is
#   generally accurate, so this isn't a coverage gap, just a scoping choice.
#
# Indices 1-9 (old_x strictly in (1.0, 1.1), i.e. genuinely interior to the
# padding transition, away from both the ordinary-but-sharper x.max() knot
# and the true extrap corner) show clean agreement: fp64 h=1e-5,
# max|diff|=6.31e-9 (matching N9's own interior precision); fp32 h=1e-3 (N9's
# own smoke step), max|diff|=3.82e-4 -- both comfortably inside N9's existing
# tolerances, reused directly below rather than introducing new ones.
# Detection power: a deliberately 2x-scaled gamma-coefficient gradient block
# (simulating a dropped or doubled chain-rule factor) gives max|diff|=4.83 --
# far above atol.

def _transported_padding_edge_mode_and_params():
    """Same fixture as test_N8_transported_padding_decay_is_monotonic_and_reaches_zero:
    pod_local peaked at x.max(), gamma=-1.0 routes old_x = x+1.0 into the
    padding-decay zone [1.0, 1.1] for x in [0, 0.1]."""
    x_edge = jnp.linspace(0.0, 1.0, 101)
    pod_edge = jnp.exp(-0.5 * jnp.power((x_edge - 1.0) / 0.05, 2.0))
    data_z = jnp.array([0.2, 0.6, 1.0])
    RBF_beta = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, data_z, bias=True)
    mode = TransportedMode(x_edge, RBF_beta, RBF_gamma)
    beta_coeff = RBF_beta.get_coeff(jnp.ones(3))
    gamma_coeff = RBF_gamma.get_coeff(-1.0 * jnp.ones(3))
    params = mode.dict_to_arr({
        "pod_local": pod_edge, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })
    return mode, params


def _padding_zone_weighted_sum_fn(mode, weights):
    def f(params):
        result = mode.get_values_grid(0.37, params)
        return jnp.sum(weights * result[1:10])  # strictly interior to the padding zone
    return f


def test_N8_transported_padding_decay_gradient_matches_fd_smoke():
    mode, params = _transported_padding_edge_mode_and_params()
    weights = jnp.asarray(np.random.default_rng(9).standard_normal(9))
    f = _padding_zone_weighted_sum_fn(mode, weights)

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_SMOKE)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_SMOKE, rtol=_FD_RTOL_SMOKE)


@_REQUIRE_X64
def test_N8_transported_padding_decay_gradient_matches_fd():
    mode, params = _transported_padding_edge_mode_and_params()
    weights = jnp.asarray(np.random.default_rng(9).standard_normal(9))
    f = _padding_zone_weighted_sum_fn(mode, weights)

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_FP64)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_FP64, rtol=_FD_RTOL_FP64)


# --- N8 extended: the corner point itself, deliberately left failing --------
#
# The two tests above deliberately exclude idx=10 (old_x=1.1 exactly, the
# outer padding knot) because it's a genuine one-sided-derivative corner (see
# this file's module-level discussion above and future.md section 8): the
# value decays smoothly to 0 approaching from inside, but extrap=0. pins
# every point beyond it flat at exactly 0, so the classical derivative
# doesn't exist there -- only a left derivative (~10, what autodiff
# deterministically returns) and a right derivative (0). Central FD
# straddles the corner and converges to their AVERAGE (~5.0), so autodiff
# and FD disagree by a clean, reproducible ~5 (fp64) / ~4.99 (fp32) at this
# single point -- not roundoff, not noise, confirmed stable across h from
# 1e-4 to 1e-7.
#
# This test intentionally includes that point and uses the SAME tolerances
# as the interior check above (not a corner-aware looser one) -- it is
# EXPECTED TO FAIL, by design, until the underlying construction changes
# (e.g. constraining get_values_grid's spline to have a matching boundary
# slope, or some other fix not yet decided on). Deliberately NOT marked
# xfail: an xfail(strict=True) would report as a passing/expected result in
# test-run summaries, but the intent here is for this to show up as a
# visible failure in every run until it's actually fixed -- a standing,
# impossible-to-miss reminder rather than a quietly-tracked known issue.

def test_N8_transported_padding_decay_corner_gradient_matches_fd_smoke():
    """Deliberately failing (see comment above): idx=10 is a genuine
    one-sided-derivative corner, so FD and autodiff disagree by design."""
    mode, params = _transported_padding_edge_mode_and_params()

    def f(p):
        return mode.get_values_grid(0.37, p)[10]

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_SMOKE)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_SMOKE, rtol=_FD_RTOL_SMOKE)


@_REQUIRE_X64
def test_N8_transported_padding_decay_corner_gradient_matches_fd():
    """Deliberately failing (see comment above): idx=10 is a genuine
    one-sided-derivative corner, so FD and autodiff disagree by design."""
    mode, params = _transported_padding_edge_mode_and_params()

    def f(p):
        return mode.get_values_grid(0.37, p)[10]

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_FP64)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_FP64, rtol=_FD_RTOL_FP64)


_X = jnp.linspace(0.0, 1.0, 21)          # small mesh -- FD loops over every param
_POD_LOCAL = jnp.exp(-0.5 * jnp.power((_X - 0.5) / 0.15, 2.0))
_DATA_Z = jnp.array([0.2, 0.6, 1.0])


def _transported_mode_and_params(seed):
    rng = np.random.default_rng(seed)
    RBF_beta = RBF_interpol(0.5, 1.0, _DATA_Z, bias=True)
    RBF_gamma = RBF_interpol(0.5, 1.0, _DATA_Z, bias=True)
    mode = TransportedMode(_X, RBF_beta, RBF_gamma)
    beta_coeff = RBF_beta.get_coeff(jnp.asarray(1.0 + 0.3 * rng.standard_normal(3)))
    gamma_coeff = RBF_gamma.get_coeff(jnp.asarray(0.3 * rng.standard_normal(3)))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })
    return mode, params


def _stretched_mode_and_params(seed):
    rng = np.random.default_rng(seed)
    RBF_beta = RBF_interpol(0.5, 1.0, _DATA_Z, bias=True)
    RBF_shift = RBF_interpol(0.5, 1.0, _DATA_Z, bias=True)
    RBF_stretch = RBF_interpol(0.5, 1.0, _DATA_Z, bias=True)
    mode = StretchedMode(_X, RBF_beta, RBF_shift, RBF_stretch)
    beta_coeff = RBF_beta.get_coeff(jnp.asarray(1.0 + 0.3 * rng.standard_normal(3)))
    shift_coeff = RBF_shift.get_coeff(jnp.asarray(0.1 * rng.standard_normal(3)))
    # keep stretch safely away from 0 -- old_x divides by it
    stretch_coeff = RBF_stretch.get_coeff(jnp.asarray(1.2 + 0.2 * rng.standard_normal(3)))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })
    return mode, params


def _weighted_sum_fn(mode, z, weights):
    """Reduce get_values_grid's (len(x),) output to a scalar for jax.grad."""
    return lambda params: jnp.sum(mode.get_values_grid(z, params) * weights)


# ---------------------------------------------------------------------------
# N9 -- TransportedMode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N9_transported_get_values_grid_gradient_matches_fd_smoke(z0):
    mode, params = _transported_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_SMOKE)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_SMOKE, rtol=_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N9_transported_get_values_grid_gradient_matches_fd(z0):
    mode, params = _transported_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_FP64)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_FP64, rtol=_FD_RTOL_FP64)


# ---------------------------------------------------------------------------
# N9 -- StretchedMode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N9_stretched_get_values_grid_gradient_matches_fd_smoke(z0):
    mode, params = _stretched_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_SMOKE)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_SMOKE, rtol=_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N9_stretched_get_values_grid_gradient_matches_fd(z0):
    mode, params = _stretched_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    g_auto = np.asarray(jax.grad(f)(params))
    g_fd = _central_fd_grad(f, params, _FD_H_FP64)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_FP64, rtol=_FD_RTOL_FP64)


# --- N9 extended: get_values_grid's gradient vs. the TRUE analytic Gaussian
# derivative (not FD) -- an independent oracle, not just internal consistency
# between autodiff and FD -----------------------------------------------------
#
# N9 above only checks that autodiff agrees with FD -- both are numerical
# estimates of the SAME actual (spline-based) function's derivative, so
# agreement between them proves autodiff is differentiating the real
# computation correctly, but says nothing about how closely that computation
# tracks the IDEALIZED continuous transform. This closes that gap: gamma/
# shift/stretch are pinned EXACTLY (bias=True + constant-target trick, as in
# N1/N2), so get_values_grid's result is SUPPOSED to equal a closed-form
# Gaussian -- chain-ruling through predict()'s known-linear dependence on
# coeff (predict(z,coeff) = coeff[-1] + sum_i coeff[i]*K(z,data_z_i), so
# d(predict)/d(coeff_i) = K(z,data_z_i), or 1 for the bias term -- see
# util.py's predict()) gives a fully closed-form target for
# d(get_values_grid)/d(gamma_coeff_i) with NO numerical differentiation
# anywhere in the reference computation.
#
# Initial calibration used gamma=0.5 (N1's own value) and found a startling
# ~0.35% relative gap -- but that value is deliberately MESH-ALIGNED (an
# exact multiple of the 0.01 spacing), which for a VALUE check (N1) is ideal
# (every query lands exactly on another node), but for a DERIVATIVE check is
# close to worst-case: catmull-rom is only C1 (continuous first derivative,
# discontinuous second derivative across knots), so every one of the 101
# evaluation points sitting exactly ON a knot means central differences (and,
# it turns out, autodiff through the same non-smooth computation) see the
# knot-to-knot curvature jump at every point simultaneously. Confirmed via a
# direct h-sweep (bypassing this test suite's harness) that even a raw
# central-FD on the same spline shows non-monotonic, oscillatory convergence
# near gamma=0.5, unlike the clean O(h^2) convergence N9 itself measures --
# because N9 uses REAL (non-exact, effectively-random) RBF-fitted
# coefficients, which are generically off-node.
#
# Re-measured at OFF-node values (gamma=0.517, shift=0.317, stretch=1.83 --
# none an exact multiple of the mesh spacing): the relative gap shrinks but
# does NOT vanish -- Transported gamma 0.083%, Stretched shift 0.111%,
# Stretched stretch 0.288% (fp32 matches fp64 to within noise, e.g.
# Transported gamma 0.0834% vs 0.0834% -- algorithmic spline-derivative
# approximation error, not roundoff, same conclusion pattern as every other
# item in this file). This residual is real and expected: differentiating an
# interpolant is inherently less accurate than evaluating it, even away from
# knots. rtol=1e-2 (~3.5x-12x headroom over the three measured relative
# errors), atol=1e-4 floor for near-zero entries. Detection power: a
# deliberately sign-flipped analytic reference gives relative error >200%,
# ruled out at this tolerance.

_N9EXT_RTOL = 1e-2
_N9EXT_ATOL = 1e-4


def test_N9_transported_get_values_grid_gradient_matches_analytic_gaussian_derivative():
    """d(get_values_grid)/d(gamma_coeff), via autodiff, checked against the
    fully closed-form derivative of the transformed Gaussian chain-ruled
    through predict()'s known-linear coefficient dependence -- independent of
    both autodiff's internal correctness (already checked by N9 against FD)
    and of catmull-rom's OWN self-consistency, since the reference here never
    calls interpax at all."""
    data_z = jnp.array([0.2, 0.6, 1.0])
    gamma_val = 0.517  # off-node: not a multiple of the 0.01 mesh spacing
    RBF_beta = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, data_z, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(3))
    gamma_coeff = RBF_gamma.get_coeff(gamma_val * jnp.ones(3))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    weights = jnp.asarray(np.random.default_rng(42).standard_normal(_X_N1.shape[0]))

    def weighted_sum(p):
        return jnp.sum(weights * mode.get_values_grid(0.37, p))

    g_auto = np.asarray(jax.grad(weighted_sum)(params))

    center, sigma, beta_val = 0.5 + gamma_val, 0.1, 1.0
    x_np = np.asarray(_X_N1)
    gauss = np.exp(-0.5 * np.power((x_np - center) / sigma, 2.0))
    dgauss_dcenter = gauss * (x_np - center) / sigma ** 2
    df_dgamma = float(np.sum(np.asarray(weights) * beta_val * dgauss_dcenter))
    K_gamma = np.array([float(RBF_gamma.rbf_kernel(0.37, data_z[i])) for i in range(3)] + [1.0])
    analytic_grad = df_dgamma * K_gamma

    i_gamma0 = _X_N1.shape[0] + RBF_beta.n_param()
    g_auto_gamma = g_auto[i_gamma0:i_gamma0 + 4]

    np.testing.assert_allclose(g_auto_gamma, analytic_grad, rtol=_N9EXT_RTOL, atol=_N9EXT_ATOL)


def test_N9_stretched_get_values_grid_gradient_matches_analytic_gaussian_derivative():
    """Same check for StretchedMode: both shift and stretch coefficient
    gradients against the closed-form derivative of the stretched Gaussian
    (peak on the scaling axis, as in N2, so the stretch derivative's
    center-shift term vanishes and only the width term survives)."""
    data_z = jnp.array([0.2, 0.6, 1.0])
    shift_val, stretch_val = 0.317, 1.83  # both off-node
    RBF_beta = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_shift = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_stretch = RBF_interpol(1.0, 1.0, data_z, bias=True)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(3))
    shift_coeff = RBF_shift.get_coeff(shift_val * jnp.ones(3))
    stretch_coeff = RBF_stretch.get_coeff(stretch_val * jnp.ones(3))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N2, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    weights = jnp.asarray(np.random.default_rng(43).standard_normal(_X_N2.shape[0]))

    def weighted_sum(p):
        return jnp.sum(weights * mode.get_values_grid(0.37, p))

    g_auto = np.asarray(jax.grad(weighted_sum)(params))

    beta_val, sigma = 1.0, 0.1
    center, width = 0.5 + shift_val, sigma * stretch_val
    x_np = np.asarray(_X_N2)
    gauss = np.exp(-0.5 * np.power((x_np - center) / width, 2.0))
    dgauss_dcenter = gauss * (x_np - center) / width ** 2
    dgauss_dwidth = gauss * np.power(x_np - center, 2) / width ** 3
    df_dshift = float(np.sum(np.asarray(weights) * beta_val * dgauss_dcenter))
    df_dstretch = float(np.sum(np.asarray(weights) * beta_val * dgauss_dwidth * sigma))

    K_shift = np.array([float(RBF_shift.rbf_kernel(0.37, data_z[i])) for i in range(3)] + [1.0])
    K_stretch = np.array([float(RBF_stretch.rbf_kernel(0.37, data_z[i])) for i in range(3)] + [1.0])
    analytic_shift = df_dshift * K_shift
    analytic_stretch = df_dstretch * K_stretch

    n_x = _X_N2.shape[0]
    i_shift0 = n_x + RBF_beta.n_param()
    i_stretch0 = i_shift0 + RBF_shift.n_param()

    np.testing.assert_allclose(g_auto[i_shift0:i_shift0 + 4], analytic_shift,
                               rtol=_N9EXT_RTOL, atol=_N9EXT_ATOL)
    np.testing.assert_allclose(g_auto[i_stretch0:i_stretch0 + 4], analytic_stretch,
                               rtol=_N9EXT_RTOL, atol=_N9EXT_ATOL)


# ---------------------------------------------------------------------------
# N10 -- non-constant shift/stretch across snapshots: a real (not pinned) RBF
# fit to a genuinely different value per snapshot, checked against the
# analytical transform at each snapshot's own z
# ---------------------------------------------------------------------------
#
# N1/N2/N9 deliberately *isolate* the transform from RBF behaviour via the
# constant-target trick (bias=True + a constant target predicts exactly,
# for any z, with zero RBF-fitting noise). This one instead exercises the
# connection between them: does a snapshot-varying RBF prediction correctly
# flow through to a correspondingly different transform result at each z --
# closer to actual training-time usage, where each snapshot has its own
# gamma(z_j). Also implicitly an integration check that a genuinely
# non-constant RBF (not a degenerate constant one) composes correctly with
# get_values_grid's interpolation pipeline.
#
# data_z=[0.2, 0.6, 1.0], gamma targets [0.2, 0.5, 1.0] -- deliberately
# NON-LINEAR (a straight line through the endpoints would give 0.6 at z=0.6,
# not the 0.5 used), so the RBF genuinely has to bend, not just extrapolate a
# trend. beta target constant 1 (isolating amplitude from this check, which
# is about the shift/stretch trajectory). bias=True, theta=0.3 chosen
# deliberately -- NOT the theta=1.0 default used elsewhere in this file, see
# below.
#
# Measured (fp32) at the two non-degenerate training z's:
# Transported z=0.6 (expect shift ~0.5): 8.3846e-4; z=1.0 (expect shift ~1.0):
# 1.5638e-6. Stretched z=0.6 (expect shift~0.3, stretch~1.5): 2.9695e-4;
# z=1.0 (expect shift~0.5, stretch~2.0): 5.0420e-4. Confirmed fp64 gives the
# same order of magnitude (Stretched: 2.9706e-4/5.0430e-4) -- NOT the ~1e5x
# tightening fp64 gave N9's gradient checks -- so this runs at the default
# fp32 with no skip-guard needed. Root-caused with a direct control:
# refitting the same gamma target with nugget=1e-10 instead of the default
# 1e-3 shrinks the training-node deviation at z=0.6 from 1.3868e-4 to
# 1.3899e-11 -- i.e. this ~1e-4-1e-3 error is the deliberate ridge-
# regularization effect of `nugget` (get_coeff doesn't exactly interpolate
# its own training data unless nugget is tiny), not floating-point roundoff
# or an interpolation-scheme artifact -- consistent with future.md section 6.
#
# atol set with ~2-2.4x headroom over each mode's worst measured case:
# Transported 2e-3, Stretched 1e-3. Detection power: comparing z=0.6's
# result against z=0.2's analytical target (a stand-in for "get_values_grid
# used the wrong z") gives max err 0.989 -- far above atol.
#
# Pick theta deliberately, don't default to 1.0: with theta=1.0 (same
# fixture) z=0.6's error jumps to 4.352e-2, ~50x worse than at theta=0.3 --
# purely because a large length scale over-smooths the genuine curvature in
# the [0.2, 0.5, 1.0] trajectory, pulling the fit toward the (wrong, more
# linear) endpoint-to-endpoint trend. Not a bug -- the same "hyperparameters
# must be chosen thoughtfully, not defaulted" lesson group M ran into -- but
# asserted here (not just noted in prose) so this comment can't silently
# rot if RBF_interpol's behaviour ever changes.

_N10_TRANSPORTED_ATOL = 2e-3
_N10_STRETCHED_ATOL = 1e-3

_N10_TRANSPORTED_CASES = [(0.6, 0.5), (1.0, 1.0)]
_N10_STRETCHED_CASES = [(0.6, 0.3, 1.5), (1.0, 0.5, 2.0)]


@pytest.mark.parametrize("z,expected_shift", _N10_TRANSPORTED_CASES,
                         ids=[f"z={c[0]}" for c in _N10_TRANSPORTED_CASES])
def test_N10_transported_nonconstant_gamma_matches_analytical_at_each_z(z, expected_shift):
    """A genuinely non-constant (real, fitted) gamma(z) trajectory should
    still produce the analytically-correct shift at each of its own training
    z's -- not just the constant-target case N1 already covers."""
    data_z = jnp.array([0.2, 0.6, 1.0])
    RBF_beta = RBF_interpol(0.3, 1.0, data_z, bias=True)
    RBF_gamma = RBF_interpol(0.3, 1.0, data_z, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(3))
    gamma_coeff = RBF_gamma.get_coeff(jnp.array([0.2, 0.5, 1.0]))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    result = mode.get_values_grid(z, params)
    analytical = jnp.exp(-0.5 * jnp.power((_X_N1 - (0.5 + expected_shift)) / 0.1, 2.0))

    np.testing.assert_allclose(np.asarray(result), np.asarray(analytical), atol=_N10_TRANSPORTED_ATOL)


@pytest.mark.parametrize("z,expected_shift,expected_stretch", _N10_STRETCHED_CASES,
                         ids=[f"z={c[0]}" for c in _N10_STRETCHED_CASES])
def test_N10_stretched_nonconstant_shift_stretch_matches_analytical_at_each_z(z, expected_shift, expected_stretch):
    """Same check for StretchedMode: genuinely non-constant shift(z) AND
    stretch(z) trajectories, both checked jointly at each training z."""
    data_z = jnp.array([0.2, 0.6, 1.0])
    RBF_beta = RBF_interpol(0.3, 1.0, data_z, bias=True)
    RBF_shift = RBF_interpol(0.3, 1.0, data_z, bias=True)
    RBF_stretch = RBF_interpol(0.3, 1.0, data_z, bias=True)
    mode = StretchedMode(_X_N2, RBF_beta, RBF_shift, RBF_stretch)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(3))
    shift_coeff = RBF_shift.get_coeff(jnp.array([0.0, 0.3, 0.5]))
    stretch_coeff = RBF_stretch.get_coeff(jnp.array([1.0, 1.5, 2.0]))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N2, "beta_coeff": beta_coeff,
        "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
    })

    result = mode.get_values_grid(z, params)
    center = 0.5 + expected_shift
    width = 0.1 * expected_stretch
    analytical = jnp.exp(-0.5 * jnp.power((_X_N2 - center) / width, 2.0))

    np.testing.assert_allclose(np.asarray(result), np.asarray(analytical), atol=_N10_STRETCHED_ATOL)


def test_N10_transported_theta_1_degrades_the_nonlinear_gamma_fit():
    """Sanity check for the module comment above: theta=1.0 (RBF_interpol's
    default length scale, used everywhere else in this file where the target
    is constant) over-smooths this genuinely non-linear gamma(z) trajectory,
    giving a meaningfully worse fit than the deliberately-chosen theta=0.3 --
    confirming the choice isn't arbitrary, and pinning the claim so it can't
    silently rot."""
    data_z = jnp.array([0.2, 0.6, 1.0])
    RBF_beta = RBF_interpol(1.0, 1.0, data_z, bias=True)
    RBF_gamma = RBF_interpol(1.0, 1.0, data_z, bias=True)
    mode = TransportedMode(_X_N1, RBF_beta, RBF_gamma)

    beta_coeff = RBF_beta.get_coeff(jnp.ones(3))
    gamma_coeff = RBF_gamma.get_coeff(jnp.array([0.2, 0.5, 1.0]))
    params = mode.dict_to_arr({
        "pod_local": _POD_LOCAL_N1, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    result = mode.get_values_grid(0.6, params)
    analytical = jnp.exp(-0.5 * jnp.power((_X_N1 - 1.0) / 0.1, 2.0))
    err = float(jnp.max(jnp.abs(result - analytical)))

    assert err > 10 * _N10_TRANSPORTED_ATOL


# ---------------------------------------------------------------------------
# N11 -- get_values_nodes gradient: exact analytic match, no tolerance to
# hide behind (extends N4's "pure arithmetic, no interpolation" philosophy to
# derivatives)
# ---------------------------------------------------------------------------
#
# get_values_nodes has no interpolation anywhere (N4), so unlike
# get_values_grid (N9-extended above), its gradient has NO spline-derivative
# approximation error to worry about -- it should match a hand-derived
# closed-form gradient EXACTLY, at any coefficients (not just the exact-bias
# trick), since predict() is linear in its coeff argument
# (d(predict)/d(coeff_i) = K(z,data_z_i), or 1 for bias -- see util.py) and
# everything downstream is elementwise arithmetic.
#
# TransportedMode: x_out = x + gamma(z), v_out = softplus(beta(z))*pod_local.
#   d(x_out)/d(gamma_coeff_i) = K_gamma_i(z)        [uniform, broadcasts]
#   d(v_out)/d(beta_coeff_i)  = softplus'(beta_raw) * K_beta_i(z) * pod_local
#   d(v_out)/d(pod_local_j)   = softplus(beta_raw) * delta_j
#   (all cross-terms zero: x_out doesn't depend on beta/pod_local; v_out
#   doesn't depend on gamma_coeff)
#
# StretchedMode: x_out = shift(z) + axis + stretch(z)*(x-axis),
#                v_out = softplus(beta(z))*pod_local.
#   d(x_out)/d(shift_coeff_i)   = K_shift_i(z)
#   d(x_out)/d(stretch_coeff_i) = K_stretch_i(z) * (x-axis)
#   d(v_out)/d(beta_coeff_i), d(v_out)/d(pod_local_j) as above.
#
# Uses N9's real (non-pinned) RBF-fitted coefficients, not the exact-bias
# trick -- deliberately, to show the gradient is exact regardless of how
# well the RBF happens to fit anything (there's no fitting-noise-vs-
# interpolation-error distinction to make here, since nothing is
# interpolated in x at all). beta's raw predicted value (~1.09 in this
# fixture) is comfortably above frankenstein_softplus's 0.01 cutoff, so
# softplus'(beta_raw) = 1 exactly (identity branch, no need to hand-derive
# the sigmoid branch).
#
# Measured: fp64 max|diff| = 0.0 (bit-for-bit exact) across all three
# parameter blocks (pod_local, beta_coeff, gamma_coeff/shift_coeff/
# stretch_coeff), both modes. fp32 max|diff| ~2e-8-5.6e-8 (floating-point
# summation-order noise from vmap'd sums over 3 data_z points -- not a
# real discrepancy); atol=1e-6 (~20-50x headroom) used for both precisions
# rather than branching, since fp64's exact 0.0 trivially satisfies it too.
# Detection power: a deliberately wrong analytic formula (dropping the
# (x-axis) factor from the stretch-coefficient derivative, as if it were
# uniform like the shift derivative) gives max|diff|=0.168 -- far above atol.

_N11_ATOL = 1e-6


def _rbf_kernel_vec(rbf, z):
    return np.array([float(rbf.rbf_kernel(z, rbf.data_z[i])) for i in range(rbf.data_z.shape[0])] + [1.0])


def test_N11_transported_get_values_nodes_gradient_matches_analytic():
    """jax.grad through get_values_nodes should match the hand-derived exact
    gradient bit-for-bit (fp64) / to floating-point summation noise (fp32) --
    there is no interpolation anywhere in this function to introduce any
    other source of discrepancy."""
    mode, params = _transported_mode_and_params(seed=7)
    n_x = _X.shape[0]
    rng = np.random.default_rng(70)
    w_x = jnp.asarray(rng.standard_normal(n_x))
    w_v = jnp.asarray(rng.standard_normal(n_x))

    def weighted_sum(p):
        x_out, v_out = mode.get_values_nodes(0.6, p)
        return jnp.sum(w_x * x_out) + jnp.sum(w_v * v_out)

    g_auto = np.asarray(jax.grad(weighted_sum)(params))

    pd = mode.arr_to_dict(params)
    beta_raw = mode._RBF_beta.predict(0.6, pd['beta_coeff'])
    beta_value = float(frankenstein_softplus(beta_raw))
    softplus_prime = float(jax.grad(frankenstein_softplus)(beta_raw))

    n_beta, n_gamma = mode._RBF_beta.n_param(), mode._RBF_gamma.n_param()
    i_beta0, i_gamma0 = n_x, n_x + n_beta

    sum_wx = float(jnp.sum(w_x))
    sum_wv_pod = float(jnp.sum(w_v * pd['pod_local']))
    K_gamma = _rbf_kernel_vec(mode._RBF_gamma, 0.6)
    K_beta = _rbf_kernel_vec(mode._RBF_beta, 0.6)

    analytic_pod = beta_value * np.asarray(w_v)
    analytic_beta = sum_wv_pod * softplus_prime * K_beta
    analytic_gamma = sum_wx * K_gamma

    np.testing.assert_allclose(g_auto[:n_x], analytic_pod, atol=_N11_ATOL)
    np.testing.assert_allclose(g_auto[i_beta0:i_beta0 + n_beta], analytic_beta, atol=_N11_ATOL)
    np.testing.assert_allclose(g_auto[i_gamma0:i_gamma0 + n_gamma], analytic_gamma, atol=_N11_ATOL)


def test_N11_stretched_get_values_nodes_gradient_matches_analytic():
    """Same exact-gradient check for StretchedMode, including the
    stretch-coefficient derivative's (x-axis) factor (the one place this
    mode's gradient isn't just a uniform broadcast like the others)."""
    mode, params = _stretched_mode_and_params(seed=7)
    n_x = _X.shape[0]
    rng = np.random.default_rng(71)
    w_x = jnp.asarray(rng.standard_normal(n_x))
    w_v = jnp.asarray(rng.standard_normal(n_x))

    def weighted_sum(p):
        x_out, v_out = mode.get_values_nodes(0.6, p)
        return jnp.sum(w_x * x_out) + jnp.sum(w_v * v_out)

    g_auto = np.asarray(jax.grad(weighted_sum)(params))

    pd = mode.arr_to_dict(params)
    beta_raw = mode._RBF_beta.predict(0.6, pd['beta_coeff'])
    beta_value = float(frankenstein_softplus(beta_raw))
    softplus_prime = float(jax.grad(frankenstein_softplus)(beta_raw))

    n_beta = mode._RBF_beta.n_param()
    n_shift = mode._RBF_gamma_shift.n_param()
    n_stretch = mode._RBF_gamma_stretch.n_param()
    i_beta0 = n_x
    i_shift0 = n_x + n_beta
    i_stretch0 = i_shift0 + n_shift

    axis = mode.scaling_axis
    sum_wx = float(jnp.sum(w_x))
    sum_wx_xaxis = float(jnp.sum(w_x * (_X - axis)))
    sum_wv_pod = float(jnp.sum(w_v * pd['pod_local']))

    K_shift = _rbf_kernel_vec(mode._RBF_gamma_shift, 0.6)
    K_stretch = _rbf_kernel_vec(mode._RBF_gamma_stretch, 0.6)
    K_beta = _rbf_kernel_vec(mode._RBF_beta, 0.6)

    analytic_pod = beta_value * np.asarray(w_v)
    analytic_beta = sum_wv_pod * softplus_prime * K_beta
    analytic_shift = sum_wx * K_shift
    analytic_stretch = sum_wx_xaxis * K_stretch

    np.testing.assert_allclose(g_auto[:n_x], analytic_pod, atol=_N11_ATOL)
    np.testing.assert_allclose(g_auto[i_beta0:i_beta0 + n_beta], analytic_beta, atol=_N11_ATOL)
    np.testing.assert_allclose(g_auto[i_shift0:i_shift0 + n_shift], analytic_shift, atol=_N11_ATOL)
    np.testing.assert_allclose(g_auto[i_stretch0:i_stretch0 + n_stretch], analytic_stretch, atol=_N11_ATOL)


# ---------------------------------------------------------------------------
# N12 -- Hessian diagonal (d^2 get_values_grid / d(param_i)^2) vs. central
# second-difference FD
# ---------------------------------------------------------------------------
#
# Same motivation as N9 (autodiff differentiating the real, non-textbook
# interpax/RBF composition correctly), one order higher. Autodiff's Hessian
# via `jnp.diag(jax.hessian(f)(params))`; the independent reference is a
# per-parameter central SECOND difference,
# `(f(p+h)-2f(p)+f(p-h))/h^2`, holding every other parameter fixed --
# this gives the Hessian's DIAGONAL only (no cross/off-diagonal terms,
# matching what was asked for).
#
# Central second-difference FD has a fundamentally different, WORSE
# roundoff/truncation tradeoff than the first-derivative FD in N9: its
# truncation error is O(h^2) same as central first differences, but its
# roundoff error is O(epsilon/h^2) (dividing numerator noise by h^2, not h),
# a much steeper roundoff penalty -- so the usable h is generally LARGER,
# and achievable precision is generally WORSE, than for first-derivative
# checks (a real, expected floor: second differences are inherently a
# noisier estimator than first differences, not something this test can
# average away). Both existing tests confirm which parameters have curvature
# at all: `pod_local` and `beta_coeff` (the amplitude) give an EXACTLY-ZERO
# analytic diagonal here -- `pod_local` enters `get_values_grid` linearly
# with no self-interaction, and `beta_coeff` enters through `predict()`
# (linear) then `frankenstein_softplus` evaluated comfortably above its 0.01
# cutoff (identity there, second derivative exactly 0), then as a pure
# linear multiplicative prefactor -- so ONLY `gamma_coeff` (Transported) /
# `gamma_shift_coeff`+`gamma_stretch_coeff` (Stretched) have real curvature
# (through `old_x`'s nonlinear entry into the catmull-rom spline). Only
# nonzero-curvature entries are checked -- an exactly-zero analytic Hessian
# entry is already covered more directly (and exactly) by N9's own gradient
# check having zero curvature to get wrong in the first place.
#
# h-sweep (fp64, 8 seeds x {Transported, Stretched} x {z=0.6, z=0.8} = 32
# configs, every nonzero-curvature diagonal entry in each): unlike N9's
# smooth, monotonic h-convergence, this is a narrow trough -- h=1e-3 down to
# h=3e-4 gave a MUCH worse worst-case (up to 8% relative) than h=1e-4 (worst
# relative `2.92e-5`), because larger h starts to alias real higher-order
# curvature in the more sharply-curved StretchedMode configs, while smaller
# h (3e-5 and below) hits the O(epsilon/h^2) roundoff wall. h=1e-4 sits in
# the trough; atol=1e-5, rtol=1e-3 (~14x-34x headroom over the swept worst
# cases, both a near-zero-magnitude entry needing the atol floor and a
# large-magnitude one needing the rtol) -- verified stable, `_REQUIRE_X64`-
# gated (this narrow a trough is not expected to survive fp32 roundoff, and
# it doesn't -- see below).
#
# fp32: the usable h window is far narrower and shallower -- h=1e-2 was the
# best available (h an order of magnitude higher or lower both give
# multi-hundred-percent relative errors on some entry), and even at that
# best h, worst-case relative error runs up to ~30% on larger-curvature
# entries and appears to blow up further on near-zero-magnitude ones (an
# artifact of dividing by a tiny reference value, not a real problem --
# handled by the atol floor rather than rtol). atol=0.05, rtol=0.5 (~1.5x-2x
# headroom over the swept worst cases) -- deliberately loose: this smoke
# tier's job is catching a GROSSLY wrong Hessian (dropped curvature term,
# sign flip, wrong scaling), not finely validating second-derivative
# accuracy at single precision, which second-order FD fundamentally cannot
# do well. Detection power (fp64): a deliberately 1.5x-scaled Hessian
# diagonal (simulating a dropped or doubled chain-rule factor) gives
# max|diff|=0.135 against a max bound of 0.00028 -- far outside tolerance.

_HESS_FD_H_FP64 = 1e-4
_HESS_FD_ATOL_FP64 = 1e-5
_HESS_FD_RTOL_FP64 = 1e-3

_HESS_FD_H_SMOKE = 1e-2
_HESS_FD_ATOL_SMOKE = 0.05
_HESS_FD_RTOL_SMOKE = 0.5


def _central_fd_hessian_diag(f, params, h):
    """Elementwise central second-difference estimate of the Hessian's
    diagonal: (f(p+h e_i) - 2f(p) + f(p-h e_i)) / h^2 for each i."""
    p = np.asarray(params, dtype=np.float64)
    f0 = float(f(jnp.asarray(p, dtype=params.dtype)))
    diag = np.zeros_like(p)
    for i in range(p.shape[0]):
        plus, minus = p.copy(), p.copy()
        plus[i] += h
        minus[i] -= h
        f_plus = float(f(jnp.asarray(plus, dtype=params.dtype)))
        f_minus = float(f(jnp.asarray(minus, dtype=params.dtype)))
        diag[i] = (f_plus - 2.0 * f0 + f_minus) / (h * h)
    return diag


def _assert_nonzero_curvature_entries_match(hess_diag_auto, hess_diag_fd, atol, rtol):
    nonzero = hess_diag_auto != 0.0
    np.testing.assert_allclose(hess_diag_auto[nonzero], hess_diag_fd[nonzero], atol=atol, rtol=rtol)


@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N12_transported_get_values_grid_hessian_diag_matches_fd_smoke(z0):
    mode, params = _transported_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    hess_diag_auto = np.asarray(jnp.diag(jax.hessian(f)(params)))
    hess_diag_fd = _central_fd_hessian_diag(f, params, _HESS_FD_H_SMOKE)
    _assert_nonzero_curvature_entries_match(hess_diag_auto, hess_diag_fd,
                                            _HESS_FD_ATOL_SMOKE, _HESS_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N12_transported_get_values_grid_hessian_diag_matches_fd(z0):
    mode, params = _transported_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    hess_diag_auto = np.asarray(jnp.diag(jax.hessian(f)(params)))
    hess_diag_fd = _central_fd_hessian_diag(f, params, _HESS_FD_H_FP64)
    _assert_nonzero_curvature_entries_match(hess_diag_auto, hess_diag_fd,
                                            _HESS_FD_ATOL_FP64, _HESS_FD_RTOL_FP64)


@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N12_stretched_get_values_grid_hessian_diag_matches_fd_smoke(z0):
    mode, params = _stretched_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    hess_diag_auto = np.asarray(jnp.diag(jax.hessian(f)(params)))
    hess_diag_fd = _central_fd_hessian_diag(f, params, _HESS_FD_H_SMOKE)
    _assert_nonzero_curvature_entries_match(hess_diag_auto, hess_diag_fd,
                                            _HESS_FD_ATOL_SMOKE, _HESS_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N12_stretched_get_values_grid_hessian_diag_matches_fd(z0):
    mode, params = _stretched_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)

    hess_diag_auto = np.asarray(jnp.diag(jax.hessian(f)(params)))
    hess_diag_fd = _central_fd_hessian_diag(f, params, _HESS_FD_H_FP64)
    _assert_nonzero_curvature_entries_match(hess_diag_auto, hess_diag_fd,
                                            _HESS_FD_ATOL_FP64, _HESS_FD_RTOL_FP64)


# ---------------------------------------------------------------------------
# N13 -- Hessian OFF-DIAGONAL entries (d^2 get_values_grid / d(param_i)d(param_j),
# i != j) vs. mixed central second-difference FD
# ---------------------------------------------------------------------------
#
# N12 only checks the diagonal. Mapping the full Hessian (not just its
# diagonal) for the N9/N12 fixtures first, rather than guessing which
# off-diagonal entries are worth checking, found real structure:
#
# - pod_local x pod_local (any i != j): exactly zero -- interpax's
#   interpolated value, for a FIXED query point (fixed gamma/shift/stretch),
#   is a linear functional of the WHOLE pod_local array (tangents are linear
#   combinations of samples, the Hermite blend weights depend only on knot
#   positions), so the entire pod-pod block, not just its diagonal, is exactly
#   zero -- a general fact, not fixture-specific.
# - beta_coeff x beta_coeff (any i != j): exactly zero IN THIS FIXTURE, for
#   the same fixture-specific reason N12 already flagged for the beta_coeff
#   diagonal (frankenstein_softplus sits in its identity regime here) -- NOT
#   a general fact, see N12's discussion.
# - Every OTHER cross-block pair (pod-beta, pod-gamma/shift/stretch,
#   beta-gamma/shift/stretch, and within gamma/shift/stretch itself) is
#   generically nonzero -- get_values_grid is a PRODUCT of pieces that are
#   each linear/smooth in their own variable but not in each other's, so
#   cross partials between different blocks don't vanish even where a
#   block's own diagonal does (the ordinary bilinear-cross-term fact:
#   d^2(xy)/dx^2 = 0 and d^2(xy)/dy^2 = 0 don't imply d^2(xy)/dxdy = 0).
#
# Rather than hardcode specific indices (fragile -- which indices are
# nonzero depends on old_x's local support and can shift with the fixture),
# one representative nonzero pair is picked PER BLOCK-COMBINATION
# automatically from the actual autodiff Hessian, mirroring N12's own
# "only check what autodiff itself reports as nonzero" philosophy.
#
# Calibration used the mixed central second difference,
# (f(p+h e_i+h e_j) - f(p+h e_i-h e_j) - f(p-h e_i+h e_j) + f(p-h e_i-h e_j))
# / (4h^2) -- same truncation/roundoff character as N12's diagonal formula
# (still O(h^2) truncation, O(epsilon/h^2) roundoff), but combining 4
# evaluations instead of 3. Swept across both modes, both z0, and (during
# calibration only) several seeds: found the same "narrow trough, not
# monotonic convergence" pattern as N12, and confirmed the trough location
# depends on the SPECIFIC pair's curvature magnitude (e.g. one
# StretchedMode config's shift-shift entry, magnitude ~-5.85, needed
# h~3e-5, an order of magnitude smaller than the h=1e-4 that worked
# everywhere else) -- i.e. there is no universal off-diagonal h any more
# than there was for the diagonal. For the actual fixture this file uses
# (seed=7, both modes, both z0), h=1e-4 (fp64) / h=3e-3 (fp32) work cleanly
# across every representative pair checked (worst fp64 relative error
# ~1.4e-3 at h=1e-4 on the pod-gamma pair; fp32 needs a larger h than N12's
# diagonal used, since some pairs here have much smaller-magnitude curvature
# -- e.g. pod-beta ~0.001 -- that needs a bigger step to clear the fp32
# roundoff floor, while others, e.g. StretchedMode's shift-shift ~-17, are
# still comfortably covered by the same atol/rtol pair). Reuses N12's exact
# atol/rtol constants (both tiers) -- same character of error, no need for
# separate ones. Detection power (fp64): a deliberately 1.5x-scaled
# off-diagonal entry gives diff=0.098 against a bound of 0.0002 -- far
# outside tolerance.

_HESS_OFFDIAG_H_FP64 = 1e-4
_HESS_OFFDIAG_H_SMOKE = 3e-3


def _mode_block_ranges(mode, n_x):
    """{block_name: (start, stop)} column ranges in dict_to_arr order."""
    if isinstance(mode, TransportedMode):
        n_beta = mode._RBF_beta.n_param()
        n_gamma = mode._RBF_gamma.n_param()
        i0, i1, i2 = n_x, n_x + n_beta, n_x + n_beta + n_gamma
        return {"pod": (0, n_x), "beta": (n_x, i1), "gamma": (i1, i2)}
    else:
        n_beta = mode._RBF_beta.n_param()
        n_shift = mode._RBF_gamma_shift.n_param()
        n_stretch = mode._RBF_gamma_stretch.n_param()
        i0 = n_x
        i1 = i0 + n_beta
        i2 = i1 + n_shift
        i3 = i2 + n_stretch
        return {"pod": (0, n_x), "beta": (n_x, i1), "shift": (i1, i2), "stretch": (i2, i3)}


def _select_representative_offdiag_pairs(hess_auto, blocks, tol=1e-9):
    """One (i, j, label) per block-combination where autodiff reports a
    nonzero entry -- found dynamically rather than hardcoded, since which
    specific indices are nonzero depends on the fixture's local support."""
    names = list(blocks.keys())
    pairs = []
    for a in range(len(names)):
        for b in range(a, len(names)):
            name_a, name_b = names[a], names[b]
            ra, rb = blocks[name_a], blocks[name_b]
            found = None
            for i in range(*ra):
                for j in range(max(i + 1, rb[0]), rb[1]):
                    if abs(hess_auto[i, j]) > tol:
                        found = (i, j)
                        break
                if found:
                    break
            if found:
                pairs.append((found[0], found[1], f"{name_a}-{name_b}"))
    return pairs


def _mixed_central_fd_hessian_entry(f, params, i, j, h):
    p = np.asarray(params, dtype=np.float64)
    pp, pm, mp, mm = p.copy(), p.copy(), p.copy(), p.copy()
    pp[i] += h; pp[j] += h
    pm[i] += h; pm[j] -= h
    mp[i] -= h; mp[j] += h
    mm[i] -= h; mm[j] -= h
    f_pp = float(f(jnp.asarray(pp, dtype=params.dtype)))
    f_pm = float(f(jnp.asarray(pm, dtype=params.dtype)))
    f_mp = float(f(jnp.asarray(mp, dtype=params.dtype)))
    f_mm = float(f(jnp.asarray(mm, dtype=params.dtype)))
    return (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)


def _check_offdiag_pairs_match_fd(mode, params, f, h, atol, rtol):
    hess_auto = np.asarray(jax.hessian(f)(params))
    pairs = _select_representative_offdiag_pairs(hess_auto, _mode_block_ranges(mode, _X.shape[0]))
    assert len(pairs) > 0, "expected at least one nonzero off-diagonal pair"
    for i, j, label in pairs:
        fd = _mixed_central_fd_hessian_entry(f, params, i, j, h)
        np.testing.assert_allclose(hess_auto[i, j], fd, atol=atol, rtol=rtol,
                                   err_msg=f"off-diagonal pair {label} ({i},{j})")


@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N13_transported_get_values_grid_hessian_offdiag_matches_fd_smoke(z0):
    mode, params = _transported_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)
    _check_offdiag_pairs_match_fd(mode, params, f, _HESS_OFFDIAG_H_SMOKE,
                                  _HESS_FD_ATOL_SMOKE, _HESS_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N13_transported_get_values_grid_hessian_offdiag_matches_fd(z0):
    mode, params = _transported_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)
    _check_offdiag_pairs_match_fd(mode, params, f, _HESS_OFFDIAG_H_FP64,
                                  _HESS_FD_ATOL_FP64, _HESS_FD_RTOL_FP64)


@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N13_stretched_get_values_grid_hessian_offdiag_matches_fd_smoke(z0):
    mode, params = _stretched_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)
    _check_offdiag_pairs_match_fd(mode, params, f, _HESS_OFFDIAG_H_SMOKE,
                                  _HESS_FD_ATOL_SMOKE, _HESS_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("z0", [0.6, 0.8], ids=["z=on-node", "z=off-node"])
def test_N13_stretched_get_values_grid_hessian_offdiag_matches_fd(z0):
    mode, params = _stretched_mode_and_params(seed=7)
    weights = jnp.asarray(np.random.default_rng(70).standard_normal(_X.shape[0]))
    f = _weighted_sum_fn(mode, z0, weights)
    _check_offdiag_pairs_match_fd(mode, params, f, _HESS_OFFDIAG_H_FP64,
                                  _HESS_FD_ATOL_FP64, _HESS_FD_RTOL_FP64)
