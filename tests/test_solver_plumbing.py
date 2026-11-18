"""Pure-plumbing unit tests for ``W_POD.solver`` (group I of ``unit_tests.md``).

Only the constructor wiring is exercised: ``OptimCfg`` field storage,
``L2_W1_optimizer`` copying ``.ansatz`` off its config, and ``IteratedSolver``'s
"all configs share one ansatz" assertion.

``AnsatzConfig`` is *stubbed* here -- the code under test reads only
``ansatz_cfg.ansatz`` in ``__init__``.  A real ``AnsatzConfig`` pulls in
``DataCreation`` and belongs to the loss-term / integration tier.  The
``IteratedSolver`` cross-config check is a bare ``assert`` (skipped under
``python -O``).
"""

import jax.numpy as jnp
import pytest

from W_POD.modes import GlobalMode
from W_POD.util import RBF_interpol
from W_POD.ansatz import Ansatz
from W_POD.solver import OptimCfg, L2_W1_optimizer, IteratedSolver


def _ansatz():
    x = jnp.linspace(0.0, 1.0, 8)
    return Ansatz([GlobalMode(x, RBF_interpol(1.0, 1.0, jnp.array([0.0, 0.5, 1.0])))])


class _StubAnsatzCfg:
    """Stands in for AnsatzConfig; the solver constructors read only .ansatz."""

    def __init__(self, ansatz):
        self.ansatz = ansatz


def _optim_cfg():
    return OptimCfg(True, False, True, 10, "SLSQP")


# ---------------------------------------------------------------------------
# I52 -- OptimCfg stores its constructor fields
# ---------------------------------------------------------------------------

def test_I52_optim_cfg_stores_fields():
    cfg = OptimCfg(
        stop_if_linesearch_fails=True,
        verbose=False,
        log_w1=True,
        maxiter=123,
        scipy_optim_method="SLSQP",
        callback_prefix="[2/5] ",
    )
    assert cfg.stop_if_linesearch_fails is True
    assert cfg.verbose is False
    assert cfg.log_w1 is True
    assert cfg.maxiter == 123
    assert cfg.scipy_optim_method == "SLSQP"
    assert cfg.callback_prefix == "[2/5] "


def test_I52_optim_cfg_callback_prefix_defaults_to_empty():
    assert OptimCfg(True, True, True, 50, "L-BFGS-B").callback_prefix == ""


# ---------------------------------------------------------------------------
# I53 -- L2_W1_optimizer.__init__ wiring
# ---------------------------------------------------------------------------

def test_I53_optimizer_takes_ansatz_from_cfg():
    ansatz = _ansatz()
    cfg = _StubAnsatzCfg(ansatz)
    optim_cfg = _optim_cfg()
    data = object()

    opt = L2_W1_optimizer(data, cfg, optim_cfg)

    assert opt.ansatz is ansatz          # copied off the config
    assert opt.ansatz_cfg is cfg
    assert opt.optim_cfg is optim_cfg
    assert opt._data is data


# ---------------------------------------------------------------------------
# I54 -- IteratedSolver: all configs must share one ansatz
# ---------------------------------------------------------------------------

def test_I54_iterated_solver_shared_ansatz_ok():
    a = _ansatz()
    optim_cfgs = [_optim_cfg(), _optim_cfg()]
    cfgs = [_StubAnsatzCfg(a), _StubAnsatzCfg(a)]

    solver = IteratedSolver(None, cfgs, optim_cfgs, n_iter=3)

    assert solver.ansatz is a
    assert solver.n_iter == 3
    assert solver.ansatz_cfgs is cfgs
    assert solver.optim_cfgs is optim_cfgs
    assert len(solver.solvers) == 2
    assert all(isinstance(s, L2_W1_optimizer) for s in solver.solvers)
    assert [s.ansatz for s in solver.solvers] == [a, a]


def test_I54_iterated_solver_mismatched_ansatz_raises():
    a1, a2 = _ansatz(), _ansatz()          # distinct Ansatz instances
    optim_cfgs = [_optim_cfg(), _optim_cfg()]

    with pytest.raises(AssertionError):
        IteratedSolver(None, [_StubAnsatzCfg(a1), _StubAnsatzCfg(a2)],
                       optim_cfgs, n_iter=1)
