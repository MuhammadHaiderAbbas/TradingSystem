"""The risk manager cannot be bypassed by any code path."""

import ast
from pathlib import Path

import pytest

from sentinel.config import CostConfig, RiskConfig
from sentinel.core.errors import RiskVeto
from sentinel.core.types import Order, Side
from sentinel.execution.broker import Broker
from sentinel.execution.sim import SimBroker
from sentinel.risk.manager import RiskManager

PKG = Path(__file__).resolve().parents[1] / "sentinel"


def test_broker_without_risk_manager_refuses():
    b = SimBroker(1000, CostConfig())
    with pytest.raises(RiskVeto):
        b.place_order(Order("X", Side.BUY, 1), None)


def test_order_without_approval_refused():
    b = SimBroker(1000, CostConfig())
    b.bind_risk(RiskManager(RiskConfig()))
    with pytest.raises(RiskVeto):
        b.place_order(Order("X", Side.BUY, 1), None)


def test_cannot_rebind_a_different_risk_manager():
    b = SimBroker(1000, CostConfig())
    b.bind_risk(RiskManager(RiskConfig()))
    with pytest.raises(RuntimeError):
        b.bind_risk(RiskManager(RiskConfig()))


def test_adapters_cannot_override_place_order():
    with pytest.raises(TypeError):
        class Evil(SimBroker):  # noqa: F841
            def place_order(self, order, approval):
                return self._submit(order)


def _calls(attr: str):
    """Yield (file, line) for every call to ``<expr>.<attr>(...)`` in the package."""
    for f in PKG.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr:
                yield f.relative_to(PKG).as_posix(), node.lineno


def test_submit_only_called_from_the_risk_gated_template():
    callers = {f for f, _ in _calls("_submit")}
    assert callers == {"execution/broker.py"}, callers


def test_place_order_only_called_from_approved_modules():
    allowed = {"execution/order_manager.py", "execution/smart.py"}
    callers = {f for f, _ in _calls("place_order")}
    assert callers <= allowed, callers - allowed


def test_approvals_only_issued_by_risk_manager():
    # `_issue` (create an approval without evaluation) is private to the risk manager
    callers = {f for f, _ in _calls("_issue")}
    assert callers <= {"risk/manager.py"}, callers


def test_every_broker_subclass_uses_template():
    import sentinel.execution.alpaca_broker  # noqa: F401
    import sentinel.execution.ccxt_broker  # noqa: F401
    import sentinel.execution.paper  # noqa: F401

    def subclasses(c):
        for s in c.__subclasses__():
            yield s
            yield from subclasses(s)

    for cls in subclasses(Broker):
        assert "place_order" not in cls.__dict__, cls
