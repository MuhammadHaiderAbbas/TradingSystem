"""Funds: deposits, withdrawals, profit accounting, and the guarantees around them."""

import ast
import base64
import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel.config import FundsConfig, RiskConfig
from sentinel.core.modes import HALT_KEY
from sentinel.core.types import Bar, Mode, OrderPurpose, Side
from sentinel.data.synthetic import generate_ohlcv
from sentinel.data.timeframes import epoch_seconds
from sentinel.execution.paper import PaperBroker
from sentinel.funds import Funds, FundsError
from sentinel.risk.manager import OrderIntent, RiskContext, RiskManager
from tests.conftest import settings_for

T0 = 1_704_067_200.0


@pytest.fixture
def funds(store):
    f = Funds(store, "PAPER", FundsConfig(min_withdrawal=10, profit_take_pct=0.1))
    f.ensure_opening(10_000, ts=T0)
    return f


def test_opening_deposits_and_profit(funds):
    funds.deposit(2_000, ts=T0 + 10)
    funds.apply_pending(ts=T0 + 10)
    s = funds.summary(equity=12_600, cash=12_600)
    assert s["net_invested"] == 12_000 and s["profit"] == 600
    assert s["profit_pct"] == pytest.approx(0.05)
    assert s["withdrawable_profit"] == 600 and s["suggestion"] is None
    s2 = funds.summary(equity=13_500, cash=13_500)
    assert s2["suggestion"] and "Consider withdrawing" in s2["suggestion"]


def test_opening_accounts_for_flows_recorded_before_first_run(store):
    f = Funds(store, "PAPER")
    f.deposit(500)
    f.ensure_opening(10_000)  # paper account before the deposit is applied
    assert f.summary(10_500, 10_500)["net_invested"] == 10_500


def test_withdrawal_limits(funds):
    with pytest.raises(FundsError, match="minimum"):
        funds.request_withdrawal(5, 11_000, 11_000)
    with pytest.raises(FundsError, match="available"):
        funds.request_withdrawal(20_000, 11_000, 11_000)
    with pytest.raises(FundsError, match="profit"):
        funds.request_withdrawal(1_500, 11_000, 11_000, profit_only=True)  # only 1,000 is profit
    fid, chk = funds.request_withdrawal(1_000, 11_000, 11_000, profit_only=True)
    assert chk.ready and funds.get(fid)["status"] == "ready" and funds.reserved() == 1_000
    with pytest.raises(FundsError):  # the reserved profit can't be requested twice
        funds.request_withdrawal(100, 11_000, 11_000, profit_only=True)


def test_withdrawal_waits_for_cash_tied_in_positions(funds):
    fid, chk = funds.request_withdrawal(3_000, equity=10_500, cash=1_000)  # most money is in a position
    assert not chk.ready and funds.get(fid)["status"] == "requested"
    funds.refresh_ready(cash=2_000)
    assert funds.get(fid)["status"] == "requested"
    funds.refresh_ready(cash=3_500)  # position closed
    assert funds.get(fid)["status"] == "ready"


def test_confirm_cancel_states(funds):
    fid, _ = funds.request_withdrawal(100, 10_000, 10_000)
    funds.cancel_withdrawal(fid)
    assert funds.reserved() == 0
    with pytest.raises(FundsError):
        funds.confirm_withdrawal(fid)
    fid2, _ = funds.request_withdrawal(100, 10_000, 10_000)
    funds.confirm_withdrawal(fid2, "kraken-ref")
    assert funds.get(fid2)["status"] == "completed" and funds.reserved() == 0
    with pytest.raises(FundsError):
        funds.cancel_withdrawal(fid2)


def test_withdrawal_is_not_a_loss_for_the_risk_manager(store, funds):
    """Taking £3,000 out of a £10,000 account must not trip the 3% daily-loss or drawdown halts."""
    risk = RiskManager(RiskConfig(max_daily_loss_pct=0.03), store, Mode.PAPER, clock=lambda: T0)
    pb = PaperBroker(10_000, settings_for(["X"]).costs)
    risk.on_equity(T0, 10_000)
    fid, _ = funds.request_withdrawal(3_000, 10_000, 10_000)
    funds.confirm_withdrawal(fid, ts=T0 + 60)
    funds.apply_pending(pb, risk, ts=T0 + 60)
    assert pb.portfolio.cash == 7_000  # PAPER: simulated payout
    risk.on_equity(T0 + 120, 7_000)
    assert store.get(HALT_KEY) is None
    risk.on_equity(T0 + 180, 6_700)  # a genuine 4.3% trading loss afterwards still halts
    assert store.get(HALT_KEY)["kind"] == "daily_loss"


def test_deposit_is_not_profit_for_twr(funds):
    curve = [{"ts": T0, "equity": 10_000}, {"ts": T0 + 100, "equity": 10_500}]
    funds.deposit(5_000)
    funds.apply_pending(ts=T0 + 150)
    curve += [{"ts": T0 + 200, "equity": 15_500}, {"ts": T0 + 300, "equity": 15_500}]
    assert funds.twr(curve) == pytest.approx(0.05)  # +5% from trading, the deposit adds nothing
    assert funds.summary(15_500, 15_500)["profit"] == pytest.approx(500)


def test_reserved_cash_is_never_traded(store, funds):
    from sentinel.core.trader import Trader
    from sentinel.execution.order_manager import OrderManager

    s = settings_for(["X"])
    pb = PaperBroker(10_000, s.costs, quote_source=lambda sym: (99.9, 100.1))
    risk = RiskManager(s.risk, store, Mode.PAPER, clock=lambda: T0)
    om = OrderManager(pb, risk, s.execution, store, Mode.PAPER)
    trader = Trader(s, pb, om, risk, Mode.PAPER)
    trader.reserved_provider = funds.reserved
    pb.on_quote("X", 99.9, 100.1)
    funds.request_withdrawal(9_995, 10_000, 10_000)
    ctx = trader.context(T0, "X")
    assert ctx.cash == pytest.approx(5) and ctx.equity == pytest.approx(5)
    d = risk.evaluate(OrderIntent("X", Side.BUY, OrderPurpose.ENTRY, 100, stop_price=98), ctx)
    assert d.qty * 100 <= 5 + 1e-9  # can never size into the reserved money
    funds.cancel_withdrawal(funds.flows(("requested", "ready"), "withdrawal")[0]["id"])
    assert trader.context(T0, "X").cash == pytest.approx(10_000)


class FakeVenue:
    id = "kraken"

    def __init__(self):
        self.deposits = [{"id": "D1", "amount": 1_000, "status": "ok", "timestamp": 1_000},
                         {"id": "D2", "amount": 50, "status": "pending", "timestamp": 2_000}]
        self.withdrawals = [{"id": "W1", "amount": 498.5, "fee": {"cost": 1.5}, "status": "ok", "timestamp": 3_000},
                            {"id": "W2", "amount": 200, "status": "ok", "timestamp": 4_000}]

    def fetch_deposits(self, code, since=None):
        return self.deposits

    def fetch_withdrawals(self, code, since=None):
        return self.withdrawals


def test_read_only_sync_records_and_matches(funds):
    req, _ = funds.request_withdrawal(500, 11_000, 11_000)
    r = funds.sync_from_exchange(FakeVenue())
    assert r == {"deposits_added": 1, "withdrawals_matched": 1, "withdrawals_external": 1}
    assert funds.get(req)["status"] == "completed" and funds.get(req)["external_id"] == "wd:W1"
    assert funds.sync_from_exchange(FakeVenue()) == {"deposits_added": 0, "withdrawals_matched": 0,
                                                     "withdrawals_external": 0}  # idempotent
    s = funds.summary(10_400, 10_400)
    assert s["deposits"] == 1_000 and s["withdrawals"] == 700


def test_no_code_path_can_withdraw_from_an_exchange():
    """Sentinel must never call a venue's withdrawal/transfer API (keys with that permission are refused)."""
    pkg = Path(__file__).resolve().parents[1] / "sentinel"
    banned = {"withdraw", "transfer", "create_withdrawal", "privatePostWithdraw"}
    hits = []
    for f in pkg.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in banned:
                hits.append(f"{f.name}:{node.lineno}")
    assert hits == []


def test_export_csv(funds, tmp_path):
    funds.deposit(100)
    p = funds.export_csv(tmp_path / "x.csv")
    rows = list(csv.DictReader(p.open()))
    assert [r["type"] for r in rows] == ["opening", "deposit"]


def test_runner_applies_deposit_and_blocks_reserved(store, tmp_path):
    from sentinel.core.runner import Runner

    df = generate_ohlcv(1100, seed=4)
    s = settings_for(["X"], tmp_path)
    now = [0.0]
    pb = PaperBroker(10_000, s.costs, clock=lambda: now[0])
    run = Runner(s, pb, store, Mode.PAPER, history={"X": df.iloc[:1000]}, clock=lambda: now[0])
    f = Funds(store, "PAPER", s.funds)
    assert f.flows(kind="opening")[0]["amount"] == 10_000
    f.deposit(1_000)
    t = epoch_seconds(df.index[[1000]])[0]
    now[0] = t + 3600
    run.on_bar("X", Bar(t, *df.iloc[1000][["open", "high", "low", "close", "volume"]].to_numpy()))
    assert pb.portfolio.cash == pytest.approx(11_000) or pb.positions()
    assert f.summary(pb.balance()["equity"], pb.balance()["cash"])["net_invested"] == 11_000
    assert run.risk.peak_equity >= 11_000 - 1e-6  # risk references moved with the deposit


def test_dashboard_funds_endpoints(settings, store):
    from sentinel.dashboard.app import create_app, set_password

    set_password(store, "a-long-dashboard-pw")
    store.set("paper.broker_state.PAPER", {"cash": 10_000, "positions": {}, "marks": {}})
    Funds(store, "PAPER", settings.funds).ensure_opening(10_000)
    c = TestClient(create_app(settings, store))
    h = {"Authorization": "Basic " + base64.b64encode(b"admin:a-long-dashboard-pw").decode()}
    assert c.get("/api/funds").status_code == 401
    assert c.post("/api/funds/deposit", json={"amount": 500}, headers=h).status_code == 200
    r = c.post("/api/funds/withdraw", json={"amount": 50_000}, headers=h)
    assert r.status_code == 400 and "available" in r.json()["detail"]
    r = c.post("/api/funds/withdraw", json={"amount": 300, "profit_only": False}, headers=h)
    assert r.status_code == 200 and r.json()["ready"]
    wid = r.json()["id"]
    d = c.get("/api/funds", headers=h).json()
    assert d["reserved_for_withdrawal"] == 300 and d["open_withdrawals"][0]["id"] == wid
    assert c.post(f"/api/funds/{wid}/confirm", headers=h).status_code == 200
    assert c.post(f"/api/funds/{wid}/confirm", headers=h).status_code == 400
    assert c.post(f"/api/funds/{wid}/explode", headers=h).status_code == 404


def test_risk_context_without_funds_unchanged():
    ctx = RiskContext(ts=T0, equity=1, cash=1, positions={}, prices={})
    assert ctx.equity == 1
