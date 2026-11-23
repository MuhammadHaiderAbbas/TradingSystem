"""Order manager idempotency/retries, reconciliation, kill switch, smart execution, recovery."""

import pandas as pd
import pytest

from sentinel.config import CostConfig, ExecutionConfig, RiskConfig, StrategyEntry
from sentinel.core.errors import TransientBrokerError
from sentinel.core.modes import HALT_KEY, KILL_KEY
from sentinel.core.runner import Runner
from sentinel.core.types import Bar, Mode, Order, OrderPurpose, OrderStatus, Side
from sentinel.data.synthetic import generate_ohlcv
from sentinel.data.timeframes import epoch_seconds
from sentinel.execution.order_manager import OrderManager, client_id_for
from sentinel.execution.paper import PaperBroker
from sentinel.execution.reconcile import diff_positions, reconcile
from sentinel.execution.sim import SimBroker
from sentinel.execution.smart import SmartExecutor, twap_slices
from sentinel.risk.manager import OrderIntent, RiskContext, RiskManager
from tests.conftest import settings_for

COSTS = CostConfig(spread_bps=0, slippage_bps=1, slippage_vol_k=0)


def setup(store=None, mode=Mode.PAPER, broker=None):
    b = broker or SimBroker(10_000, COSTS)
    clock = [1_700_000_000.0]
    risk = RiskManager(RiskConfig(), store, mode, clock=lambda: clock[0])
    om = OrderManager(b, risk, ExecutionConfig(backoff_base_s=0), store, mode, initial_cash=10_000, sleep=lambda s: None)
    b.on_bar("X", Bar(clock[0], 100, 101, 99, 100, 1e9)) if isinstance(b, SimBroker) else None
    return b, risk, om, clock


def ctx(b, ts=1_700_000_000.0):
    return RiskContext(ts=ts, equity=b.balance()["equity"], cash=b.balance()["cash"], positions=b.positions(),
                       prices={"X": 100.0})


def intent(key="k1"):
    return OrderIntent("X", Side.BUY, OrderPurpose.ENTRY, 100.0, stop_price=98.0, key=key, strategy="t")


def test_idempotent_client_ids(store):
    b, risk, om, _ = setup(store)
    o1 = om.submit(intent("same"), ctx(b))
    o2 = om.submit(intent("same"), ctx(b))
    assert o1 is o2 and len(b.all_orders) == 1
    assert o1.client_id == client_id_for("same")
    # survives a restart: a fresh order manager on the same store won't duplicate
    b2, risk2, om2, _ = setup(store)
    om2.recover()
    assert om2.submit(intent("same"), ctx(b2)) is None and len(b2.all_orders) == 0


class FlakyBroker(SimBroker):
    """Fails transiently; on the 2nd failure the order actually landed (lost response)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.attempts = 0

    def _submit(self, order):
        self.attempts += 1
        if self.attempts == 1:
            raise TransientBrokerError("timeout")
        if self.attempts == 2:
            super()._submit(order)
            raise TransientBrokerError("connection reset after send")
        return super()._submit(order)


def test_retry_without_duplicates(store):
    b = FlakyBroker(10_000, COSTS)
    _, risk, om, _ = setup(store, broker=b)
    o = om.submit(intent("retry"), ctx(b))
    assert o is not None and b.attempts == 2
    assert len([x for x in b.all_orders.values() if x.client_id == o.client_id]) == 1
    assert om.api_errors == 2


def test_order_state_persisted(store):
    b, risk, om, clock = setup(store)
    o = om.submit(intent("p"), ctx(b))
    assert store.get_order(o.client_id)["status"] == "open"
    b.on_bar("X", Bar(clock[0] + 3600, 100, 101, 99, 100, 1e9))
    assert store.get_order(o.client_id)["status"] == "filled"
    assert om.ledger.qty("X") == pytest.approx(b.positions()["X"])


def test_reconciliation_mismatch_halts(store):
    b, risk, om, clock = setup(store)
    om.submit(intent("r"), ctx(b))
    b.on_bar("X", Bar(clock[0] + 3600, 100, 101, 99, 100, 1e9))
    internal = {s: p.qty for s, p in om.ledger.open_positions().items()}
    assert reconcile(internal, b.positions(), store, risk, Mode.PAPER) == []
    venue = dict(b.positions())
    venue["X"] *= 0.5  # someone traded manually on the account
    venue["Y"] = 3.0
    mism = reconcile(internal, venue, store, risk, Mode.PAPER)
    assert {m.symbol for m in mism} == {"X", "Y"}
    assert store.get(HALT_KEY)["kind"] == "reconciliation"
    assert len(store.unresolved_recon_errors()) == 1
    assert diff_positions({"A": 1.0}, {"A": 1.0 + 1e-9}) == []


def test_kill_switch_cancels_closes_and_locks(store):
    b, risk, om, clock = setup(store, broker=PaperBroker(10_000, COSTS, quote_source=lambda s: (99.9, 100.1)))
    b.on_bar("X", Bar(clock[0], 100, 101, 99, 100, 1e9))
    om.submit(intent("kill"), ctx(b))
    assert b.positions()  # paper broker filled at the quote
    om.submit(OrderIntent("X", Side.SELL, OrderPurpose.EXIT, 100, qty=1), ctx(b))  # some other open order
    res = om.kill(close_positions=True)
    assert b.positions() == {} and b.open_orders() == []
    assert res["closed"] == ["X"] and store.get(KILL_KEY)
    assert not om.submit(intent("after-kill"), ctx(b))  # locked


def test_twap_and_limit_first_fallback():
    cfg = ExecutionConfig(twap_threshold_notional=1000, twap_slices=4, limit_timeout_s=1)
    assert twap_slices(20, 100, cfg) == [5.0] * 4
    assert twap_slices(5, 10, cfg) == [5]
    pb = PaperBroker(100_000, COSTS, quote_source=lambda s: (100.0, 100.2))
    risk = RiskManager(RiskConfig())
    pb.bind_risk(risk)
    now = [0.0]
    ex = SmartExecutor(pb, cfg, pb.quote_source, sleep=lambda s: now.__setitem__(0, now[0] + s), now=lambda: now[0])
    parent = Order("X", Side.BUY, 20)
    ap = risk._issue("X", Side.BUY, 20, OrderPurpose.ENTRY)
    rep = ex.execute(parent, ap)
    # passive limits never fill (quote doesn't move) -> each slice falls back to market
    assert rep.used_fallback and rep.filled_qty == pytest.approx(20)
    assert pb.positions()["X"] == pytest.approx(20)
    # the parent approval's qty budget is exhausted: splitting can't exceed what risk approved
    assert not risk.verify_and_consume(ap, Order("X", Side.BUY, 0.01))


def test_runner_paper_crash_recovery(store, tmp_path):
    df = generate_ohlcv(1300, seed=9)
    s = settings_for(["X"], tmp_path, strategies=[StrategyEntry(name="ema_adx", params={"adx_min": 10.0})])
    hist, live = df.iloc[:1000], df.iloc[1000:]
    now = [0.0]

    def make():
        b = PaperBroker(10_000, s.costs, clock=lambda: now[0])
        return b, Runner(s, b, store, Mode.PAPER, history={"X": hist}, clock=lambda: now[0])

    b, run = make()
    ts = epoch_seconds(live.index)
    rows = live[["open", "high", "low", "close", "volume"]].to_numpy()
    half = 150
    for t, r in zip(ts[:half], rows[:half], strict=True):
        now[0] = t + 3600
        run.on_bar("X", Bar(t, *r))
    snapshot_positions = b.positions()
    snapshot_orders = sorted(o.client_id for o in b.open_orders())
    del b, run  # "crash"
    b2, run2 = make()
    assert b2.positions() == snapshot_positions
    assert sorted(o.client_id for o in b2.open_orders()) == snapshot_orders
    run2.on_bar("X", Bar(ts[half - 1], *rows[half - 1]))  # replayed bar after restart is ignored
    for t, r in zip(ts[half:], rows[half:], strict=True):
        now[0] = t + 3600
        run2.on_bar("X", Bar(t, *r))
    run2.maybe_reconcile(force=True)
    assert store.unresolved_recon_errors() == []
    assert len(store.equity_curve("PAPER")) == len(live)


def test_runner_executes_dashboard_kill(store, tmp_path):
    df = generate_ohlcv(1100, seed=2)
    s = settings_for(["X"], tmp_path)
    b = PaperBroker(10_000, s.costs)
    run = Runner(s, b, store, Mode.PAPER, history={"X": df.iloc[:1000]})
    store.set(KILL_KEY, {"ts": 0, "reason": "dashboard", "close_positions": True, "executed": False, "test": True})
    t = epoch_seconds(df.index[[1000]])[0]
    run.on_bar("X", Bar(t, *df.iloc[1000][["open", "high", "low", "close", "volume"]].to_numpy()))
    assert store.get(KILL_KEY)["executed"] and store.get("kill.tested")
    assert isinstance(pd.Timestamp.now(), pd.Timestamp)
    assert all(o.status in (OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED)
               for o in b.all_orders.values())
