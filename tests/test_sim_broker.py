"""Order types, partial fills, exchange rules and funding in the simulated broker."""

import pytest

from sentinel.config import CostConfig, RiskConfig
from sentinel.core.types import Bar, MarketRules, Order, OrderPurpose, OrderStatus, OrderType, Side
from sentinel.execution.paper import PaperBroker
from sentinel.execution.precision import round_price, round_qty, rules_from_ccxt_market, validate
from sentinel.execution.sim import SimBroker
from sentinel.risk.manager import RiskManager

COSTS = CostConfig(maker_fee=0.0005, taker_fee=0.001, spread_bps=0, slippage_bps=0, slippage_vol_k=0,
                   funding_rate_8h=0.0003, max_participation=1.0)


def broker(cash=10_000.0, costs=COSTS, rules=None):
    b = SimBroker(cash, costs, rules or MarketRules(), bar_seconds=3600)
    risk = RiskManager(RiskConfig(), mode=__import__("sentinel.core.types", fromlist=["Mode"]).Mode.BACKTEST,
                       clock=lambda: b.clock)
    b.bind_risk(risk)
    return b, risk


def place(b, risk, order):
    from sentinel.risk.manager import EXIT_PURPOSES

    ap = (risk.approve_exit(order.symbol, order.side, order.qty, order.purpose) if order.purpose in EXIT_PURPOSES
          else risk._issue(order.symbol, order.side, order.qty, order.purpose))
    return b.place_order(order, ap)


def bar(ts, o, h, lo, c, v=1e9):
    return Bar(ts, o, h, lo, c, v)


def test_market_fills_next_bar_open_with_taker_fee():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    o = place(b, r, Order("X", Side.BUY, 10))
    assert o.status is OrderStatus.OPEN and b.positions() == {}
    b.on_bar("X", bar(3600, 102, 103, 101, 102))
    assert o.status is OrderStatus.FILLED and o.avg_fill_price == 102
    assert o.fees_paid == pytest.approx(10 * 102 * 0.001)
    assert b.portfolio.cash == pytest.approx(10_000 - 1020 - 1.02)


def test_limit_requires_trade_through_and_pays_maker():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    o = place(b, r, Order("X", Side.BUY, 1, OrderType.LIMIT, limit_price=98))
    b.on_bar("X", bar(3600, 100, 101, 98, 99))   # touches 98 only: no fill
    assert o.status is OrderStatus.OPEN
    b.on_bar("X", bar(7200, 99, 100, 97.5, 99))  # trades through
    assert o.status is OrderStatus.FILLED and o.avg_fill_price == 98
    assert o.fees_paid == pytest.approx(98 * 0.0005)


def test_marketable_limit_fills_at_open():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    o = place(b, r, Order("X", Side.BUY, 1, OrderType.LIMIT, limit_price=105))
    b.on_bar("X", bar(3600, 101, 102, 100, 101))
    assert o.avg_fill_price == 101


def test_stop_and_stop_limit():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    o = place(b, r, Order("X", Side.BUY, 1, OrderType.STOP, stop_price=103))
    b.on_bar("X", bar(3600, 100, 104, 99, 103.5))
    assert o.status is OrderStatus.FILLED and o.avg_fill_price == 103
    sl = place(b, r, Order("X", Side.BUY, 1, OrderType.STOP_LIMIT, stop_price=106, limit_price=106.5))
    b.on_bar("X", bar(7200, 104, 107, 104, 106))  # triggers -> becomes a resting limit next bar
    assert sl.status is OrderStatus.OPEN and sl._triggered
    b.on_bar("X", bar(10800, 107, 108, 106, 107))  # open above limit, trades through 106.5
    assert sl.status is OrderStatus.FILLED and sl.avg_fill_price == 106.5


def test_bracket_stop_wins_over_take_profit_same_bar_and_oco_cancels():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    e = place(b, r, Order("X", Side.BUY, 1, stop_loss=95, take_profit=110))
    b.on_bar("X", bar(3600, 100, 101, 99, 100))
    assert b.positions() == {"X": 1}
    kids = {o.type: o for o in b.open_orders("X")}
    assert set(kids) == {OrderType.STOP, OrderType.TAKE_PROFIT}
    b.on_bar("X", bar(7200, 100, 111, 94, 100))  # both levels touched: conservative -> stop
    assert b.positions() == {}
    assert kids[OrderType.STOP].status is OrderStatus.FILLED
    assert kids[OrderType.TAKE_PROFIT].status is OrderStatus.CANCELED
    assert b.portfolio.closed_trades[-1]["exit_reason"] == "stop_loss"
    assert e.status is OrderStatus.FILLED


def test_take_profit_fills_when_only_tp_touched():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    place(b, r, Order("X", Side.BUY, 1, stop_loss=95, take_profit=110))
    b.on_bar("X", bar(3600, 100, 101, 99, 100))
    b.on_bar("X", bar(7200, 105, 111, 104, 109))
    t = b.portfolio.closed_trades[-1]
    assert t["exit_reason"] == "take_profit" and t["exit_price"] == 110
    assert b.open_orders() == []


def test_entry_bar_stop_is_live_same_bar():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    place(b, r, Order("X", Side.BUY, 1, stop_loss=97))
    b.on_bar("X", bar(3600, 100, 100.5, 96, 98))  # enters at open, then drops through the stop
    assert b.positions() == {} and b.portfolio.closed_trades[-1]["exit_price"] == 97


def test_trailing_stop_ratchets():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    place(b, r, Order("X", Side.BUY, 1, trail_pct=0.05))
    b.on_bar("X", bar(3600, 100, 101, 99.5, 100))       # entry @100, trail ref -> 101
    b.on_bar("X", bar(7200, 101, 120, 100, 119))         # ref -> 120 (stop 114)
    assert b.positions() == {"X": 1}
    b.on_bar("X", bar(10800, 118, 118, 113, 114))        # 114 hit
    t = b.portfolio.closed_trades[-1]
    assert t["exit_price"] == pytest.approx(120 * 0.95)


def test_partial_fills_respect_participation():
    costs = COSTS.model_copy(update={"max_participation": 0.1})
    b, r = broker(costs=costs)
    b.on_bar("X", bar(0, 100, 101, 99, 100, v=1000))
    o = place(b, r, Order("X", Side.BUY, 250))
    b.on_bar("X", bar(3600, 100, 101, 99, 100, v=1000))
    assert o.status is OrderStatus.PARTIALLY_FILLED and o.filled_qty == 100
    b.on_bar("X", bar(7200, 100, 101, 99, 100, v=1000))
    b.on_bar("X", bar(10800, 100, 101, 99, 100, v=0))     # zero volume: nothing fills
    assert o.filled_qty == 200
    b.on_bar("X", bar(14400, 100, 101, 99, 100, v=1000))
    assert o.status is OrderStatus.FILLED


def test_exchange_rules_rounding_and_rejections():
    rules = MarketRules(min_qty=0.001, step_size=0.001, tick_size=0.01, min_notional=10)
    assert round_qty(0.12345, rules) == 0.123
    assert round_price(101.234, rules) == 101.23
    assert validate(0.0005, 100, rules) and validate(0.05, 100, rules)
    assert validate(0.2, 100, rules) is None
    b, r = broker(rules=rules)
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    o = place(b, r, Order("X", Side.BUY, 0.05))  # $5 < min notional
    assert o.status is OrderStatus.REJECTED and "min_notional" in o.reject_reason
    o2 = place(b, r, Order("X", Side.BUY, 0.12345))
    assert o2.qty == 0.123


def test_ccxt_market_rules_parsing():
    m = {"precision": {"amount": 0.0001, "price": 0.01}, "limits": {"amount": {"min": 0.0001}, "cost": {"min": 5}}}
    assert rules_from_ccxt_market(m) == MarketRules(0.0001, 0.0001, 0.01, 5.0)
    m2 = {"precision": {"amount": 3, "price": 2}, "limits": {}}
    r = rules_from_ccxt_market(m2)
    assert r.step_size == 0.001 and r.tick_size == 0.01


def test_reduce_only_cannot_open_or_flip():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    o = place(b, r, Order("X", Side.SELL, 1, purpose=OrderPurpose.EXIT, reduce_only=True))
    assert o.status is OrderStatus.REJECTED


def test_funding_charged_on_shorts():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    place(b, r, Order("X", Side.SELL, 10))
    b.on_bar("X", bar(3600, 100, 101, 99, 100))
    cash_before = b.portfolio.cash
    b.on_bar("X", bar(7200, 100, 101, 99, 100))
    assert cash_before - b.portfolio.cash == pytest.approx(10 * 100 * 0.0003 / 8)


def test_paper_broker_fills_immediately_at_live_quote():
    pb = PaperBroker(10_000, COSTS.model_copy(update={"slippage_bps": 10}), quote_source=lambda s: (99.0, 101.0))
    r = RiskManager(RiskConfig())
    pb.bind_risk(r)
    o = pb.place_order(Order("X", Side.BUY, 1), r._issue("X", Side.BUY, 1, OrderPurpose.ENTRY))
    assert o.status is OrderStatus.FILLED and o.avg_fill_price == pytest.approx(101 * 1.001)
    lim = pb.place_order(Order("X", Side.SELL, 1, OrderType.LIMIT, limit_price=105, purpose=OrderPurpose.TAKE_PROFIT,
                               reduce_only=True), r.approve_exit("X", Side.SELL, 1, OrderPurpose.TAKE_PROFIT))
    pb.on_quote("X", 104, 106)
    assert lim.status is OrderStatus.OPEN
    pb.on_quote("X", 105.5, 106)
    assert lim.status is OrderStatus.FILLED and pb.positions() == {}


def test_state_roundtrip_for_crash_recovery():
    b, r = broker()
    b.on_bar("X", bar(0, 100, 101, 99, 100))
    place(b, r, Order("X", Side.BUY, 2, stop_loss=90))
    b.on_bar("X", bar(3600, 100, 101, 99, 100))
    st = b.to_state()
    b2, _ = broker()
    b2.restore_state(st)
    assert b2.positions() == b.positions()
    assert [o.client_id for o in b2.open_orders()] == [o.client_id for o in b.open_orders()]
    b2.on_bar("X", bar(7200, 95, 96, 85, 86))  # restored stop still protects
    assert b2.positions() == {}
