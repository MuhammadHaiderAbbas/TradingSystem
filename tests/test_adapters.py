"""ccxt and Alpaca adapters against fake clients (no network)."""

from types import SimpleNamespace

import pytest

from sentinel.config import RiskConfig
from sentinel.core.errors import KeySafetyError, TransientBrokerError
from sentinel.core.types import Mode, Order, OrderPurpose, OrderStatus, OrderType, Side
from sentinel.execution.alpaca_broker import AlpacaBroker
from sentinel.execution.ccxt_broker import CcxtBroker
from sentinel.risk.manager import RiskManager
from sentinel.security.key_safety import enforce_key_safety, fetch_permissions


class NetworkError(Exception):
    pass


class FakeCcxt:
    id = "binance"
    has = {"createStopMarketOrder": False, "fetchOpenOrders": True}

    def __init__(self, withdraw=False, fail_once=False):
        self.withdraw = withdraw
        self.fail_once = fail_once
        self.orders = {}
        self.balance = {"USDT": {"total": 1000.0}, "BTC": {"total": 0.5}}  # 0.5 BTC held before the bot

    def sapi_get_account_apirestrictions(self):
        return {"enableWithdrawals": self.withdraw, "ipRestrict": True, "enableSpotAndMarginTrading": True}

    def load_markets(self):
        return {"BTC/USDT": {"precision": {"amount": 0.0001, "price": 0.01},
                             "limits": {"amount": {"min": 0.0001}, "cost": {"min": 5}}}}

    def amount_to_precision(self, s, a):
        return f"{a:.4f}"

    def price_to_precision(self, s, p):
        return f"{p:.2f}"

    def fetch_ticker(self, s):
        return {"last": 50_000.0, "bid": 49_990.0, "ask": 50_010.0}

    def create_order(self, symbol, typ, side, amount, price, params):
        if self.fail_once:
            self.fail_once = False
            raise NetworkError("reset")
        cid = params.get("newClientOrderId") or params.get("clientOrderId")
        o = {"id": f"x{len(self.orders)}", "clientOrderId": cid, "symbol": symbol, "side": side, "amount": amount,
             "type": typ, "status": "closed", "filled": amount, "average": 50_000.0, "fee": {"cost": 0.5}}
        self.orders[o["id"]] = o
        self.balance["BTC"]["total"] += amount if side == "buy" else -amount
        return o

    def fetch_order(self, oid, symbol):
        return self.orders[oid]

    def fetch_orders(self, symbol, since=None, limit=50):
        return list(self.orders.values())

    def fetch_open_orders(self, symbol):
        return []

    def fetch_balance(self):
        return self.balance

    def cancel_order(self, oid, symbol):
        return {}


def test_ccxt_withdrawal_key_refused():
    with pytest.raises(KeySafetyError):
        enforce_key_safety(fetch_permissions(FakeCcxt(withdraw=True)), Mode.LIVE)
    enforce_key_safety(fetch_permissions(FakeCcxt()), Mode.LIVE)


def test_ccxt_order_flow_baseline_and_client_id(store):
    ex = FakeCcxt()
    b = CcxtBroker(ex, ["BTC/USDT"], store)
    risk = RiskManager(RiskConfig())
    b.bind_risk(risk)
    assert b.positions() == {}  # pre-existing 0.5 BTC is the baseline, not a bot position
    o = Order("BTC/USDT", Side.BUY, 0.01, client_id="snt-abc")
    b.place_order(o, risk._issue("BTC/USDT", Side.BUY, 0.01, OrderPurpose.ENTRY))
    assert o.status is OrderStatus.FILLED and o.exchange_id == "x0"
    assert ex.orders["x0"]["clientOrderId"] == "snt-abc"
    assert b.positions() == {"BTC/USDT": pytest.approx(0.01)}
    assert b.market_rules("BTC/USDT").min_notional == 5
    assert b.get_order(client_id="snt-abc") is o


def test_ccxt_transient_errors_mapped(store):
    b = CcxtBroker(FakeCcxt(fail_once=True), ["BTC/USDT"], store)
    risk = RiskManager(RiskConfig())
    b.bind_risk(risk)
    ap = risk._issue("BTC/USDT", Side.BUY, 0.01, OrderPurpose.ENTRY)
    with pytest.raises(TransientBrokerError):
        b.place_order(Order("BTC/USDT", Side.BUY, 0.01), ap)
    b.place_order(Order("BTC/USDT", Side.BUY, 0.01), ap)  # budget refunded after a failed send


class FakeAlpaca:
    def __init__(self):
        self.submitted = []

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id="a1", status="filled", filled_qty=req.qty, filled_avg_price=190.0)

    def get_all_positions(self):
        return [SimpleNamespace(symbol="AAPL", qty="3")]

    def get_account(self):
        return SimpleNamespace(cash="1000", equity="1570")

    def get_order_by_client_id(self, cid):
        return SimpleNamespace(id="a1", status="filled", filled_qty=3, filled_avg_price=190.0)

    def cancel_orders(self):
        return []


def test_alpaca_adapter_same_interface():
    pytest.importorskip("alpaca")
    fake = FakeAlpaca()
    b = AlpacaBroker(trading_client=fake)
    risk = RiskManager(RiskConfig())
    b.bind_risk(risk)
    o = Order("AAPL", Side.BUY, 3, client_id="snt-1")
    b.place_order(o, risk._issue("AAPL", Side.BUY, 3, OrderPurpose.ENTRY))
    assert o.status is OrderStatus.FILLED and fake.submitted[0].client_order_id == "snt-1"
    assert b.positions() == {"AAPL": 3.0} and b.balance()["equity"] == 1570.0
    assert b.key_permissions().can_withdraw is False


def test_alpaca_positions_without_sdk():
    b = AlpacaBroker(trading_client=FakeAlpaca())
    assert b.positions() == {"AAPL": 3.0}
    assert b.market_rules("AAPL").step_size == 1.0
    assert OrderType.MARKET  # interface parity
