# Execution

* `broker.py`: the unified interface. `place_order` is a non-overridable template that verifies the risk approval.
* `sim.py`: the fill model (next-bar open, spread, slippage, fees, partial fills, all order types, brackets/OCO, funding, exchange rules). `paper.py` adds live-quote fills.
* `ccxt_broker.py`, `alpaca_broker.py`: venue adapters.
* `order_manager.py`: idempotent client ids, retries without duplicates, persistence, the kill switch, protective stops.
* `reconcile.py`: internal ledger vs venue → halt on mismatch. `smart.py`: liquidity check, limit-first with market fallback, TWAP. `precision.py`: tick, lot and min-notional rules.

Verify: `pytest tests/test_execution.py tests/test_sim_broker.py tests/test_adapters.py`.
