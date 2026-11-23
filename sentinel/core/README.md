# Core: types, modes, state, trader, runner

* `types.py`: Order, Fill, Bar, MarketRules, enums.
* `modes.py`: BACKTEST/PAPER/LIVE. LIVE requires a recorded gate pass for the current config fingerprint, no kill or halt, and the confirmation phrase.
* `state.py`: the shared SQLite (WAL) store: kv, events and vetoes, orders, fills, trades, equity, heartbeats, reconciliation errors.
* `trader.py`: signal → order intents (exits on sign change, entries with a mandatory ATR stop). The same class runs backtest, paper and live.
* `runner.py`: the paper/live loop: bar history, signals, kill-switch polling, reconciliation, crash recovery.

Verify: `pytest tests/test_execution.py tests/test_gate.py`.
