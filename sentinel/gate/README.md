# LIVE readiness gate

`readiness.py` checks: 8+ weeks of continuous paper trading, 50+ trades, paper within tolerance of the backtest expectation, paper drawdown under the limit, positive expectancy (paper and OOS), clean reconciliation, passing tests, a kill-switch drill, and a safe API key.
A pass is bound to the config fingerprint. The capital cap can be raised at most ×2 per step, at least 4 weeks apart, and only while live results match paper.

Run: `sentinel gate`. Verify: `pytest tests/test_gate.py`.
