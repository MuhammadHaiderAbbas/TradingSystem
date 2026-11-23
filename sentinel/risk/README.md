# Risk manager

Every order passes `RiskManager.evaluate`. Approvals are HMAC-signed, single-use, qty-budgeted and expire after 5 minutes, and the broker base class verifies them.
Checks: kill/halt, stale data, extreme volatility, mandatory stop on the correct side, max open positions, no pyramiding, correlation, sizing (fixed fractional / ATR / capped fractional Kelly), per-asset, total and leverage exposure caps, cash, exchange minimums.
Account guards: daily loss, weekly loss, drawdown, consecutive losses, API error bursts → halt. A halt needs a manual re-enable after the cooldown. Every veto is logged.

Verify: `pytest tests/test_risk.py tests/test_no_bypass.py`.
