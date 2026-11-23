# Features: indicators, registry, lookahead tests

All indicators are causal. They are validated against pandas-ta to ~1e-12 after warm-up, plus an independent reference implementation for Supertrend.
`registry.py` lets strategies declare `FeatureRequest("ema", n=12, timeframe="4h")`.
`lookahead.py` runs a truncation test and a future-perturbation test on any function; the test suite runs it on every feature and every strategy.

Verify: `pytest tests/test_features.py`.
