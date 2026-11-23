# Backtest

* `engine.py`: event-driven, bar-by-bar, using the same Trader, Risk, OrderManager and SimBroker code as live.
* `metrics.py`: returns, Sharpe/Sortino/Calmar, drawdown and its duration, trade stats, turnover, buy & hold benchmark.
* `robustness.py`: holdout split and reuse tracking, walk-forward (grid or optuna), Monte Carlo, sensitivity heatmaps, market-period and per-symbol breakdowns, overfitting warnings.
* `report.py`: a self-contained HTML report.

Verify: `pytest tests/test_backtest_engine.py tests/test_robustness.py`; `sentinel backtest --demo --walk-forward`.
