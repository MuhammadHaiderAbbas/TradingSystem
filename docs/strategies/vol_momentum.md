# Volatility-adjusted time-series momentum (`vol_momentum`)

## Logic
Momentum = log return over `lookback` bars divided by realised volatility scaled to the horizon. Hold when |z| ≥ `threshold`, with exposure scaled by `target_vol / vol` in 25% steps.

Signals are computed at bar close and executed at the next bar's open. Every entry gets a mandatory
2×ATR stop from the risk manager, and size is set by risk (1% of equity by default), not by the strategy.
An automated test proves the signals have no lookahead.

## Parameters (defaults)
`lookback=72, vol_n=48, target_vol=0.01, threshold=0.5`. Ranges are declared in `sentinel/strategies/builtin.py`; see `sentinel strategies`.

## Backtest results (synthetic demo data, net of costs)

| Period | Return | Notes |
|---|---:|---|
| Development (in-sample) | -15.1% | buy & hold -65.6%, Sharpe -1.55, max DD -15.6%, 79 trades |
| Walk-forward OOS (stitched) | -20.0% | mean OOS Sharpe -1.84 vs IS -0.48 |
| Holdout (untouched) | -15.2% | buy & hold -11.3%, 87 trades |

These numbers come from `scripts/generate_performance.py` on **synthetic** data, and they show
the strategy has no edge there after costs. Re-run on your real data with `sentinel backtest
--strategy vol_momentum --walk-forward --sensitivity <a>,<b>`.

## Known weaknesses
Momentum on hourly bars is mostly noise; the reversals after sharp moves hurt; position size only changes at entry.
