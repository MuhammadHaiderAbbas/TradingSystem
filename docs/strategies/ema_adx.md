# EMA crossover with ADX trend filter (`ema_adx`)

## Logic
Long when the fast EMA is above the slow EMA *and* ADX(n) ≥ `adx_min` at entry; exit when the fast EMA crosses back below. Regime filter: trend only.

Signals are computed at bar close and executed at the next bar's open. Every entry gets a mandatory
2×ATR stop from the risk manager, and size is set by risk (1% of equity by default), not by the strategy.
An automated test proves the signals have no lookahead.

## Parameters (defaults)
`fast=12, slow=48, adx_n=14, adx_min=20`. Ranges are declared in `sentinel/strategies/builtin.py`; see `sentinel strategies`.

## Backtest results (synthetic demo data, net of costs)

| Period | Return | Notes |
|---|---:|---|
| Development (in-sample) | -14.7% | buy & hold -65.6%, Sharpe -1.22, max DD -15.0%, 44 trades |
| Walk-forward OOS (stitched) | -17.7% | mean OOS Sharpe -0.95 vs IS 0.48 |
| Holdout (untouched) | 1.9% | buy & hold -11.3%, 53 trades |

These numbers come from `scripts/generate_performance.py` on **synthetic** data, and they show
the strategy has no edge there after costs. Re-run on your real data with `sentinel backtest
--strategy ema_adx --walk-forward --sensitivity <a>,<b>`.

## Known weaknesses
Whipsaws in ranges (the ADX filter only partly helps); ADX lags, so entries come late; a 2×ATR stop on hourly bars is often hit by noise.
