# RSI mean reversion with Bollinger confirmation (`rsi_bb`)

## Logic
Long when RSI < `rsi_low` **and** close < lower Bollinger band; exit at the middle band or when RSI > `rsi_high`. Regime filter: range only.

Signals are computed at bar close and executed at the next bar's open. Every entry gets a mandatory
2×ATR stop from the risk manager, and size is set by risk (1% of equity by default), not by the strategy.
An automated test proves the signals have no lookahead.

## Parameters (defaults)
`rsi_n=14, rsi_low=30, rsi_high=70, bb_n=20, bb_k=2.0`. Ranges are declared in `sentinel/strategies/builtin.py`; see `sentinel strategies`.

## Backtest results (synthetic demo data, net of costs)

| Period | Return | Notes |
|---|---:|---|
| Development (in-sample) | -14.6% | buy & hold -65.6%, Sharpe -1.09, max DD -15.9%, 144 trades |
| Walk-forward OOS (stitched) | 5.9% | mean OOS Sharpe 0.47 vs IS 1.20 |
| Holdout (untouched) | -6.0% | buy & hold -11.3%, 44 trades |

These numbers come from `scripts/generate_performance.py` on **synthetic** data, and they show
the strategy has no edge there after costs. Re-run on your real data with `sentinel backtest
--strategy rsi_bb --walk-forward --sensitivity <a>,<b>`.

## Known weaknesses
Catches falling knives in trends; frequent trading makes it very cost-sensitive (the most fees of all baselines).
