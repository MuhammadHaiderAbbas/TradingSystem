# Donchian breakout (turtle-style) (`donchian`)

## Logic
Long when the close breaks above the previous `entry_n`-bar high; exit below the previous `exit_n`-bar low. The channel is lagged one bar so today's high can't trigger today's breakout.

Signals are computed at bar close and executed at the next bar's open. Every entry gets a mandatory
2×ATR stop from the risk manager, and size is set by risk (1% of equity by default), not by the strategy.
An automated test proves the signals have no lookahead.

## Parameters (defaults)
`entry_n=55, exit_n=20`. Ranges are declared in `sentinel/strategies/builtin.py`; see `sentinel strategies`.

## Backtest results (synthetic demo data, net of costs)

| Period | Return | Notes |
|---|---:|---|
| Development (in-sample) | -15.1% | buy & hold -65.6%, Sharpe -0.74, max DD -15.1%, 59 trades |
| Walk-forward OOS (stitched) | -17.6% | mean OOS Sharpe -0.78 vs IS 0.04 |
| Holdout (untouched) | -15.9% | buy & hold -11.3%, 41 trades |

These numbers come from `scripts/generate_performance.py` on **synthetic** data, and they show
the strategy has no edge there after costs. Re-run on your real data with `sentinel backtest
--strategy donchian --walk-forward --sensitivity <a>,<b>`.

## Known weaknesses
Low win rate by design: it depends on rare large trends. False breakouts in sideways markets are its main cost (-19% in sideways periods on the demo data).
