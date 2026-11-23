# Supertrend with 4h EMA filter (example plugin) (`supertrend_mtf`)

## Logic
Long when the 1h Supertrend points up **and** the close is above the 4h EMA. The 4h EMA is computed on completed 4h bars and aligned without lookahead. It lives in `sentinel/strategies/plugins/` as the template for your own strategies.

Signals are computed at bar close and executed at the next bar's open. Every entry gets a mandatory
2×ATR stop from the risk manager, and size is set by risk (1% of equity by default), not by the strategy.
An automated test proves the signals have no lookahead.

## Parameters (defaults)
`st_n=10, st_mult=3.0, htf_ema=50`. Ranges are declared in `sentinel/strategies/builtin.py`; see `sentinel strategies`.

## Backtest results (synthetic demo data, net of costs)

| Period | Return | Notes |
|---|---:|---|
| Development (in-sample) | -14.2% | buy & hold -65.6%, Sharpe -1.67, max DD -15.0%, 54 trades |
| Walk-forward OOS (stitched) | -39.1% | mean OOS Sharpe -2.70 vs IS -0.85 |
| Holdout (untouched) | -8.9% | buy & hold -11.3%, 80 trades |

These numbers come from `scripts/generate_performance.py` on **synthetic** data, and they show
the strategy has no edge there after costs. Re-run on your real data with `sentinel backtest
--strategy supertrend_mtf --walk-forward --sensitivity <a>,<b>`.

## Known weaknesses
Two lagging filters: late entries, and the worst walk-forward result on the demo data.
