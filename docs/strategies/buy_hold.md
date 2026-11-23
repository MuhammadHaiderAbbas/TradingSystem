# Buy & hold benchmark (`buy_hold`)

## Logic
Fully long from the first bar. Every report also computes a true buy-and-hold *benchmark* curve (fees included), and every strategy is compared against it.

Signals are computed at bar close and executed at the next bar's open. Every entry gets a mandatory
2×ATR stop from the risk manager, and size is set by risk (1% of equity by default), not by the strategy.
An automated test proves the signals have no lookahead.

## Parameters (defaults)
`none`. Ranges are declared in `sentinel/strategies/builtin.py`; see `sentinel strategies`.

## Backtest results (synthetic demo data, net of costs)

| Period | Return | Notes |
|---|---:|---|
| Development (in-sample) | -1.6% | buy & hold -65.6%, Sharpe -1.33, max DD -1.6%, 2 trades |
| Holdout (untouched) | -2.1% | buy & hold -11.3%, 2 trades |

These numbers come from `scripts/generate_performance.py` on **synthetic** data, and they show
the strategy has no edge there after costs. Re-run on your real data with `sentinel backtest
--strategy buy_hold --walk-forward --sensitivity <a>,<b>`.

## Known weaknesses
As a *tradable* strategy it gets a mandatory stop, is stopped out, and then never re-enters (the signal never resets). Use the benchmark column instead.
