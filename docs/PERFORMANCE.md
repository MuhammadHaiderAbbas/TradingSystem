# Performance Report

> **Read this first.** These results are from a **synthetic, regime-switching random walk** bundled for the demo (`sentinel.data.synthetic.demo_universe`, 2 assets x 2 years of 1h bars). It has no real edge to find, so the honest expectation is that baseline strategies **lose money after costs** - and they do. Use this page as a template for your own runs on real data. Past performance does not guarantee future results; you can lose capital.

Regenerate: `python scripts/generate_performance.py` (last run 2026-11-23). Costs: 0.10% taker/maker fee, 2 bps spread, 2 bps + 2% of bar range slippage, funding on shorts. Risk manager on (1% risk/trade, 2xATR stop, 15% max-drawdown halt).

## Development period (in-sample, default parameters)

| Strategy | Return | Buy & hold | Sharpe | Max DD | Trades | Win rate | Profit factor | Fees paid | Halts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| buy_hold | -1.6% | -65.6% | -1.33 | -1.6% | 2 | 0.0% | 0.00 | 20 | 0 |
| ema_adx | -14.7% | -65.6% | -1.22 | -15.0% | 44 | 15.9% | 0.42 | 395 | 8 |
| rsi_bb | -14.6% | -65.6% | -1.09 | -15.9% | 144 | 52.8% | 0.77 | 1151 | 3 |
| donchian | -15.1% | -65.6% | -0.74 | -15.1% | 59 | 30.5% | 0.56 | 516 | 6 |
| vol_momentum | -15.1% | -65.6% | -1.55 | -15.6% | 79 | 19.0% | 0.38 | 717 | 11 |
| supertrend_mtf | -14.2% | -65.6% | -1.67 | -15.0% | 54 | 11.1% | 0.31 | 493 | 9 |

## Walk-forward (out-of-sample, parameters re-optimised on each train window)

| Strategy | Mean IS Sharpe | Mean OOS Sharpe | Stitched OOS return |
|---|---:|---:|---:|
| ema_adx | 0.48 | -0.95 | -17.7% |
| rsi_bb | 1.20 | 0.47 | 5.9% |
| donchian | 0.04 | -0.78 | -17.6% |
| vol_momentum | -0.48 | -1.84 | -20.0% |
| supertrend_mtf | -0.85 | -2.70 | -39.1% |

A large gap between in-sample and out-of-sample Sharpe is the signature of overfitting.

## Holdout (final 20%, never used for tuning, evaluated once)

| Strategy | Return | Buy & hold | Sharpe | Max DD | Trades |
|---|---:|---:|---:|---:|---:|
| buy_hold | -2.1% | -11.3% | -1.55 | -2.9% | 2 |
| ema_adx | 1.9% | -11.3% | 0.32 | -12.7% | 53 |
| rsi_bb | -6.0% | -11.3% | -1.67 | -9.0% | 44 |
| donchian | -15.9% | -11.3% | -2.58 | -16.8% | 41 |
| vol_momentum | -15.2% | -11.3% | -2.41 | -15.8% | 87 |
| supertrend_mtf | -8.9% | -11.3% | -0.96 | -14.9% | 80 |

## Monte Carlo (bootstrap of trades, 1,000 runs, development period)

| Strategy | 5th pct return | 95th pct return |
|---|---:|---:|
| buy_hold | -1.7% | -1.5% |
| ema_adx | -23.6% | -2.9% |
| rsi_bb | -28.5% | 0.9% |
| donchian | -25.8% | -0.4% |
| vol_momentum | -22.4% | -5.4% |
| supertrend_mtf | -20.1% | -7.3% |

## Losing periods: return by market regime (development period)

| Strategy | Bull | Bear | Sideways | Crash |
|---|---:|---:|---:|---:|
| buy_hold | 0.0% | 0.0% | -1.6% | 0.0% |
| ema_adx | 0.0% | -4.7% | -10.5% | 0.0% |
| rsi_bb | 5.6% | -2.6% | -8.1% | -9.6% |
| donchian | 0.8% | 4.2% | -19.2% | 0.0% |
| vol_momentum | 0.0% | -2.9% | -12.5% | 0.0% |
| supertrend_mtf | 0.0% | 0.0% | -14.2% | 0.0% |

## What this tells you

* On data with no structure, costs dominate: every active strategy pays more in fees and slippage than it captures. That is the correct, expected outcome, and a good sanity check that the backtester isn't leaking future information.
* `buy_hold` as a *tradable strategy* is stopped out by the mandatory stop-loss and waits for its signal to reset (which never happens). The *benchmark* column is the true buy-and-hold.
* The risk manager's drawdown halt caps losses at roughly the configured 15%, then keeps the strategy flat. That's why losses cluster around -15% instead of compounding further.
* Before trading real money, run the same pipeline on real data, require a positive out-of-sample and holdout result, then paper trade for 8+ weeks and pass `sentinel gate`.
