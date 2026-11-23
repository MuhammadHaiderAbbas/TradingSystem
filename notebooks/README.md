# Notebooks

Research scratch space. Load data with `OHLCVStore("var/market.db").load(symbol, "1h")` or
`sentinel.data.synthetic.demo_universe()`, and always evaluate ideas with
`sentinel.backtest.robustness` (walk-forward and holdout), never by eyeballing in-sample charts.
Don't touch the holdout while exploring.
