# Strategies

`Strategy.generate_signals(data) -> Series` of target exposure in [-1, 1], with declared params, features, timeframe and regimes.
Built-ins: `ema_adx`, `rsi_bb`, `donchian`, `vol_momentum`, `buy_hold`. Drop new files into `plugins/`, a directory listed in `SENTINEL_STRATEGY_PATH`, or the `sentinel.strategies` entry point.
`combine.py` provides the vote/weighted combiner, the regime filter (`entry` mode blocks new positions only; `strict` mode forces flat) and the config-driven `SignalPipeline`.
See `docs/strategies/` for logic, results and weaknesses.
