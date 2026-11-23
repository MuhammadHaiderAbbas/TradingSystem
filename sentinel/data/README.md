# Data: ingestion, quality, storage

* `downloader.py`: ccxt OHLCV with pagination, rate-limit backoff and resume. The forming candle is never stored.
* `live_feed.py`: WebSocket (ccxt.pro) with reconnect, a REST polling fallback, and replay for demos. All feeds emit closed bars only and keep a heartbeat.
* `quality.py`: gaps, duplicates, misalignment, outliers (robust z), zero volume, invalid OHLC, future bars. `repair` fills short gaps and flags rows instead of hiding problems.
* `storage.py`: SQLite `WITHOUT ROWID` time-indexed table (UTC ms), funding and order-book tables, Parquet export.
* `resample.py`: complete bins only; `align_higher_tf` maps higher-TF values without lookahead.
* `synthetic.py`: a seeded regime-switching dataset for the demo and tests (**not real data**).

Verify: `pytest tests/test_data.py`; `sentinel data report`.
