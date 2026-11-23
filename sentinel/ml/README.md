# ML: meta-labelling filter (off by default)

Triple-barrier labels on baseline entries → features at the signal bar → purged and embargoed K-fold → isotonic calibration → threshold.
`pipeline.train_and_evaluate` backtests the baseline against the filtered strategy on an unseen segment. The model is deployed only if it beats the baseline out-of-sample after costs **and** scores at least as well as the current model (`registry.py`).
`meta.drift_report` computes PSI per feature. `sentiment.py` is an optional local Ollama scorer that outputs a structured score only and never trades.

Verify: `pytest tests/test_ml.py`; `sentinel ml train --demo`.
