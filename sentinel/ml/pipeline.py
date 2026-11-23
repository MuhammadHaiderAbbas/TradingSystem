"""Train -> evaluate -> register -> (maybe) deploy, with the honesty rule built in:

    If ML doesn't beat the baseline out-of-sample after costs, it stays switched off.

The development data (holdout excluded) is split chronologically. The model is trained
only on entries whose labels resolve before the split; the baseline strategy and the
ML-filtered strategy are then backtested on the later, unseen segment with the full
cost model and risk manager.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sentinel.backtest.engine import run_backtest
from sentinel.backtest.robustness import _settings_for, signals_for, split_holdout
from sentinel.config import Settings
from sentinel.core.state import StateStore
from sentinel.features import indicators as ind
from sentinel.ml.meta import build_dataset, drift_report, feature_matrix, filter_signals, train_meta
from sentinel.ml.registry import ModelRegistry
from sentinel.monitoring.logging import get_logger

log = get_logger("ml.pipeline")


def train_and_evaluate(data: dict[str, pd.DataFrame], settings: Settings, strategy: str | None = None,
                       params: dict | None = None, train_frac: float = 0.7, store: StateStore | None = None,
                       deploy: bool = True) -> dict[str, Any]:
    ml = settings.ml
    entry = next(e for e in settings.strategies if e.enabled)
    name, params = strategy or entry.name, params if params is not None else entry.params
    dev, _ = split_holdout(data, settings.backtest.holdout_fraction)
    union = sorted(set().union(*[set(df.index) for df in dev.values()]))
    split = union[int(len(union) * train_frac)]
    sig = signals_for(dev, settings, name, params)
    atr = {s: ind.atr(df["high"], df["low"], df["close"], 14) for s, df in dev.items()}
    cost = 2 * (settings.costs.taker_fee + settings.costs.slippage_bps / 1e4 + settings.costs.spread_bps / 2e4)
    train_data = {s: df[df.index < split] for s, df in dev.items()}
    ds = build_dataset(train_data, {s: sig[s][sig[s].index < split] for s in dev},
                       {s: atr[s][atr[s].index < split] for s in dev}, ml.tp_atr, ml.sl_atr, ml.max_holding_bars, cost)
    mm = train_meta(ds, ml.model, ml.n_splits, ml.embargo_pct, ml.threshold)

    test = {s: df[df.index >= split] for s, df in dev.items()}
    base_sig = {s: sig[s].reindex(test[s].index).fillna(0.0) for s in test}
    filt_sig = {s: filter_signals(sig[s], dev[s], mm).reindex(test[s].index).fillna(0.0) for s in test}
    st = _settings_for(settings, name, params)
    base = run_backtest(test, st, signals=base_sig, source=name)
    filt = run_backtest(test, st, signals=filt_sig, source=f"{name}+ml")
    bm, fm = base.metrics, filt.metrics
    beats = (fm["trades"] >= 10 and fm["sharpe"] > bm["sharpe"] and fm["total_return"] > bm["total_return"])
    X_recent = feature_matrix(next(iter(test.values()))).dropna()
    drift = drift_report(mm, X_recent, ml.drift_psi_threshold)
    reg = ModelRegistry(ml.registry_dir)
    meta = {
        "strategy": name, "params": params, "train_range": [str(union[0]), str(split)],
        "oos_range": [str(split), str(union[-1])], "n_events": int(len(ds.y)),
        "oos_baseline": {k: bm.get(k) for k in ("total_return", "sharpe", "max_drawdown", "trades")},
        "oos_filtered": {k: fm.get(k) for k in ("total_return", "sharpe", "max_drawdown", "trades")},
        "oos_beats_baseline": bool(beats), "drift": {"drifted": drift["drifted"]},
    }
    version = reg.save(mm, meta)
    deployed, msg = (reg.deploy(version) if deploy else (False, "deploy skipped"))
    verdict = {"version": version, "beats_baseline": bool(beats), "deployed": deployed, "message": msg,
               "scores": mm.scores, "importance": dict(list(mm.importance.items())[:10]), **meta}
    if store is not None:
        store.set("ml.verdict", verdict)
    log.info("ml evaluation", extra={"version": version, "beats": beats, "deploy": msg})
    return verdict


def ml_allowed(settings: Settings, store: StateStore | None) -> tuple[bool, str]:
    if not settings.ml.enabled:
        return False, "ml.enabled is false"
    v = (store.get("ml.verdict") if store else None) or {}
    if not v.get("beats_baseline"):
        return False, "latest ML evaluation did not beat the baseline out-of-sample"
    if ModelRegistry(settings.ml.registry_dir).current_version() is None:
        return False, "no deployed model"
    return True, "ok"
