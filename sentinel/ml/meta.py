"""Meta-labelling model: a *filter* on baseline strategy entries, not a price predictor.

Pipeline:
  base signals -> entry events -> triple-barrier labels -> features at the signal bar
  -> purged/embargoed CV (out-of-fold probabilities) -> isotonic calibration on the
  OOF predictions -> final model fit on all training data.

At trade time the calibrated probability that an entry will hit its take-profit
before its stop must exceed ``threshold``; otherwise the entry is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from sentinel.features.registry import FeatureRequest as F
from sentinel.features.registry import compute_features
from sentinel.ml.cv import PurgedKFold
from sentinel.ml.labeling import entry_events, triple_barrier

META_FEATURES = [F("rsi", n=14), F("adx", n=14), F("bbands", n=20, k=2.0), F("rvol", n=24), F("vol_z", n=50),
                 F("roc", n=24), F("hurst", n=128, max_lag=20), F("autocorr", n=50, lag=1), F("macd"),
                 F("ema", n=50), F("atr", n=14), F("mfi", n=14)]


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary-ish features (ratios / oscillators, no raw price levels)."""
    f = compute_features(df, META_FEATURES)
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["rsi"] = f["rsi_14"]
    out["adx"] = f["adx_14"]
    out["di_spread"] = f["plus_di_14"] - f["minus_di_14"]
    out["bb_pctb"] = f["bb_pctb_2.0_20"]
    out["bb_width"] = f["bb_width_2.0_20"]
    out["rvol"] = f["rvol_24"]
    out["vol_z"] = f["vol_z_50"]
    out["roc24"] = f["roc_24"]
    out["hurst"] = f["hurst_20_128"]
    out["autocorr"] = f["autocorr_1_50"]
    out["macd_hist_n"] = f["macd_hist_12_9_26"] / c
    out["dist_ema50"] = c / f["ema_50"] - 1
    out["atr_n"] = f["atr_14"] / c
    out["mfi"] = f["mfi_14"]
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    return out


def make_model(kind: str, seed: int = 0):
    if kind == "lightgbm":
        try:
            import lightgbm as lgb

            return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                      random_state=seed, verbose=-1)
        except ImportError:
            kind = "random_forest"
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=30, random_state=seed, n_jobs=-1)
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    raise ValueError(f"unknown model {kind!r}")


@dataclass
class MetaDataset:
    X: pd.DataFrame
    y: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    ret: np.ndarray
    side: np.ndarray


def build_dataset(data: dict[str, pd.DataFrame], signals: dict[str, pd.Series], atr: dict[str, pd.Series],
                  tp: float, sl: float, max_hold: int, cost: float) -> MetaDataset:
    parts = []
    offset = 0
    for sym, df in data.items():
        ev = entry_events(signals[sym])
        lab = triple_barrier(df, ev, atr[sym], tp, sl, max_hold, cost)
        if lab.empty:
            offset += len(df)
            continue
        X = feature_matrix(df).loc[lab.index]
        X["side"] = lab["side"].to_numpy()
        lab = lab.assign(t0_pos=lab["t0_pos"] + offset, t1_pos=lab["t1_pos"] + offset)
        parts.append((X, lab))
        offset += len(df)
    if not parts:
        raise ValueError("no labelled entry events")
    X = pd.concat([p[0] for p in parts])
    lab = pd.concat([p[1] for p in parts])
    # Sort by time across symbols so CV folds are time-contiguous.
    order = np.argsort(lab.index.to_numpy(), kind="stable")
    X, lab = X.iloc[order], lab.iloc[order]
    t0 = np.argsort(np.argsort(lab.index.to_numpy(), kind="stable"))  # time rank
    # Express t1 on the same rank scale: rank of the first sample starting after t1.
    t1_times = lab["t1"].to_numpy()
    starts = lab.index.to_numpy()
    t1 = np.searchsorted(starts, t1_times, side="right") - 1
    t1 = np.maximum(t1, t0)
    keep = X.notna().all(axis=1).to_numpy()
    return MetaDataset(X[keep], lab["label"].to_numpy(int)[keep], t0[keep], t1[keep], lab["ret"].to_numpy()[keep],
                       lab["side"].to_numpy()[keep])


@dataclass
class MetaModel:
    kind: str
    model: Any
    calibrator: IsotonicRegression | None
    features: list[str]
    threshold: float
    scores: dict[str, float] = field(default_factory=dict)
    importance: dict[str, float] = field(default_factory=dict)
    train_dist: dict[str, list[float]] = field(default_factory=dict)  # quantile bins for drift (PSI)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X[self.features])[:, 1]
        return self.calibrator.predict(p) if self.calibrator is not None else p


def train_meta(ds: MetaDataset, kind: str = "lightgbm", n_splits: int = 5, embargo_pct: float = 0.01,
               threshold: float = 0.55, seed: int = 0) -> MetaModel:
    if len(ds.y) < 100:
        raise ValueError(f"only {len(ds.y)} labelled events: too few to train a model")
    if len(set(ds.y)) < 2:
        raise ValueError("labels have a single class")
    cv = PurgedKFold(n_splits, embargo_pct)
    oof = np.full(len(ds.y), np.nan)
    for tr, te in cv.split(ds.t0, ds.t1):
        if len(set(ds.y[tr])) < 2 or len(tr) < 30:
            continue
        m = make_model(kind, seed)
        m.fit(ds.X.iloc[tr], ds.y[tr])
        oof[te] = m.predict_proba(ds.X.iloc[te])[:, 1]
    ok = ~np.isnan(oof)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof[ok], ds.y[ok])
    p_cal = cal.predict(oof[ok])
    y = ds.y[ok]
    take = p_cal >= threshold
    scores = {
        "n": int(ok.sum()), "base_rate": float(y.mean()),
        "auc_oof": float(roc_auc_score(y, oof[ok])) if len(set(y)) > 1 else float("nan"),
        "logloss_oof": float(log_loss(y, np.clip(oof[ok], 1e-6, 1 - 1e-6))),
        "brier_cal_oof": float(brier_score_loss(y, p_cal)),
        "precision_at_threshold": float(y[take].mean()) if take.any() else float("nan"),
        "take_rate": float(take.mean()),
        "mean_ret_taken": float(ds.ret[ok][take].mean()) if take.any() else float("nan"),
        "mean_ret_all": float(ds.ret[ok].mean()),
    }
    final = make_model(kind, seed)
    final.fit(ds.X, ds.y)
    mm = MetaModel(kind, final, cal, list(ds.X.columns), threshold, scores)
    mm.importance = feature_importance(mm, ds.X)
    mm.train_dist = {c: list(np.nanquantile(ds.X[c].to_numpy(float), np.linspace(0, 1, 11))) for c in ds.X.columns}
    return mm


def feature_importance(mm: MetaModel, X: pd.DataFrame) -> dict[str, float]:
    """Mean |SHAP| for LightGBM (native TreeSHAP via pred_contrib), else permutation-free
    model importances / |coef|."""
    model = mm.model
    try:
        if mm.kind == "lightgbm" and hasattr(model, "booster_"):
            contrib = model.booster_.predict(X[mm.features], pred_contrib=True)[:, :-1]
            vals = np.abs(contrib).mean(axis=0)
        elif hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        else:
            vals = np.abs(model[-1].coef_[0])
    except Exception:
        return {}
    s = pd.Series(vals, index=mm.features).sort_values(ascending=False)
    return {k: float(v) for k, v in (s / s.sum()).items()}


def psi(expected_bins: list[float], actual: np.ndarray) -> float:
    """Population stability index of ``actual`` vs training quantile bins. > 0.25 = significant drift."""
    edges = np.unique(np.asarray(expected_bins, dtype=float))
    if len(edges) < 3:
        return 0.0
    a = actual[np.isfinite(actual)]
    if len(a) == 0:
        return 0.0
    exp_frac = np.full(len(edges) - 1, 1 / (len(edges) - 1))
    act = np.histogram(np.clip(a, edges[0], edges[-1]), bins=edges)[0] / len(a)
    act = np.clip(act, 1e-4, None)
    exp_frac = np.clip(exp_frac, 1e-4, None)
    return float(np.sum((act - exp_frac) * np.log(act / exp_frac)))


def drift_report(mm: MetaModel, X_recent: pd.DataFrame, threshold: float = 0.25) -> dict[str, Any]:
    vals = {c: psi(bins, X_recent[c].to_numpy(float)) for c, bins in mm.train_dist.items() if c in X_recent}
    drifted = sorted([c for c, v in vals.items() if v > threshold], key=lambda c: -vals[c])
    return {"psi": vals, "drifted": drifted, "ok": not drifted}


def filter_signals(signal: pd.Series, df: pd.DataFrame, mm: MetaModel) -> pd.Series:
    """Zero out whole position episodes whose entry probability is below threshold."""
    ev = entry_events(signal)
    if ev.empty:
        return signal
    X = feature_matrix(df).loc[ev.index]
    X["side"] = ev.to_numpy()
    good = X.notna().all(axis=1)
    proba = pd.Series(0.0, index=ev.index)
    if good.any():
        proba[good] = mm.predict_proba(X[good])
    allowed = proba >= mm.threshold
    s = signal.to_numpy(float).copy()
    sign = np.sign(s)
    event_pos = signal.index.get_indexer(ev.index)
    blocked = set(event_pos[~allowed.to_numpy()])
    events = set(event_pos)
    blocking = False
    for i in range(len(s)):
        if i in events:
            blocking = i in blocked  # each new episode gets its own decision
        elif sign[i] == 0:
            blocking = False
        if blocking:
            s[i] = 0.0
    return pd.Series(s, index=signal.index, name=signal.name)
