"""Robustness tests: walk-forward optimisation, an untouchable holdout, Monte Carlo
trade shuffling, parameter sensitivity, and per-market-period evaluation.

All optimisation happens on the *development* set. The final ``holdout_fraction`` of
history is split off first and only ever touched by :func:`evaluate_holdout`, which
records every evaluation so re-using the holdout for tuning is detectable.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinel.backtest.engine import BacktestResult, run_backtest
from sentinel.backtest.metrics import compute_metrics, drawdown_series
from sentinel.config import Settings, StrategyEntry
from sentinel.core.state import StateStore
from sentinel.data.timeframes import periods_per_year
from sentinel.strategies import registry
from sentinel.strategies.combine import SignalPipeline


# ---------------------------------------------------------------- data splitting
def split_holdout(data: dict[str, pd.DataFrame], fraction: float) -> tuple[dict, dict]:
    """Split each symbol at the same timestamp: (development, holdout)."""
    union = sorted(set().union(*[set(df.index) for df in data.values()]))
    cut = union[int(len(union) * (1 - fraction))]
    dev = {s: df[df.index < cut] for s, df in data.items()}
    hold = {s: df for s, df in data.items()}  # holdout keeps history for warm-up; evaluated from cut
    return dev, {"__cut__": cut, **hold}


def _settings_for(settings: Settings, name: str, params: dict) -> Settings:
    entry = StrategyEntry(name=name, params=params)
    return settings.model_copy(update={"strategies": [entry], "combiner": settings.combiner.model_copy(
        update={"method": "none"})})


def signals_for(data: dict[str, pd.DataFrame], settings: Settings, name: str, params: dict,
                end: pd.Timestamp | None = None) -> dict[str, pd.Series]:
    """Causal signals computed only on data up to ``end`` (exclusive)."""
    s = _settings_for(settings, name, params)
    pipe = SignalPipeline.from_settings(s)
    out = {}
    for sym, df in data.items():
        d = df[df.index < end] if end is not None else df
        out[sym] = pipe.run(d).target
    return out


def backtest_window(data: dict[str, pd.DataFrame], settings: Settings, name: str, params: dict,
                    start: pd.Timestamp | None, end: pd.Timestamp | None, risk_halts: bool = True) -> BacktestResult:
    sig = signals_for(data, settings, name, params, end)
    def _slice(df: pd.DataFrame) -> pd.DataFrame:
        m = np.ones(len(df), dtype=bool)
        if start is not None:
            m &= df.index >= start
        if end is not None:
            m &= df.index < end
        return df[m]

    window = {s: _slice(df) for s, df in data.items()}
    window = {s: w for s, w in window.items() if len(w) > 1}
    sig = {s: sig[s].reindex(window[s].index).fillna(0.0) for s in window}
    return run_backtest(window, _settings_for(settings, name, params), signals=sig, source=name,
                        risk_halts=risk_halts, params=params)


# ---------------------------------------------------------------- parameter search
def param_grid(name: str, overrides: dict[str, list] | None = None, n: int = 4) -> list[dict]:
    cls = registry.get(name)
    space = {k: (overrides or {}).get(k) or p.grid(n) for k, p in cls.params.items()}
    keys = list(space)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(space[k] for k in keys))]
    valid = []
    for c in combos:
        try:
            cls(**c)
            valid.append(c)
        except ValueError:
            continue
    return valid


def _objective(m: dict, objective: str) -> float:
    v = m.get(objective, float("-inf"))
    if m.get("trades", 0) < 5:  # don't reward doing nothing
        return float("-inf")
    return v if math.isfinite(v) else float("-inf")


def optimise(data, settings, name, start, end, grid: list[dict], objective="sharpe", max_combos=40,
             seed=0, method="grid") -> tuple[dict, float, list[tuple[dict, float]]]:
    if method == "optuna":
        return _optimise_optuna(data, settings, name, start, end, objective, max_combos, seed)
    rng = random.Random(seed)
    cands = grid if len(grid) <= max_combos else rng.sample(grid, max_combos)
    scored = []
    for p in cands:
        r = backtest_window(data, settings, name, p, start, end, risk_halts=False)
        scored.append((p, _objective(r.metrics, objective)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0], scored[0][1], scored


def _optimise_optuna(data, settings, name, start, end, objective, n_trials, seed):
    import optuna  # optional dependency

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cls = registry.get(name)

    def obj(trial):
        params = {}
        for k, p in cls.params.items():
            if p.low is None:
                continue
            if isinstance(p.default, int):
                params[k] = trial.suggest_int(k, int(p.low), int(p.high), step=int(p.step or 1))
            else:
                params[k] = trial.suggest_float(k, p.low, p.high, step=p.step)
        try:
            cls(**params)
        except ValueError:
            return float("-inf")
        r = backtest_window(data, settings, name, params, start, end, risk_halts=False)
        return _objective(r.metrics, objective)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials)
    trials = [(t.params, t.value) for t in study.trials if t.value is not None]
    return study.best_params, study.best_value, sorted(trials, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------- walk-forward
@dataclass
class WalkForwardResult:
    folds: list[dict]
    oos_equity: pd.Series
    oos_metrics: dict
    is_mean: float
    oos_mean: float
    objective: str


def walk_forward(data: dict[str, pd.DataFrame], settings: Settings, name: str, train_bars: int, test_bars: int,
                 grid: list[dict] | None = None, objective: str = "sharpe", max_combos: int = 30,
                 method: str = "grid", seed: int = 0, anchored: bool = False) -> WalkForwardResult:
    """Rolling (or anchored) train -> test windows. Parameters are chosen on each train
    window and evaluated on the following unseen test window; OOS results are stitched."""
    grid = grid or param_grid(name)
    union = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    folds, pieces = [], []
    start = 0
    while start + train_bars + test_bars <= len(union):
        tr_s = union[0] if anchored else union[start]
        tr_e = union[start + train_bars]
        te_e_i = min(start + train_bars + test_bars, len(union) - 1)
        te_e = union[te_e_i]
        best, is_score, _ = optimise(data, settings, name, tr_s, tr_e, grid, objective, max_combos, seed, method)
        oos = backtest_window(data, settings, name, best, tr_e, te_e, risk_halts=True)
        folds.append({"train": [str(tr_s), str(tr_e)], "test": [str(tr_e), str(te_e)], "params": best,
                      "is_score": is_score, "oos_score": oos.metrics.get(objective), "oos_return": oos.metrics.get(
                          "total_return"), "oos_trades": oos.metrics.get("trades")})
        pieces.append(oos)
        start += test_bars
    if not pieces:
        raise ValueError("not enough data for one walk-forward fold")
    rets = pd.concat([p.equity.pct_change().fillna(0.0) for p in pieces])
    rets = rets[~rets.index.duplicated(keep="first")]
    eq = settings.backtest.initial_cash * (1 + rets).cumprod()
    trades = pd.concat([p.trades for p in pieces if len(p.trades)], ignore_index=True) if any(
        len(p.trades) for p in pieces) else pd.DataFrame()
    m = compute_metrics(eq, trades, periods_per_year(settings.market.timeframe))
    is_vals = [f["is_score"] for f in folds if math.isfinite(f["is_score"])]
    oos_vals = [f["oos_score"] for f in folds if f["oos_score"] is not None and math.isfinite(f["oos_score"])]
    return WalkForwardResult(folds, eq, m, float(np.mean(is_vals)) if is_vals else float("nan"),
                             float(np.mean(oos_vals)) if oos_vals else float("nan"), objective)


# ---------------------------------------------------------------- holdout
def evaluate_holdout(holdout: dict, settings: Settings, name: str, params: dict,
                     store: StateStore | None = None) -> tuple[BacktestResult, int]:
    """Evaluate on the holdout. Returns (result, number of distinct configs evaluated so far).
    Any number > 1 means the holdout has been used for selection and is no longer clean."""
    cut = holdout["__cut__"]
    data = {k: v for k, v in holdout.items() if k != "__cut__"}
    r = backtest_window(data, settings, name, params, cut, None, risk_halts=True)
    n = 1
    if store is not None:
        key = "holdout.evaluations"
        evals = store.get(key, [])
        fp = hashlib.sha256(json.dumps([name, params], sort_keys=True, default=str).encode()).hexdigest()[:12]
        if fp not in [e["fp"] for e in evals]:
            evals.append({"fp": fp, "strategy": name, "params": params, "ts": time.time(),
                          "sharpe": r.metrics.get("sharpe")})
            store.set(key, evals)
        n = len(evals)
    return r, n


# ---------------------------------------------------------------- Monte Carlo
@dataclass
class MonteCarloResult:
    n: int
    final_return: dict[str, float]
    max_drawdown: dict[str, float]
    prob_loss: float
    method: str


def monte_carlo(trade_returns: np.ndarray | pd.Series, n: int = 2000, seed: int = 0,
                method: str = "shuffle", risk_fraction: float = 1.0) -> MonteCarloResult:
    """Resample the *order* of trades (shuffle) or the trades themselves (bootstrap) and
    compound per-trade equity returns. ``trade_returns`` are trade PnL / equity at entry."""
    r = np.asarray(trade_returns, dtype=float) * risk_fraction
    if len(r) < 2:
        raise ValueError("need at least two trades for Monte Carlo")
    rng = np.random.default_rng(seed)
    finals, dds = np.empty(n), np.empty(n)
    for i in range(n):
        seq = rng.permutation(r) if method == "shuffle" else rng.choice(r, size=len(r), replace=True)
        eq = np.cumprod(1 + seq)
        finals[i] = eq[-1] - 1
        peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
        dds[i] = (np.concatenate([[1.0], eq]) / peak - 1).min()
    pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)}  # noqa: E731
    return MonteCarloResult(n, pct(finals), pct(dds), float((finals < 0).mean()), method)


def trade_equity_returns(result: BacktestResult) -> np.ndarray:
    """Per-trade PnL as a fraction of equity at the time of entry."""
    t = result.trades
    if t.empty:
        return np.array([])
    eq = result.equity
    from sentinel.data.timeframes import epoch_seconds

    pos = np.searchsorted(epoch_seconds(eq.index), t["entry_ts"].to_numpy(float), side="right") - 1
    eq_at = np.where(pos >= 0, eq.to_numpy()[np.clip(pos, 0, None)], eq.iloc[0])
    return t["pnl"].to_numpy() / eq_at


# ---------------------------------------------------------------- sensitivity
@dataclass
class SensitivityResult:
    x: str
    y: str
    x_values: list
    y_values: list
    grid: pd.DataFrame  # rows=y, cols=x
    metric: str
    best: tuple
    stability: float  # share of neighbours of the best cell with >= 50% of its score
    positive_share: float


def sensitivity(data, settings, name: str, x: str, y: str, x_values: list | None = None,
                y_values: list | None = None, base_params: dict | None = None, metric: str = "sharpe",
                n: int = 5) -> SensitivityResult:
    cls = registry.get(name)
    xv = x_values or cls.params[x].grid(n)
    yv = y_values or cls.params[y].grid(n)
    base = dict(base_params or {})
    grid = pd.DataFrame(index=yv, columns=xv, dtype=float)
    for a in xv:
        for b in yv:
            p = base | {x: a, y: b}
            try:
                cls(**p)
            except ValueError:
                continue
            r = backtest_window(data, settings, name, p, None, None, risk_halts=False)
            grid.loc[b, a] = r.metrics.get(metric, np.nan)
    arr = grid.to_numpy(dtype=float)
    if np.all(np.isnan(arr)):
        return SensitivityResult(x, y, xv, yv, grid, metric, (None, None), 0.0, 0.0)
    bi, bj = np.unravel_index(np.nanargmax(arr), arr.shape)
    best = arr[bi, bj]
    neigh = [arr[i, j] for i in range(max(0, bi - 1), min(arr.shape[0], bi + 2))
             for j in range(max(0, bj - 1), min(arr.shape[1], bj + 2)) if (i, j) != (bi, bj) and not np.isnan(arr[i, j])]
    stab = float(np.mean([v >= 0.5 * best for v in neigh])) if neigh and best > 0 else 0.0
    return SensitivityResult(x, y, xv, yv, grid, metric, (xv[bj], yv[bi]), stab,
                             float(np.nanmean(arr > 0)))


# ---------------------------------------------------------------- market periods
def label_periods(close: pd.Series, window_bars: int, bull: float = 0.10, bear: float = -0.10,
                  crash: float = -0.20, crash_bars: int | None = None) -> pd.Series:
    """Label each bar bull / bear / sideways / crash from TRAILING returns (for reporting
    only; never used as a trading input)."""
    trailing = close.pct_change(window_bars)
    lab = pd.Series("sideways", index=close.index, dtype=object)
    lab[trailing > bull] = "bull"
    lab[trailing < bear] = "bear"
    cb = crash_bars or max(window_bars // 4, 2)
    lab[close / close.rolling(cb).max() - 1 < crash] = "crash"
    return lab


def by_period(result: BacktestResult, close: pd.Series, window_bars: int) -> pd.DataFrame:
    lab = label_periods(close, window_bars).reindex(result.equity.index).ffill()
    rets = result.equity.pct_change().fillna(0.0)
    bench = result.benchmark.pct_change().fillna(0.0)
    rows = []
    for name in ("bull", "bear", "sideways", "crash"):
        m = lab == name
        if m.sum() < 2:
            continue
        r, b = rets[m], bench[m]
        rows.append({"period": name, "bars": int(m.sum()), "strategy_return": float((1 + r).prod() - 1),
                     "benchmark_return": float((1 + b).prod() - 1),
                     "worst_bar": float(r.min()), "share_of_time": float(m.mean())})
    return pd.DataFrame(rows)


def per_symbol(data, settings, name, params) -> pd.DataFrame:
    rows = []
    for s, df in data.items():
        r = backtest_window({s: df}, settings, name, params, None, None)
        rows.append({"symbol": s, **{k: r.metrics.get(k) for k in ("total_return", "sharpe", "max_drawdown", "trades",
                                                                     "benchmark_total_return")}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- overfitting warnings
@dataclass
class Warning_:
    level: str
    message: str


@dataclass
class OverfitReport:
    warnings: list[Warning_] = field(default_factory=list)

    def add(self, level: str, msg: str) -> None:
        self.warnings.append(Warning_(level, msg))

    @property
    def ok(self) -> bool:
        return not any(w.level == "high" for w in self.warnings)

    def to_list(self) -> list[dict[str, Any]]:
        return [{"level": w.level, "message": w.message} for w in self.warnings]


def overfit_warnings(metrics: dict, wf: WalkForwardResult | None = None, sens: SensitivityResult | None = None,
                     mc: MonteCarloResult | None = None, holdout_evals: int | None = None,
                     min_trades: int = 30) -> OverfitReport:
    rep = OverfitReport()
    sh = metrics.get("sharpe", 0) or 0
    if sh > 3:
        rep.add("high", f"Sharpe {sh:.2f} > 3: almost certainly overfit or a bug (lookahead, missing costs)")
    if metrics.get("trades", 0) < min_trades:
        rep.add("high", f"only {metrics.get('trades', 0)} trades (< {min_trades}): not statistically meaningful")
    pf = metrics.get("profit_factor", 0) or 0
    if math.isfinite(pf) and pf > 3:
        rep.add("medium", f"profit factor {pf:.2f} > 3 is unusually high")
    if metrics.get("win_rate", 0) > 0.8 and metrics.get("trades", 0) >= min_trades:
        rep.add("medium", "win rate > 80%: check for hidden tail risk")
    if wf is not None and math.isfinite(wf.is_mean) and math.isfinite(wf.oos_mean):
        if wf.is_mean > 0 and wf.oos_mean < 0.5 * wf.is_mean:
            rep.add("high", f"in-sample {wf.objective} {wf.is_mean:.2f} vs out-of-sample {wf.oos_mean:.2f}: "
                            "large IS/OOS gap")
        if wf.oos_mean <= 0:
            rep.add("high", f"walk-forward out-of-sample {wf.objective} is {wf.oos_mean:.2f} (no OOS edge)")
    if sens is not None and sens.stability < 0.5:
        rep.add("medium", f"parameter sensitivity: only {sens.stability:.0%} of neighbouring parameters keep >= 50% "
                          "of the best score (fragile 'magic value')")
    if mc is not None and mc.final_return["p5"] < 0:
        rep.add("info", f"Monte Carlo 5th percentile return {mc.final_return['p5']:.1%}: losses are plausible")
    if holdout_evals is not None and holdout_evals > 1:
        rep.add("high", f"holdout has been evaluated with {holdout_evals} different configs: it is no longer "
                        "out-of-sample")
    if metrics.get("total_return", 0) < metrics.get("benchmark_total_return", float("-inf")):
        rep.add("info", "strategy underperformed buy & hold over this period")
    return rep


def drawdown(equity: pd.Series) -> pd.Series:
    return drawdown_series(equity)
