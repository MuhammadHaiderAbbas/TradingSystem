"""Performance metrics. All inputs are net of fees, slippage and funding."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown_duration(equity: pd.Series) -> pd.Timedelta:
    """Longest time from a peak until equity recovers to it (or until the end if it never does)."""
    peak_val, peak_ts, longest = -np.inf, None, pd.Timedelta(0)
    under = False
    for ts, v in equity.items():
        if v >= peak_val:
            if under:
                longest = max(longest, ts - peak_ts)
                under = False
            peak_val, peak_ts = v, ts
        else:
            under = True
    if under:
        longest = max(longest, equity.index[-1] - peak_ts)
    return longest


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None, periods_per_year: float,
                    exposure: pd.Series | None = None, benchmark: pd.Series | None = None,
                    turnover_notional: float | None = None) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {"error": "not enough data"}
    rets = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    years = len(rets) / periods_per_year
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else -1.0
    sd = rets.std()
    sharpe = rets.mean() / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0
    downside = rets[rets < 0]
    dsd = math.sqrt((downside**2).sum() / len(rets)) if len(rets) else 0.0
    sortino = rets.mean() / dsd * math.sqrt(periods_per_year) if dsd > 0 else 0.0
    mdd = float(drawdown_series(eq).min())
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    m: dict = {
        "start": str(eq.index[0]), "end": str(eq.index[-1]), "bars": len(eq),
        "initial_equity": float(eq.iloc[0]), "final_equity": float(eq.iloc[-1]),
        "total_return": float(total), "cagr": float(cagr), "volatility_ann": float(sd * math.sqrt(periods_per_year)),
        "sharpe": float(sharpe), "sortino": float(sortino), "calmar": float(calmar),
        "max_drawdown": mdd, "max_drawdown_duration_days": max_drawdown_duration(eq).total_seconds() / 86400,
    }
    if exposure is not None and len(exposure):
        m["exposure_time"] = float((exposure.reindex(eq.index).fillna(0) > 1e-9).mean())
        m["avg_exposure"] = float(exposure.reindex(eq.index).fillna(0).mean())
    m.update(trade_stats(trades))
    if turnover_notional is not None:
        m["turnover"] = float(turnover_notional / eq.mean() / max(years, 1e-9))  # x equity per year
    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = benchmark.dropna()
        m["benchmark_total_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        m["benchmark_max_drawdown"] = float(drawdown_series(b).min())
        br = b.pct_change().dropna()
        m["benchmark_sharpe"] = float(br.mean() / br.std() * math.sqrt(periods_per_year)) if br.std() > 0 else 0.0
        m["excess_return_vs_benchmark"] = m["total_return"] - m["benchmark_total_return"]
    return m


def trade_stats(trades: pd.DataFrame | None) -> dict:
    if trades is None or len(trades) == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "expectancy_pct": 0.0, "fees_paid_trades": 0.0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gp, gl = wins.sum(), -losses.sum()
    return {
        "trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)),
        "profit_factor": float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "expectancy": float(pnl.mean()),
        "expectancy_pct": float(trades["return_pct"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "largest_loss": float(pnl.min()),
        "avg_holding_hours": float(((trades["exit_ts"] - trades["entry_ts"]) / 3600).mean()),
        "fees_paid_trades": float(trades["fees"].sum()),
    }


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    m = equity.resample("ME").last()
    first = equity.iloc[:1]
    m = pd.concat([first, m])
    r = m.pct_change().dropna()
    r = r[~r.index.duplicated(keep="last")]
    df = pd.DataFrame({"year": r.index.year, "month": r.index.month, "ret": r.to_numpy()})
    return df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")


def buy_and_hold_equity(data: dict[str, pd.DataFrame], initial_cash: float, taker_fee: float,
                        index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Equal-weight buy & hold from the first bar's close, with entry fee. The benchmark
    every strategy must be compared against."""
    parts = []
    for df in data.values():
        c = df["close"]
        parts.append(c / c.iloc[0])
    rel = pd.concat(parts, axis=1).ffill().mean(axis=1)
    eq = initial_cash * (1 - taker_fee) * rel
    eq.iloc[0] = initial_cash  # before buying: the entry fee shows up in the benchmark's return
    return eq.reindex(index).ffill() if index is not None else eq
