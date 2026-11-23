"""Technical indicators, all strictly causal.

Rule: the value at bar *t* uses only data from bars <= t (bar t's own OHLCV is known
at its close). No centred windows, no ``shift(-k)``, no full-sample statistics.
Ichimoku's forward-plotted spans are reported at the time they become known, and
the Chikou span (which looks into the future) is intentionally omitted.

Each function takes pandas Series and returns a Series/DataFrame aligned to the input.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average (RMA)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------- trend
def sma(close: pd.Series, n: int = 20) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int = 20) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - close.ewm(
        span=slow, adjust=False, min_periods=slow).mean()
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan  # no previous close on the first bar (matches TA-Lib/pandas-ta)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _rma(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm.iloc[0] = minus_dm.iloc[0] = np.nan
    a = atr(high, low, close, n)
    plus_di = 100 * _rma(plus_dm, n) / a
    minus_di = 100 * _rma(minus_dm, n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": _rma(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Supertrend (pandas-ta / TA-Lib-style band ratcheting). dir = +1 uptrend, -1 downtrend."""
    a = atr(high, low, close, n).to_numpy()
    hl2 = ((high + low) / 2).to_numpy()
    c = close.to_numpy()
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    direction = np.ones(len(c))
    st = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if np.isnan(a[i]) or np.isnan(a[i - 1]):
            continue
        if c[i] > ub[i - 1]:
            direction[i] = 1
        elif c[i] < lb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] > 0 and lb[i] < lb[i - 1]:
                lb[i] = lb[i - 1]
            if direction[i] < 0 and ub[i] > ub[i - 1]:
                ub[i] = ub[i - 1]
        st[i] = lb[i] if direction[i] > 0 else ub[i]
    d = pd.Series(direction, index=close.index)
    d[np.isnan(st)] = np.nan
    return pd.DataFrame({"supertrend": st, "supertrend_dir": d}, index=close.index)


def ichimoku(high: pd.Series, low: pd.Series, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """Ichimoku components *as known at bar t* (no forward plotting, no Chikou)."""
    t = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    k = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = (t + k) / 2
    span_b = (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2
    # The cloud plotted at t was computed kijun bars ago: shift *backward-looking*.
    return pd.DataFrame({
        "ichi_tenkan": t, "ichi_kijun": k,
        "ichi_span_a": span_a.shift(kijun), "ichi_span_b": span_b.shift(kijun),
    })


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"dc_upper": high.rolling(n).max(), "dc_lower": low.rolling(n).min()})


# ---------------------------------------------------------------- momentum
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _rma(d.clip(lower=0), n)
    loss = _rma(-d.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3,
               smooth_k: int = 3) -> pd.DataFrame:
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    raw = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    kk = raw.rolling(smooth_k).mean()
    return pd.DataFrame({"stoch_k": kk, "stoch_d": kk.rolling(d).mean()})


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return 100 * (close / close.shift(n) - 1)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


# ---------------------------------------------------------------- volatility
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": up, "bb_lower": lo,
        "bb_pctb": (close - lo) / (up - lo).replace(0, np.nan), "bb_width": (up - lo) / mid,
    })


def keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, mult: float = 2.0,
            atr_n: int = 10) -> pd.DataFrame:
    mid = ema(close, n)
    a = atr(high, low, close, atr_n)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + mult * a, "kc_lower": mid - mult * a})


def realized_vol(close: pd.Series, n: int = 24, periods_per_year: float | None = None) -> pd.Series:
    r = np.log(close).diff()
    v = r.rolling(n).std()
    return v * np.sqrt(periods_per_year) if periods_per_year else v


def parkinson_vol(high: pd.Series, low: pd.Series, n: int = 24, periods_per_year: float | None = None) -> pd.Series:
    hl = np.log(high / low) ** 2
    v = np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)))
    return v * np.sqrt(periods_per_year) if periods_per_year else v


# ---------------------------------------------------------------- volume
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int | None = 24,
         session: bool = False) -> pd.Series:
    """Rolling VWAP over n bars, or a daily session VWAP (UTC midnight reset)."""
    tp = (high + low + close) / 3
    pv = tp * volume
    if session:
        day = close.index.floor("1D")
        return pv.groupby(day).cumsum() / volume.groupby(day).cumsum().replace(0, np.nan)
    return pv.rolling(n).sum() / volume.rolling(n).sum().replace(0, np.nan)


def volume_zscore(volume: pd.Series, n: int = 50) -> pd.Series:
    m = volume.rolling(n).mean()
    s = volume.rolling(n).std()
    return (volume - m) / s.replace(0, np.nan)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume
    d = tp.diff()
    pos = rmf.where(d > 0, 0.0).rolling(n).sum()
    neg = rmf.where(d < 0, 0.0).rolling(n).sum()
    out = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    out = out.where(neg != 0, 100.0)
    out.iloc[:n] = np.nan
    return out


# ---------------------------------------------------------------- statistical
def rolling_zscore(s: pd.Series, n: int = 50) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std().replace(0, np.nan)


def hurst(close: pd.Series, n: int = 128, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent via the scaling of lagged log-price differences.

    H ~ 0.5 random walk, > 0.5 trending (persistent), < 0.5 mean-reverting.
    Vectorised: for each lag compute a rolling std, then regress log(std) on log(lag).
    """
    lp = np.log(close)
    lags = np.arange(2, max_lag + 1)
    cols = [np.log(lp.diff(lag).rolling(n).std().replace(0, np.nan)) for lag in lags]
    y = np.column_stack(cols)
    x = np.log(lags)
    xm = x - x.mean()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ymean = np.nanmean(y, axis=1, keepdims=True)
    slope = ((y - ymean) * xm).sum(axis=1) / (xm**2).sum()
    out = pd.Series(slope, index=close.index)
    out[np.isnan(y).any(axis=1)] = np.nan
    return out


def autocorr(close: pd.Series, n: int = 50, lag: int = 1) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(n).corr(r.shift(lag))


def regime_labels(high: pd.Series, low: pd.Series, close: pd.Series, adx_n: int = 14, adx_trend: float = 25.0,
                  vol_n: int = 24, vol_lookback: int = 500) -> pd.DataFrame:
    """Trend/range from ADX; high/low volatility vs the trailing median of realized vol."""
    a = adx(high, low, close, adx_n)["adx"]
    trend = pd.Series(np.where(a >= adx_trend, "trend", "range"), index=close.index).where(a.notna())
    rv = realized_vol(close, vol_n)
    med = rv.rolling(vol_lookback, min_periods=max(vol_n * 2, 50)).median()
    vol = pd.Series(np.where(rv > med, "high_vol", "low_vol"), index=close.index).where(med.notna() & rv.notna())
    return pd.DataFrame({"regime_trend": trend, "regime_vol": vol})
