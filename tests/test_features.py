import numpy as np
import pandas as pd
import pytest

from sentinel.features import indicators as ind
from sentinel.features.lookahead import find_lookahead
from sentinel.features.registry import REGISTRY, FeatureRequest, compute_features

pta = pytest.importorskip("pandas_ta", reason="pandas-ta reference library not installed")

WARM = 400  # compare after warm-up (seeding conventions differ, then converge)


def close_enough(a, b, rtol=1e-6):
    a = np.asarray(a, float)[WARM:]
    b = np.asarray(b, float)[WARM:]
    m = np.isfinite(a) & np.isfinite(b)
    assert m.mean() > 0.95
    np.testing.assert_allclose(a[m], b[m], rtol=rtol, atol=1e-8)


@pytest.fixture(scope="module")
def d(ohlcv):
    return ohlcv


def test_reference_trend(d):
    h, lo, c = d["high"], d["low"], d["close"]
    close_enough(ind.sma(c, 20), pta.sma(c, 20))
    close_enough(ind.ema(c, 20), pta.ema(c, 20))
    m, ref = ind.macd(c), pta.macd(c, 12, 26, 9)
    close_enough(m["macd"], ref["MACD_12_26_9"])
    close_enough(m["macd_signal"], ref["MACDs_12_26_9"])
    a, ref = ind.adx(h, lo, c, 14), pta.adx(h, lo, c, 14)
    close_enough(a["adx"], ref["ADX_14"])
    close_enough(a["plus_di"], ref["DMP_14"])
    close_enough(a["minus_di"], ref["DMN_14"])


def test_reference_momentum(d):
    h, lo, c = d["high"], d["low"], d["close"]
    close_enough(ind.rsi(c, 14), pta.rsi(c, 14))
    st, ref = ind.stochastic(h, lo, c), pta.stoch(h, lo, c, 14, 3, 3)
    close_enough(st["stoch_k"], ref.iloc[:, 0])
    close_enough(st["stoch_d"], ref.iloc[:, 1])
    close_enough(ind.roc(c, 10), pta.roc(c, 10))
    close_enough(ind.williams_r(h, lo, c, 14), pta.willr(h, lo, c, 14))


def test_reference_volatility_volume(d):
    h, lo, c, v = d["high"], d["low"], d["close"], d["volume"]
    close_enough(ind.atr(h, lo, c, 14), pta.atr(h, lo, c, 14))
    close_enough(ind.mfi(h, lo, c, v, 14), pta.mfi(h, lo, c, v, 14))
    close_enough(ind.obv(c, v).diff(), pta.obv(c, v).diff())
    # Bollinger: TA-Lib convention (population std, ddof=0)
    bb = ind.bollinger(c, 20, 2.0)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    close_enough(bb["bb_upper"], mid + 2 * sd)
    close_enough(bb["bb_mid"], pta.bbands(c, 20, 2).iloc[:, 1])


def test_supertrend_matches_reference_algorithm(d):
    """Exact match vs an independent loop implementation of the pandas-ta algorithm fed the
    same ATR; plus high agreement with pandas-ta itself (its ATR seeding differs early on,
    and Supertrend is path-dependent)."""
    h, lo, c = d["high"], d["low"], d["close"]
    ours = ind.supertrend(h, lo, c, 10, 3.0)
    a = ind.atr(h, lo, c, 10)
    hl2 = (h + lo) / 2
    ub, lb = (hl2 + 3 * a).to_list(), (hl2 - 3 * a).to_list()
    cl = c.to_list()
    direction = [1] * len(cl)
    for i in range(1, len(cl)):
        if np.isnan(a.iloc[i]) or np.isnan(a.iloc[i - 1]):
            continue
        if cl[i] > ub[i - 1]:
            direction[i] = 1
        elif cl[i] < lb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] > 0 and lb[i] < lb[i - 1]:
                lb[i] = lb[i - 1]
            if direction[i] < 0 and ub[i] > ub[i - 1]:
                ub[i] = ub[i - 1]
    ref_dir = pd.Series(direction, index=c.index)
    m = ours["supertrend_dir"].notna()
    assert (ours["supertrend_dir"][m] == ref_dir[m]).all()
    agree = (ours["supertrend_dir"].iloc[WARM:] == pta.supertrend(h, lo, c, length=10, multiplier=3.0).iloc[WARM:, 1]).mean()
    assert agree > 0.6


def test_regime_and_stats_ranges(d):
    hurst = ind.hurst(d["close"]).dropna()
    assert 0.1 < hurst.mean() < 0.9
    lab = ind.regime_labels(d["high"], d["low"], d["close"]).dropna()
    assert set(lab["regime_trend"]) <= {"trend", "range"}
    assert set(lab["regime_vol"]) <= {"high_vol", "low_vol"}
    assert ind.autocorr(d["close"]).dropna().abs().max() <= 1.0


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_feature_has_no_lookahead(d, name):
    req = FeatureRequest(name)
    problems = find_lookahead(lambda x: compute_features(x, [req]), d.iloc[:1500], n_checks=4, min_history=700)
    assert problems == []


def test_multi_timeframe_feature_no_lookahead(d):
    req = FeatureRequest("ema", timeframe="4h", n=20)
    assert find_lookahead(lambda x: compute_features(x, [req], base_tf="1h"), d.iloc[:1500], 5, 700) == []


def test_lookahead_detector_catches_leaks(d):
    assert find_lookahead(lambda x: x["close"].shift(-1), d.iloc[:1000], 3, 500)
    assert find_lookahead(lambda x: x["close"].rolling(5, center=True).mean(), d.iloc[:1000], 3, 500)
    assert find_lookahead(lambda x: x["close"] - x["close"].mean(), d.iloc[:1000], 3, 500)  # full-sample stat


def test_feature_column_names(d):
    f = compute_features(d.iloc[:300], [FeatureRequest("ema", n=12), FeatureRequest("bbands"), FeatureRequest("adx")])
    assert {"ema_12", "bb_upper_2.0_20", "adx_14", "plus_di_14"} <= set(f.columns)
    with pytest.raises(KeyError):
        compute_features(d, [FeatureRequest("nope")])
