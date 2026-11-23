import textwrap

import numpy as np
import pandas as pd
import pytest

from sentinel.config import StrategyEntry
from sentinel.features.lookahead import find_lookahead
from sentinel.strategies import registry
from sentinel.strategies.base import Param, Strategy, hold_state
from sentinel.strategies.combine import SignalPipeline, apply_regime_filter, combine_signals, regime_mask


@pytest.mark.parametrize("name", registry.available())
def test_strategy_has_no_lookahead(ohlcv, name):
    strat = registry.create(name)
    for short in (False, True):
        problems = find_lookahead(lambda d, short=short: strat.run(d, base_tf="1h", allow_short=short), ohlcv.iloc[:1600],
                                  n_checks=4, min_history=900)
        assert problems == [], f"{name}: {problems}"


@pytest.mark.parametrize("name", registry.available())
def test_signals_bounded_and_flat_during_warmup(ohlcv, name):
    s = registry.create(name)
    sig = s.run(ohlcv, base_tf="1h")
    assert sig.between(0, 1).all()  # long-only by default
    assert (sig.iloc[: s.warmup()] == 0).all()
    assert len(sig) == len(ohlcv)


def test_parameter_validation():
    with pytest.raises(ValueError):
        registry.create("ema_adx", fast=50, slow=20)
    with pytest.raises(ValueError):
        registry.create("ema_adx", bogus=1)
    with pytest.raises(ValueError):
        registry.create("donchian", entry_n=1)


def test_plugin_directory_discovered(tmp_path, monkeypatch):
    (tmp_path / "my_strat.py").write_text(textwrap.dedent('''
        import pandas as pd
        from sentinel.strategies.base import Strategy

        class AlwaysFlat(Strategy):
            name = "always_flat_test"
            params = {}
            def generate_signals(self, data):
                return pd.Series(0.0, index=data.index)
    '''))
    monkeypatch.setenv("SENTINEL_STRATEGY_PATH", str(tmp_path))
    registry.load_all(force=True)
    assert "always_flat_test" in registry.available()
    assert "supertrend_mtf" in registry.available()  # bundled plugin


def test_param_grid():
    assert Param(10, 5, 15, 5).grid() == [5, 10, 15]
    assert Param(10, 5, 50, 1).grid(4) == [5, 20, 35, 50]
    assert len(Param(10, 5, 50, 1).grid(None)) == 46
    assert Param(1.0).grid() == [1.0]


def test_hold_state():
    e = pd.Series([0, 1, 0, 0, 0, 1, 0], dtype=bool)
    x = pd.Series([0, 0, 0, 1, 0, 0, 1], dtype=bool)
    assert hold_state(e, x).tolist() == [0, 1, 1, 0, 0, 1, 0]


def test_combiner_vote_and_weighted():
    idx = pd.RangeIndex(4)
    sig = {"a": pd.Series([1, 1, 0, -1], idx, dtype=float), "b": pd.Series([1, 0, 0, -1], idx, dtype=float),
           "c": pd.Series([0, 0, 1, -1], idx, dtype=float)}
    v = combine_signals(sig, method="vote", threshold=0.5)
    assert v.tolist() == [1.0, 0.0, 0.0, -1.0]
    w = combine_signals(sig, {"a": 2, "b": 1, "c": 1}, method="weighted", threshold=0.5)
    assert w.iloc[0] == pytest.approx(0.75) and w.iloc[3] == -1.0
    with pytest.raises(ValueError):
        combine_signals(sig, method="none")


def test_regime_filter_entry_mode():
    sig = pd.Series([0, 1, 1, 1, 0, 1, 1], dtype=float)
    mask = pd.Series([True, False, True, False, False, False, True])
    out = apply_regime_filter(sig, mask, "entry")
    # bar1 entry blocked (regime off); bar2 enters; bar3 keeps holding although regime off;
    # bar5 new entry blocked; bar6 enters
    assert out.tolist() == [0, 0, 1, 1, 0, 0, 1]
    assert apply_regime_filter(sig, mask, "strict").tolist() == [0, 0, 1, 0, 0, 0, 1]


def test_regime_mask_labels(ohlcv):
    m = regime_mask(ohlcv, ["trend", "low_vol"])
    assert m.dtype == bool and 0 < m.mean() < 1
    with pytest.raises(ValueError):
        regime_mask(ohlcv, ["moon"])


def test_pipeline_ensemble(ohlcv):
    entries = [StrategyEntry(name="ema_adx"), StrategyEntry(name="donchian"), StrategyEntry(name="vol_momentum")]
    res = SignalPipeline(entries, method="vote", timeframe="1h").run(ohlcv)
    assert set(res.per_strategy) == {"ema_adx", "donchian", "vol_momentum"}
    assert res.source == "ensemble:vote"
    assert set(np.unique(res.target)) <= {0.0, 1.0}


def test_buy_and_hold_benchmark_strategy(ohlcv):
    s = registry.create("buy_hold").run(ohlcv)
    assert s.iloc[0] == 0 and (s.iloc[1:] == 1).all()


class _Abstract(Strategy):
    name = "abstract_test"


def test_abstract_strategy_rejected():
    with pytest.raises(TypeError):
        registry.register(_Abstract)
