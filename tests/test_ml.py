import numpy as np
import pandas as pd
import pytest

from sentinel.ml.cv import PurgedKFold
from sentinel.ml.labeling import entry_events, triple_barrier
from sentinel.ml.meta import MetaModel, build_dataset, filter_signals, psi, train_meta
from sentinel.ml.pipeline import ml_allowed, train_and_evaluate
from sentinel.ml.registry import ModelRegistry
from tests.conftest import make_bars, settings_for


def test_entry_events():
    s = pd.Series([0, 1, 1, 0, -1, -1, 1, 0], dtype=float)
    assert entry_events(s).to_dict() == {1: 1.0, 4: -1.0, 6: 1.0}


def test_triple_barrier_labels():
    opens = [100.0] * 10 + [100, 101, 103, 106, 110, 112] + [100.0] * 20
    df = make_bars(opens, spread=0.2)
    atr = pd.Series(2.0, index=df.index)
    ev = pd.Series([1.0], index=[df.index[9]])
    lab = triple_barrier(df, ev, atr, tp_mult=2, sl_mult=2, max_hold=10, cost=0)
    row = lab.iloc[0]
    assert row["label"] == 1 and row["ret"] == pytest.approx(0.04)  # TP at entry 100 + 4
    ev_short = pd.Series([-1.0], index=[df.index[9]])
    assert triple_barrier(df, ev_short, atr, 2, 2, 10, 0).iloc[0]["label"] == 0  # stop for a short
    # time barrier on a flat path: label from net return after cost
    flat = make_bars([100.0] * 40, spread=0.1)
    tb = triple_barrier(flat, pd.Series([1.0], index=[flat.index[5]]), pd.Series(2.0, index=flat.index), 2, 2, 5,
                        cost=0.001)
    assert tb.iloc[0]["label"] == 0 and tb.iloc[0]["t1_pos"] == 10


def test_purged_kfold_no_overlap_and_embargo():
    n = 200
    t0 = np.arange(n)
    t1 = t0 + 5  # each label spans 5 bars
    cv = PurgedKFold(5, embargo_pct=0.05)
    for train, test in cv.split(t0, t1, n_bars=n + 5):
        T0, T1 = t0[test].min(), t1[test].max()
        assert not np.any((t0[train] <= T1) & (t1[train] >= T0))  # purged
        assert not np.any((t0[train] > T1) & (t0[train] <= T1 + 11))  # embargoed (ceil(205*0.05)=11)
        assert len(np.intersect1d(train, test)) == 0
    with pytest.raises(ValueError):
        list(cv.split(t0[::-1], t1[::-1]))


def test_psi_drift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    bins = list(np.quantile(base, np.linspace(0, 1, 11)))
    assert psi(bins, rng.normal(0, 1, 2000)) < 0.05
    assert psi(bins, rng.normal(1.5, 1, 2000)) > 0.25


def test_meta_model_and_filter(big_universe):
    universe = big_universe
    from sentinel.backtest.robustness import signals_for
    from sentinel.features import indicators as ind

    s = settings_for(list(universe))
    sig = signals_for(universe, s, "rsi_bb", {})
    atr = {k: ind.atr(v["high"], v["low"], v["close"]) for k, v in universe.items()}
    ds = build_dataset(universe, sig, atr, 2, 2, 48, 0.002)
    assert np.all(np.diff(ds.t0) >= 0) and len(ds.y) > 100
    mm = train_meta(ds, "logistic", n_splits=4)
    assert 0 <= mm.scores["auc_oof"] <= 1 and mm.importance
    p = mm.predict_proba(ds.X.iloc[:50])
    assert ((p >= 0) & (p <= 1)).all()
    sym = next(iter(universe))
    filt = filter_signals(sig[sym], universe[sym], mm)
    assert (filt.abs() <= sig[sym].abs()).all()  # a filter can only remove trades, never add


def test_ml_stays_off_unless_it_beats_baseline(big_universe, tmp_path, store):
    universe = big_universe
    s = settings_for(list(universe), tmp_path, ml={"registry_dir": str(tmp_path / "models"), "model": "logistic",
                                                  "enabled": True},
                     strategies=[{"name": "rsi_bb"}])
    v = train_and_evaluate(universe, s, store=store)
    reg = ModelRegistry(tmp_path / "models")
    assert reg.versions()
    if not v["beats_baseline"]:
        assert not v["deployed"] and reg.current_version() is None
        assert ml_allowed(s, store) == (False, "latest ML evaluation did not beat the baseline out-of-sample")
    else:
        assert v["deployed"] and ml_allowed(s, store)[0]


def test_registry_refuses_worse_model(tmp_path):
    reg = ModelRegistry(tmp_path)
    mm = MetaModel("logistic", None, None, ["a"], 0.55, scores={"mean_ret_taken": 0.01})
    v1 = reg.save(mm, {"oos_beats_baseline": True})
    assert reg.deploy(v1)[0]
    worse = MetaModel("logistic", None, None, ["a"], 0.55, scores={"mean_ret_taken": 0.005})
    v2 = reg.save(worse, {"oos_beats_baseline": True})
    ok, msg = reg.deploy(v2)
    assert not ok and "worse" in msg and reg.current_version() == v1
    v3 = reg.save(MetaModel("logistic", None, None, ["a"], 0.55, scores={"mean_ret_taken": 0.05}),
                  {"oos_beats_baseline": False})
    assert not reg.deploy(v3)[0]  # better score but failed the OOS-vs-baseline test


def test_sentiment_is_local_only_and_structured():
    import httpx

    from sentinel.ml.sentiment import OllamaSentiment

    with pytest.raises(ValueError):
        OllamaSentiment("https://example.com")

    def handler(req):
        return httpx.Response(200, json={"response": '{"score": 3.0, "confidence": 0.9}'})

    s = OllamaSentiment(client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = s.score("BTC ETF approved")
    assert out["score"] == 1.0 and out["confidence"] == 0.9  # clipped to [-1, 1]
    bad = OllamaSentiment(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    assert bad.score("x") is None
