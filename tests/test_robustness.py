import numpy as np
import pandas as pd
import pytest

from sentinel.backtest import robustness as rb
from sentinel.backtest.report import render_report
from tests.conftest import settings_for


@pytest.fixture(scope="module")
def data(universe):
    return {k: v.iloc[:4000] for k, v in universe.items()}


def test_holdout_split_is_clean(data):
    dev, hold = rb.split_holdout(data, 0.2)
    cut = hold["__cut__"]
    for s, df in dev.items():
        assert df.index.max() < cut
        assert len(df) == pytest.approx(0.8 * len(data[s]), abs=2)


def test_holdout_reuse_is_detected(data, store):
    s = settings_for(list(data))
    _, hold = rb.split_holdout(data, 0.2)
    _, n1 = rb.evaluate_holdout(hold, s, "ema_adx", {}, store)
    _, n1b = rb.evaluate_holdout(hold, s, "ema_adx", {}, store)  # same config again: still 1
    r, n2 = rb.evaluate_holdout(hold, s, "ema_adx", {"fast": 8}, store)
    assert (n1, n1b, n2) == (1, 1, 2)
    assert r.equity.index[0] >= hold["__cut__"]
    w = rb.overfit_warnings(r.metrics, holdout_evals=n2)
    assert any("no longer out-of-sample" in x.message for x in w.warnings)


def test_walk_forward_folds_are_out_of_sample(data):
    s = settings_for(list(data))
    grid = [{"fast": 8, "slow": 48}, {"fast": 12, "slow": 96}]
    wf = rb.walk_forward(data, s, "ema_adx", train_bars=1500, test_bars=800, grid=grid, max_combos=2)
    assert len(wf.folds) >= 2
    for f in wf.folds:
        assert f["train"][1] <= f["test"][0]  # test strictly after train
    tests = [pd.Timestamp(f["test"][0]) for f in wf.folds]
    assert tests == sorted(tests)
    assert wf.oos_equity.index.is_unique and wf.oos_equity.index[0] >= pd.Timestamp(wf.folds[0]["test"][0])


def test_monte_carlo():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025])
    sh = rb.monte_carlo(r, 500, method="shuffle")
    assert sh.final_return["p5"] == pytest.approx(sh.final_return["p95"])  # order can't change compounding
    assert sh.max_drawdown["p5"] <= sh.max_drawdown["p95"] <= 0
    bs = rb.monte_carlo(r, 500, method="bootstrap")
    assert bs.final_return["p5"] < bs.final_return["p95"]
    assert 0 <= bs.prob_loss <= 1
    with pytest.raises(ValueError):
        rb.monte_carlo([0.1], 10)


def test_sensitivity_grid(data):
    s = settings_for(list(data))
    sens = rb.sensitivity(data, s, "ema_adx", "fast", "slow", [8, 12], [48, 96])
    assert sens.grid.shape == (2, 2) and 0 <= sens.stability <= 1


def test_overfit_warnings():
    w = rb.overfit_warnings({"sharpe": 4.2, "trades": 12, "profit_factor": 5})
    msgs = " ".join(x.message for x in w.warnings)
    assert "Sharpe" in msgs and "trades" in msgs and "profit factor" in msgs and not w.ok

    class WF:
        is_mean, oos_mean, objective = 2.0, 0.3, "sharpe"

    w2 = rb.overfit_warnings({"sharpe": 1.0, "trades": 100}, wf=WF())
    assert any("IS/OOS gap" in x.message for x in w2.warnings)


def test_periods_and_report(data, tmp_path):
    s = settings_for(list(data))
    r = rb.backtest_window(data, s, "ema_adx", {}, None, None)
    per = rb.by_period(r, next(iter(data.values()))["close"], 720)
    assert set(per["period"]) <= {"bull", "bear", "sideways", "crash"}
    mc = rb.monte_carlo(rb.trade_equity_returns(r), 200, method="bootstrap") if len(r.trades) > 1 else None
    p = render_report(r, "t", tmp_path / "r.html", [{"level": "info", "message": "x"}], None, mc, None, per)
    html = p.read_text()
    assert "<svg" in html and "Monthly returns" in html and "buy &amp; hold" in html and "net of fees" in html
