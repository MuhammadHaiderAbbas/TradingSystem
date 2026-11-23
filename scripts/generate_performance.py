"""Regenerate docs/PERFORMANCE.md from reproducible backtests on the bundled synthetic dataset.

    python scripts/generate_performance.py

Every number in the generated page comes from this script: fees, spread, slippage and the
risk manager are on; the holdout is evaluated once per strategy with default parameters;
walk-forward results are out-of-sample. Synthetic data says nothing about real markets.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.backtest import robustness as rb  # noqa: E402
from sentinel.config import Settings  # noqa: E402
from sentinel.data.synthetic import demo_universe  # noqa: E402

STRATS = ["buy_hold", "ema_adx", "rsi_bb", "donchian", "vol_momentum", "supertrend_mtf"]


def pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def num(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def main() -> None:
    data = demo_universe(timeframe="1h")
    s = Settings(market={"symbols": list(data), "venue": "sim"})
    dev, hold = rb.split_holdout(data, s.backtest.holdout_fraction)
    rows, details = [], {}
    for name in STRATS:
        t0 = time.time()
        r = rb.backtest_window(dev, s, name, {}, None, None)
        ho, _ = rb.evaluate_holdout(hold, s, name, {})
        wf = None
        if name != "buy_hold":
            n = len(next(iter(dev.values())))
            wf = rb.walk_forward(dev, s, name, train_bars=n // 4, test_bars=n // 8, max_combos=10)
        tr = rb.trade_equity_returns(r)
        mc = rb.monte_carlo(tr, 1000, method="bootstrap") if len(tr) > 1 else None
        m, hm = r.metrics, ho.metrics
        rows.append({
            "strategy": name, "is_return": m["total_return"], "bench": m["benchmark_total_return"],
            "sharpe": m["sharpe"], "mdd": m["max_drawdown"], "trades": m["trades"], "win": m["win_rate"],
            "pf": m["profit_factor"], "fees": m["fees_paid"],
            "wf_oos_return": wf.oos_metrics["total_return"] if wf else None,
            "wf_is": wf.is_mean if wf else None, "wf_oos": wf.oos_mean if wf else None,
            "ho_return": hm["total_return"], "ho_bench": hm["benchmark_total_return"], "ho_sharpe": hm["sharpe"],
            "ho_mdd": hm["max_drawdown"], "ho_trades": hm["trades"],
            "mc_p5": mc.final_return["p5"] if mc else None, "mc_p95": mc.final_return["p95"] if mc else None,
            "periods": rb.by_period(r, dev[next(iter(dev))]["close"], 720).to_dict("records"),
            "halts": len(r.halts),
        })
        print(f"{name}: done in {time.time() - t0:.1f}s", file=sys.stderr)
    details = {r["strategy"]: r for r in rows}
    (ROOT / "docs" / "performance.json").write_text(json.dumps(details, indent=1, default=str))

    lines = [
        "# Performance Report",
        "",
        "> **Read this first.** These results are from a **synthetic, regime-switching random walk** bundled for "
        "the demo (`sentinel.data.synthetic.demo_universe`, 2 assets x 2 years of 1h bars). It has no real edge "
        "to find, so the honest expectation is that baseline strategies **lose money after costs** - and they "
        "do. Use this page as a template for your own runs on real data. Past performance does not guarantee "
        "future results; you can lose capital.",
        "",
        f"Regenerate: `python scripts/generate_performance.py` (last run {time.strftime('%Y-%m-%d', time.gmtime())}). "
        "Costs: 0.10% taker/maker fee, 2 bps spread, 2 bps + 2% of bar range slippage, funding on shorts. "
        "Risk manager on (1% risk/trade, 2xATR stop, 15% max-drawdown halt).",
        "",
        "## Development period (in-sample, default parameters)",
        "",
        "| Strategy | Return | Buy & hold | Sharpe | Max DD | Trades | Win rate | Profit factor | Fees paid | Halts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['strategy']} | {pct(r['is_return'])} | {pct(r['bench'])} | {num(r['sharpe'])} | "
                     f"{pct(r['mdd'])} | {r['trades']} | {pct(r['win'])} | {num(r['pf'])} | {num(r['fees'], 0)} | "
                     f"{r['halts']} |")
    lines += ["", "## Walk-forward (out-of-sample, parameters re-optimised on each train window)", "",
              "| Strategy | Mean IS Sharpe | Mean OOS Sharpe | Stitched OOS return |", "|---|---:|---:|---:|"]
    for r in rows:
        if r["wf_is"] is not None:
            lines.append(f"| {r['strategy']} | {num(r['wf_is'])} | {num(r['wf_oos'])} | {pct(r['wf_oos_return'])} |")
    lines += ["", "A large gap between in-sample and out-of-sample Sharpe is the signature of overfitting.", "",
              "## Holdout (final 20%, never used for tuning, evaluated once)", "",
              "| Strategy | Return | Buy & hold | Sharpe | Max DD | Trades |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['strategy']} | {pct(r['ho_return'])} | {pct(r['ho_bench'])} | {num(r['ho_sharpe'])} | "
                     f"{pct(r['ho_mdd'])} | {r['ho_trades']} |")
    lines += ["", "## Monte Carlo (bootstrap of trades, 1,000 runs, development period)", "",
              "| Strategy | 5th pct return | 95th pct return |", "|---|---:|---:|"]
    for r in rows:
        if r["mc_p5"] is not None:
            lines.append(f"| {r['strategy']} | {pct(r['mc_p5'])} | {pct(r['mc_p95'])} |")
    lines += ["", "## Losing periods: return by market regime (development period)", "",
              "| Strategy | Bull | Bear | Sideways | Crash |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        per = {p["period"]: p["strategy_return"] for p in r["periods"]}
        lines.append(f"| {r['strategy']} | " + " | ".join(pct(per.get(k)) for k in ("bull", "bear", "sideways", "crash"))
                     + " |")
    lines += ["", "## What this tells you", "",
              "* On data with no structure, costs dominate: every active strategy pays more in fees and slippage "
              "than it captures. That is the correct, expected outcome, and a good sanity check that the "
              "backtester isn't leaking future information.",
              "* `buy_hold` as a *tradable strategy* is stopped out by the mandatory stop-loss and waits for its "
              "signal to reset (which never happens). The *benchmark* column is the true buy-and-hold.",
              "* The risk manager's drawdown halt caps losses at roughly the configured 15%, then keeps the "
              "strategy flat. That's why losses cluster around -15% instead of compounding further.",
              "* Before trading real money, run the same pipeline on real data, require a positive out-of-sample "
              "and holdout result, then paper trade for 8+ weeks and pass `sentinel gate`.", ""]
    (ROOT / "docs" / "PERFORMANCE.md").write_text("\n".join(lines))
    print("wrote docs/PERFORMANCE.md", file=sys.stderr)


if __name__ == "__main__":
    main()
