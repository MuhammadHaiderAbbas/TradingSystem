"""Self-contained HTML backtest report (inline SVG charts, no external assets/JS).

Sections: headline metrics vs buy & hold, overfitting warnings, equity curve,
drawdown, monthly returns heatmap, robustness (walk-forward, Monte Carlo,
sensitivity heatmap, market periods), trade list and full metrics.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.backtest.engine import BacktestResult
from sentinel.backtest.metrics import drawdown_series, monthly_returns

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--card:#fff;--accent:#2f5d9e;--bench:#a3a29b;
--pos:#2e7d4f;--neg:#b3412f;--warn:#a86b00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#2d2d2a;--card:#1f1f1d;
--accent:#7aa5e0;--bench:#6b6b66;--pos:#5bbf85;--neg:#e0735f;--warn:#e0a640}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,
"Segoe UI",sans-serif}main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}.kpi{background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:10px 12px}.kpi b{display:block;font-size:18px;
font-variant-numeric:tabular-nums}.kpi span{font-size:12px;color:var(--muted)}table{border-collapse:collapse;width:100%;
font-variant-numeric:tabular-nums;font-size:12.5px}th,td{padding:4px 8px;border-bottom:1px solid var(--line);
text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}.warn{border-left:3px solid
var(--warn);padding:6px 10px;margin:6px 0;background:var(--card)}.warn.high{border-color:var(--neg)}
.warn.info{border-color:var(--line)}svg{width:100%;height:auto;display:block}.pos{color:var(--pos)}.neg{color:var(--neg)}
.disclaimer{margin-top:40px;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}
"""


def _fmt(v, kind: str = "num") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–" if v is None or math.isnan(v) else ("∞" if v > 0 else "-∞")
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
    return html.escape(str(v))


PCT_KEYS = {"total_return", "cagr", "max_drawdown", "win_rate", "exposure_time", "avg_exposure", "volatility_ann",
            "benchmark_total_return", "benchmark_max_drawdown", "excess_return_vs_benchmark", "expectancy_pct"}


def line_chart(series: dict[str, pd.Series], height: int = 260, colors: dict[str, str] | None = None,
               area: bool = False, pct_axis: bool = False) -> str:
    w, h, pl, pr, pt, pb = 1000, height, 64, 12, 10, 26
    vals = pd.concat(series.values()).dropna()
    if vals.empty:
        return ""
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    idx = next(iter(series.values())).index
    t0, t1 = idx[0].value, idx[-1].value
    span = max(t1 - t0, 1)

    def xy(ts, v):
        return pl + (ts.value - t0) / span * (w - pl - pr), pt + (hi - v) / (hi - lo) * (h - pt - pb)

    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = pt + (hi - v) / (hi - lo) * (h - pt - pb)
        lab = f"{v * 100:.0f}%" if pct_axis else (f"{v:,.0f}" if abs(hi) > 100 else f"{v:.2f}")
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--line)"/>'
                   f'<text x="{pl - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="var(--muted)">{lab}</text>')
    years = pd.date_range(idx[0].normalize(), idx[-1], freq="MS")
    step = max(1, len(years) // 8)
    for d in years[::step]:
        x, _ = xy(d, lo)
        out.append(f'<text x="{x:.1f}" y="{h - 8}" font-size="11" text-anchor="middle" fill="var(--muted)">'
                   f'{d.strftime("%b %y")}</text>')
    palette = colors or {}
    for name, s in series.items():
        s = s.dropna()
        if len(s) > 1500:  # downsample for file size
            s = s.iloc[:: len(s) // 1500 + 1]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(t, v) for t, v in s.items()))
        col = palette.get(name, "var(--accent)")
        if area:
            x0, y0 = xy(s.index[0], 0)
            x1, _ = xy(s.index[-1], 0)
            out.append(f'<polygon points="{x0:.1f},{y0:.1f} {pts} {x1:.1f},{y0:.1f}" fill="{col}" fill-opacity="0.25"/>')
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/>')
    legend_x = pl + 8
    for name in series:
        col = palette.get(name, "var(--accent)")
        out.append(f'<rect x="{legend_x}" y="{pt + 4}" width="10" height="10" fill="{col}"/>'
                   f'<text x="{legend_x + 14}" y="{pt + 13}" font-size="12" fill="var(--fg)">{html.escape(name)}</text>')
        legend_x += 14 + 8 * len(name) + 16
    out.append("</svg>")
    return "".join(out)


def _cell_color(v: float, scale: float) -> str:
    if v is None or not math.isfinite(v):
        return "transparent"
    a = min(abs(v) / scale, 1.0) * 0.55 + 0.05
    return f"color-mix(in srgb, var({'--pos' if v >= 0 else '--neg'}) {a * 100:.0f}%, transparent)"


def heatmap_table(df: pd.DataFrame, pct: bool = True, row_label: str = "", col_fmt=str) -> str:
    vals = df.to_numpy(dtype=float)
    scale = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    scale = scale or 1.0
    head = "".join(f"<th>{html.escape(col_fmt(c))}</th>" for c in df.columns)
    rows = []
    for r, row in df.iterrows():
        cells = "".join(
            f'<td style="background:{_cell_color(v, scale)}">{"" if not math.isfinite(v) else (f"{v * 100:.1f}%" if pct else f"{v:.2f}")}</td>'
            for v in row.to_numpy(dtype=float))
        rows.append(f"<tr><td>{html.escape(str(r))}</td>{cells}</tr>")
    return f'<div class="scroll"><table><tr><th>{html.escape(row_label)}</th>{head}</tr>{"".join(rows)}</table></div>'


def render_report(result: BacktestResult, title: str, path: str | Path, warnings: list[dict] | None = None,
                  wf=None, mc=None, sens=None, periods: pd.DataFrame | None = None,
                  per_symbol: pd.DataFrame | None = None, holdout: BacktestResult | None = None,
                  notes: str = "") -> Path:
    m = result.metrics
    kpis = [("Total return", m.get("total_return"), "pct"), ("Buy & hold", m.get("benchmark_total_return"), "pct"),
            ("CAGR", m.get("cagr"), "pct"), ("Sharpe", m.get("sharpe"), "num"), ("Sortino", m.get("sortino"), "num"),
            ("Max drawdown", m.get("max_drawdown"), "pct"), ("Trades", m.get("trades"), "num"),
            ("Win rate", m.get("win_rate"), "pct"), ("Profit factor", m.get("profit_factor"), "num"),
            ("Expectancy / trade", m.get("expectancy"), "num"), ("Exposure time", m.get("exposure_time"), "pct"),
            ("Fees paid", m.get("fees_paid"), "num")]
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{_fmt(v, f)}</b></div>' for k, v, f in kpis)
    warn_html = "".join(f'<div class="warn {html.escape(w["level"])}"><b>{html.escape(w["level"].upper())}</b> '
                        f'{html.escape(w["message"])}</div>' for w in (warnings or [])) or '<p class="muted">None.</p>'
    eq_chart = line_chart({"strategy": result.equity, "buy & hold": result.benchmark},
                          colors={"strategy": "var(--accent)", "buy & hold": "var(--bench)"})
    dd_chart = line_chart({"drawdown": drawdown_series(result.equity)}, height=180, colors={"drawdown": "var(--neg)"},
                          area=True, pct_axis=True)
    mr = monthly_returns(result.equity)
    mr.columns = [pd.Timestamp(2000, int(c), 1).strftime("%b") for c in mr.columns]
    sections = [
        f"<h1>{html.escape(title)}</h1><p class='muted'>{html.escape(m.get('start', ''))} → "
        f"{html.escape(m.get('end', ''))} · {m.get('bars', 0):,} bars · strategy <b>{html.escape(result.source)}</b> "
        f"{html.escape(str(result.params or ''))}. All figures are net of fees, spread, slippage and funding.</p>",
        f'<div class="grid">{kpi_html}</div>',
        f"<h2>Overfitting & sanity warnings</h2>{warn_html}",
        f"<h2>Equity curve</h2>{eq_chart}", f"<h2>Drawdown</h2>{dd_chart}",
        f"<h2>Monthly returns</h2>{heatmap_table(mr, row_label='Year')}",
    ]
    if notes:
        sections.insert(1, f"<p>{html.escape(notes)}</p>")
    if holdout is not None:
        hm = holdout.metrics
        sections.append("<h2>Out-of-sample holdout (never used for tuning)</h2>" + _kv_table(
            {k: hm.get(k) for k in ("start", "end", "total_return", "benchmark_total_return", "sharpe", "max_drawdown",
                                    "trades", "win_rate", "expectancy")}))
    if wf is not None:
        rows = "".join(
            f"<tr><td>{html.escape(f['test'][0][:10])} → {html.escape(f['test'][1][:10])}</td>"
            f"<td>{html.escape(str(f['params']))}</td><td>{_fmt(f['is_score'])}</td><td>{_fmt(f['oos_score'])}</td>"
            f"<td>{_fmt(f['oos_return'], 'pct')}</td><td>{_fmt(f['oos_trades'])}</td></tr>" for f in wf.folds)
        sections.append(
            f"<h2>Walk-forward ({len(wf.folds)} folds, objective {html.escape(wf.objective)})</h2>"
            f"<p>Mean in-sample {wf.objective}: <b>{_fmt(wf.is_mean)}</b> · mean out-of-sample: <b>{_fmt(wf.oos_mean)}</b> · "
            f"stitched OOS return <b>{_fmt(wf.oos_metrics.get('total_return'), 'pct')}</b>, max DD "
            f"<b>{_fmt(wf.oos_metrics.get('max_drawdown'), 'pct')}</b></p>"
            f'<div class="scroll"><table><tr><th>Test window</th><th>Chosen params</th><th>IS</th><th>OOS</th>'
            f"<th>OOS return</th><th>OOS trades</th></tr>{rows}</table></div>"
            + line_chart({"walk-forward OOS equity": wf.oos_equity}, height=200))
    if mc is not None:
        sections.append(
            f"<h2>Monte Carlo ({mc.n:,} {html.escape(mc.method)} runs)</h2>"
            f"<p>Probability of a losing outcome: <b>{mc.prob_loss:.1%}</b></p>"
            + _kv_table({f"final return {k}": v for k, v in mc.final_return.items()} |
                        {f"max drawdown {k}": v for k, v in mc.max_drawdown.items()}, pct=True))
    if sens is not None:
        sections.append(
            f"<h2>Parameter sensitivity: {html.escape(sens.metric)} by {html.escape(sens.x)} (cols) × "
            f"{html.escape(sens.y)} (rows)</h2><p>Stability: <b>{sens.stability:.0%}</b> of neighbours keep ≥ 50% of the "
            f"best score · {sens.positive_share:.0%} of cells positive</p>"
            + heatmap_table(sens.grid, pct=False, row_label=sens.y, col_fmt=lambda c: f"{sens.x}={c}"))
    if periods is not None and len(periods):
        sections.append("<h2>By market period</h2>" + _df_table(periods, pct_cols={"strategy_return", "benchmark_return",
                                                                                   "worst_bar", "share_of_time"}))
    if per_symbol is not None and len(per_symbol):
        sections.append("<h2>By symbol</h2>" + _df_table(per_symbol, pct_cols={"total_return", "max_drawdown",
                                                                               "benchmark_total_return"}))
    t = result.trades
    if len(t):
        tt = t.copy()
        tt["entry"] = pd.to_datetime(tt["entry_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M")
        tt["exit"] = pd.to_datetime(tt["exit_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M")
        cols = ["entry", "exit", "symbol", "side", "qty", "entry_price", "exit_price", "pnl", "fees", "return_pct",
                "exit_reason"]
        sections.append(f"<h2>Trades ({len(t):,}; first 500)</h2>" + _df_table(tt[cols].head(500),
                                                                               pct_cols={"return_pct"}))
    sections.append("<h2>All metrics</h2>" + _kv_table(m))
    sections.append(
        "<div class='disclaimer'>Backtest results are hypothetical and do not guarantee future results. They assume the "
        "modelled fees, spread and slippage; real fills can be worse. Trading can lose some or all of your capital.</div>")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body><main>{''.join(sections)}</main></body></html>")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p


def _kv_table(d: dict, pct: bool = False) -> str:
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{_fmt(v, 'pct' if (pct or k in PCT_KEYS) else 'num')}</td></tr>"
                   for k, v in d.items())
    return f'<div class="scroll"><table>{rows}</table></div>'


def _df_table(df: pd.DataFrame, pct_cols: set | None = None) -> str:
    pct_cols = pct_cols or set()
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cls = ""
            if c in ("pnl", "return_pct", "strategy_return", "total_return") and isinstance(v, (int, float)):
                cls = ' class="pos"' if v > 0 else ' class="neg"' if v < 0 else ""
            cells.append(f"<td{cls}>{_fmt(v, 'pct' if c in pct_cols else 'num')}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(body)}</table></div>'
