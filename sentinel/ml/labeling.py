"""Triple-barrier labelling for meta-labelling (López de Prado, AFML ch. 3).

For each base-strategy entry signal at bar i (decided at i's close, filled at i+1's
open), look forward up to ``max_hold`` bars:

* upper barrier (take-profit) = entry +/- tp_mult x ATR_i  -> label 1
* lower barrier (stop-loss)   = entry -/+ sl_mult x ATR_i  -> label 0
* time barrier                -> 1 if the net return (after ``cost``) is > 0, else 0

If both barriers are touched in the same bar the stop is assumed first
(conservative). ``t1`` (the bar at which the label is resolved) is returned so that
cross-validation can purge overlapping samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def entry_events(signal: pd.Series) -> pd.Series:
    """Bars where the signal goes from flat/other side to a new non-zero direction."""
    s = np.sign(signal.fillna(0.0))
    prev = s.shift(1).fillna(0.0)
    ev = (s != 0) & (s != prev)
    return s[ev]


def triple_barrier(df: pd.DataFrame, events: pd.Series, atr: pd.Series, tp_mult: float = 2.0,
                   sl_mult: float = 2.0, max_hold: int = 48, cost: float = 0.002) -> pd.DataFrame:
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr.reindex(df.index).to_numpy(float)
    pos = df.index.get_indexer(events.index)
    rows = []
    n = len(df)
    for p, side in zip(pos, events.to_numpy(), strict=True):
        if p < 0 or p + 1 >= n or not np.isfinite(a[p]) or a[p] <= 0:
            continue
        entry = o[p + 1]
        tp = entry + side * tp_mult * a[p]
        sl = entry - side * sl_mult * a[p]
        label, t1, ret = None, None, None
        last = min(p + max_hold, n - 1)
        for j in range(p + 1, last + 1):
            hit_sl = lo[j] <= sl if side > 0 else h[j] >= sl
            hit_tp = h[j] >= tp if side > 0 else lo[j] <= tp
            if hit_sl:
                label, t1, ret = 0, j, side * (sl / entry - 1) - cost
                break
            if hit_tp:
                label, t1, ret = 1, j, side * (tp / entry - 1) - cost
                break
        if label is None:
            if last == n - 1 and last - p < max_hold:
                continue  # unresolved at the end of data: drop (no peeking past the end)
            ret = side * (c[last] / entry - 1) - cost
            label, t1 = int(ret > 0), last
        rows.append({"ts": df.index[p], "side": int(side), "label": label, "t1": df.index[t1], "ret": ret,
                     "t0_pos": p, "t1_pos": t1})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.set_index("ts")
    return out
