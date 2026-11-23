"""`sentinel data report`: coverage, gaps and last update per symbol."""

from __future__ import annotations

import pandas as pd

from sentinel.data.quality import check
from sentinel.data.storage import OHLCVStore


def data_report(store: OHLCVStore, outlier_z: float = 8.0) -> pd.DataFrame:
    cov = store.coverage()
    if cov.empty:
        return cov
    gaps, missing, ok = [], [], []
    for _, row in cov.iterrows():
        df = store.load(row["symbol"], row["timeframe"])
        rep = check(df, row["symbol"], row["timeframe"], outlier_z=outlier_z)
        gaps.append(len(rep.gaps))
        missing.append(rep.missing)
        ok.append(rep.ok)
    cov["gaps"] = gaps
    cov["missing_bars"] = missing
    cov["quality_ok"] = ok
    return cov
