"""Example plugin: Supertrend direction with a 4h EMA trend filter (multi-timeframe).

Copy this file, rename the class and ``name``, and it is picked up automatically.
"""

from __future__ import annotations

import pandas as pd

from sentinel.features.registry import FeatureRequest as F
from sentinel.features.registry import column
from sentinel.strategies.base import Param, Strategy


class SupertrendMTF(Strategy):
    name = "supertrend_mtf"
    description = "Long when 1h Supertrend is up AND close is above the 4h EMA (higher-timeframe filter)."
    regimes = ("trend",)
    timeframe = "1h"
    params = {
        "st_n": Param(10, 5, 30, 1),
        "st_mult": Param(3.0, 1.5, 5.0, 0.5),
        "htf_ema": Param(50, 10, 200, 10),
    }

    def features(self):
        return [F("supertrend", n=self.p["st_n"], mult=self.p["st_mult"]),
                F("ema", timeframe="4h", n=self.p["htf_ema"])]

    def prepare(self, ohlcv: pd.DataFrame, base_tf: str | None = None) -> pd.DataFrame:
        return super().prepare(ohlcv, base_tf or "1h")

    def generate_signals(self, data):
        d = data[column(F("supertrend", n=self.p["st_n"], mult=self.p["st_mult"]), "supertrend_dir")]
        htf = data[column(F("ema", timeframe="4h", n=self.p["htf_ema"]))]
        return ((d > 0) & (data["close"] > htf)).astype(float)
