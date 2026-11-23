"""Baseline strategies + buy & hold benchmark.

These are deliberately simple, well-known baselines. They are *not* expected to be
profitable out of the box; they exist so that every more complex idea (including
ML) has an honest bar to beat after costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.features.registry import FeatureRequest as F
from sentinel.features.registry import column
from sentinel.strategies.base import Param, Strategy, hold_state


class BuyAndHold(Strategy):
    name = "buy_hold"
    description = "Benchmark: fully invested from the first tradable bar."
    params = {}

    def warmup(self) -> int:
        return 1

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index)


class EmaAdx(Strategy):
    name = "ema_adx"
    description = "Fast/slow EMA crossover, only when ADX confirms a trend."
    regimes = ("trend",)
    params = {
        "fast": Param(12, 5, 50, 1, "fast EMA length"),
        "slow": Param(48, 20, 200, 2, "slow EMA length"),
        "adx_n": Param(14, 7, 30, 1, "ADX length"),
        "adx_min": Param(20.0, 10.0, 40.0, 2.5, "minimum ADX to enter"),
    }

    def validate(self) -> None:
        super().validate()
        if self.p["fast"] >= self.p["slow"]:
            raise ValueError("fast EMA must be shorter than slow EMA")

    def features(self):
        return [F("ema", n=self.p["fast"]), F("ema", n=self.p["slow"]), F("adx", n=self.p["adx_n"])]

    def generate_signals(self, data):
        f, s = data[column(F("ema", n=self.p["fast"]))], data[column(F("ema", n=self.p["slow"]))]
        a = data[column(F("adx", n=self.p["adx_n"]), "adx")]
        above = f > s
        entries = above & (a >= self.p["adx_min"])
        exits = ~above
        return hold_state(entries, exits, ~above & (a >= self.p["adx_min"]), above)


class RsiBollinger(Strategy):
    name = "rsi_bb"
    description = "Mean reversion: buy oversold RSI below the lower Bollinger band; exit at the mid band."
    regimes = ("range",)
    params = {
        "rsi_n": Param(14, 5, 30, 1),
        "rsi_low": Param(30.0, 15.0, 40.0, 2.5),
        "rsi_high": Param(70.0, 60.0, 85.0, 2.5),
        "bb_n": Param(20, 10, 50, 2),
        "bb_k": Param(2.0, 1.5, 3.0, 0.25),
    }

    def features(self):
        return [F("rsi", n=self.p["rsi_n"]), F("bbands", n=self.p["bb_n"], k=self.p["bb_k"])]

    def generate_signals(self, data):
        r = data[column(F("rsi", n=self.p["rsi_n"]))]
        bb = F("bbands", n=self.p["bb_n"], k=self.p["bb_k"])
        c = data["close"]
        lower, mid, upper = data[column(bb, "bb_lower")], data[column(bb, "bb_mid")], data[column(bb, "bb_upper")]
        entries = (r < self.p["rsi_low"]) & (c < lower)
        exits = (c >= mid) | (r > self.p["rsi_high"])
        s_entries = (r > self.p["rsi_high"]) & (c > upper)
        s_exits = (c <= mid) | (r < self.p["rsi_low"])
        return hold_state(entries, exits, s_entries, s_exits)


class DonchianBreakout(Strategy):
    name = "donchian"
    description = "Turtle-style: enter on an N-bar high breakout, exit on an M-bar low."
    regimes = ("trend",)
    params = {
        "entry_n": Param(55, 10, 120, 5),
        "exit_n": Param(20, 5, 60, 5),
    }

    def features(self):
        return [F("donchian", n=self.p["entry_n"]), F("donchian", n=self.p["exit_n"])]

    def generate_signals(self, data):
        ent, ex = F("donchian", n=self.p["entry_n"]), F("donchian", n=self.p["exit_n"])
        # Compare with the channel as of the PREVIOUS bar (today's high is part of today's channel).
        upper_prev = data[column(ent, "dc_upper")].shift(1)
        lower_prev = data[column(ent, "dc_lower")].shift(1)
        exit_low = data[column(ex, "dc_lower")].shift(1)
        exit_high = data[column(ex, "dc_upper")].shift(1)
        c = data["close"]
        return hold_state(c > upper_prev, c < exit_low, c < lower_prev, c > exit_high)


class VolAdjMomentum(Strategy):
    name = "vol_momentum"
    description = "Time-series momentum scaled by inverse realized volatility, with a trend filter."
    regimes = ("trend",)
    params = {
        "lookback": Param(72, 12, 336, 12, "momentum lookback in bars"),
        "vol_n": Param(48, 12, 200, 12),
        "target_vol": Param(0.01, 0.002, 0.05, 0.002, "per-bar volatility target"),
        "threshold": Param(0.5, 0.0, 2.0, 0.25, "min vol-adjusted momentum (z) to hold"),
    }

    def features(self):
        return [F("rvol", n=self.p["vol_n"])]

    def generate_signals(self, data):
        c = data["close"]
        vol = data[column(F("rvol", n=self.p["vol_n"]))]
        mom = np.log(c / c.shift(self.p["lookback"]))
        z = mom / (vol * np.sqrt(self.p["lookback"]))
        scale = (self.p["target_vol"] / vol).clip(upper=1.0)
        raw = np.sign(z) * (z.abs() >= self.p["threshold"]) * scale
        # discretise exposure in 25% steps to avoid churn from tiny rebalances
        return (raw * 4).round() / 4


BUILTIN = [BuyAndHold, EmaAdx, RsiBollinger, DonchianBreakout, VolAdjMomentum]
