from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.config import Settings, StrategyEntry
from sentinel.core.state import StateStore
from sentinel.data.synthetic import demo_universe, generate_ohlcv
from sentinel.monitoring.logging import REDACTOR

FAST_KDF = {"t": 1, "m": 8192, "p": 1}


@pytest.fixture(autouse=True)
def _clean_redactor():
    yield
    REDACTOR.clear()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(app={"home": str(tmp_path / "var")}, logging={"dir": str(tmp_path / "logs")},
                    ml={"registry_dir": str(tmp_path / "models")})


@pytest.fixture
def store(tmp_path) -> StateStore:
    return StateStore(tmp_path / "state.db")


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    return generate_ohlcv(3000, seed=11)


@pytest.fixture(scope="session")
def universe() -> dict[str, pd.DataFrame]:
    return demo_universe(n_bars=6000, seed=3)


@pytest.fixture(scope="session")
def big_universe() -> dict[str, pd.DataFrame]:
    return demo_universe(n_bars=17520, seed=7)


def make_bars(opens, spread: float = 1.0, start="2024-01-01", freq="1h", volume=1e6) -> pd.DataFrame:
    """Deterministic bars: close[i] = open[i+1] (no gaps), high/low = max/min(o,c) +/- spread."""
    o = np.asarray(opens, dtype=float)
    c = np.append(o[1:], o[-1])
    h = np.maximum(o, c) + spread
    lo = np.minimum(o, c) - spread
    idx = pd.date_range(start, periods=len(o), freq=freq, tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": volume}, index=idx)


def settings_for(symbols, tmp_path=None, **upd) -> Settings:
    base = dict(market={"symbols": list(symbols), "venue": "sim"},
                strategies=[StrategyEntry(name="ema_adx")])
    if tmp_path is not None:
        base["app"] = {"home": str(tmp_path / "var")}
    base.update(upd)
    return Settings(**base)
