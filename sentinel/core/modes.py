"""Mode system: BACKTEST, PAPER, LIVE.

PAPER is the default. LIVE can only be entered when *all* of these hold:

* the readiness gate (``sentinel gate``) has passed and recorded a pass for the
  current config fingerprint;
* the kill switch is not engaged;
* no halt is active;
* the operator typed the confirmation phrase.

Changing risk, cost or strategy config after the gate passed invalidates the pass.
"""

from __future__ import annotations

import hashlib
import json

from sentinel.config import Settings
from sentinel.core.errors import KillSwitchEngaged, ModeLockedError, TradingHalted
from sentinel.core.state import StateStore
from sentinel.core.types import Mode

LIVE_CONFIRMATION = "I understand that I can lose real money"
GATE_KEY = "gate.pass"
KILL_KEY = "kill.locked"
HALT_KEY = "risk.halt"


def config_fingerprint(settings: Settings) -> str:
    """Hash of the settings that matter for the gate (strategy, risk, costs, market)."""
    relevant = {
        "market": settings.market.model_dump(mode="json"),
        "strategies": [s.model_dump(mode="json") for s in settings.strategies],
        "combiner": settings.combiner.model_dump(mode="json"),
        "risk": settings.risk.model_dump(mode="json"),
        "costs": settings.costs.model_dump(mode="json"),
        "ml": settings.ml.model_dump(mode="json"),
    }
    relevant["market"].pop("testnet", None)
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


class ModeManager:
    def __init__(self, settings: Settings, store: StateStore) -> None:
        self.settings = settings
        self.store = store

    def live_unlocked(self) -> tuple[bool, str]:
        gate = self.store.get(GATE_KEY)
        if not gate or not gate.get("passed"):
            return False, "readiness gate has not passed (run `sentinel gate`)"
        if gate.get("fingerprint") != config_fingerprint(self.settings):
            return False, "config changed since the gate passed; paper-trade the new config and re-run the gate"
        return True, "gate passed"

    def require(self, mode: Mode, confirmation: str | None = None) -> Mode:
        if self.store.get(KILL_KEY):
            raise KillSwitchEngaged("kill switch is engaged; run `sentinel unlock` after investigating")
        halt = self.store.get(HALT_KEY)
        if halt and mode is not Mode.BACKTEST:
            raise TradingHalted(f"trading halted: {halt.get('reason')}; manual re-enable required")
        if mode is Mode.LIVE:
            ok, why = self.live_unlocked()
            if not ok:
                raise ModeLockedError(f"LIVE is locked: {why}")
            if confirmation != LIVE_CONFIRMATION:
                raise ModeLockedError(f"LIVE requires the confirmation phrase: '{LIVE_CONFIRMATION}'")
        self.store.set("mode.current", mode.value)
        return mode
