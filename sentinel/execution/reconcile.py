"""Reconciliation: compare the bot's internal ledger with the venue's real positions.

Run on startup and every ``reconcile_interval_min``. Any mismatch beyond tolerance is
stored as an unresolved reconciliation error and **halts trading** (manual review via
``sentinel recon`` / the dashboard). The readiness gate requires zero unresolved errors.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.core.state import StateStore
from sentinel.core.types import Mode
from sentinel.monitoring.logging import get_logger
from sentinel.risk.manager import RiskManager

log = get_logger("execution.reconcile")


@dataclass
class Mismatch:
    symbol: str
    internal: float
    venue: float

    @property
    def diff(self) -> float:
        return self.venue - self.internal


def diff_positions(internal: dict[str, float], venue: dict[str, float], tolerance: float = 1e-6,
                   rel_tolerance: float = 1e-4) -> list[Mismatch]:
    out = []
    for s in sorted(set(internal) | set(venue)):
        a, b = internal.get(s, 0.0), venue.get(s, 0.0)
        tol = max(tolerance, rel_tolerance * max(abs(a), abs(b)))
        if abs(a - b) > tol:
            out.append(Mismatch(s, a, b))
    return out


def reconcile(internal: dict[str, float], venue: dict[str, float], store: StateStore | None, risk: RiskManager,
              mode: Mode, tolerance: float = 1e-6) -> list[Mismatch]:
    mism = diff_positions(internal, venue, tolerance)
    if store is not None:
        store.heartbeat("reconcile", {"mismatches": len(mism)}, ts=None)
    if mism:
        details = {"mismatches": [m.__dict__ | {"diff": m.diff} for m in mism]}
        if store is not None:
            store.add_recon_error(mode.value, details)
        log.critical("RECONCILIATION MISMATCH", extra=details)
        risk.halt("reconciliation", f"position mismatch in {', '.join(m.symbol for m in mism)}")
    return mism
