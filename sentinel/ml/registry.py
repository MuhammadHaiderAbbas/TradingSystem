"""Model registry with versioning and a deploy gate.

Layout::

    registry_dir/
      <version>/model.joblib    # MetaModel
      <version>/meta.json       # training date, data range, scores, params, features
      CURRENT                   # version currently deployed

``deploy`` refuses a model that scores worse than the current one on the
comparison metric (default: out-of-fold mean return of taken trades), or that
failed the out-of-sample "beats baseline after costs" evaluation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import joblib

from sentinel.ml.meta import MetaModel


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, mm: MetaModel, meta: dict[str, Any]) -> str:
        version = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{mm.kind}"
        d = self.root / version
        n = 1
        while d.exists():
            d = self.root / f"{version}.{n}"
            n += 1
        d.mkdir(parents=True)
        joblib.dump(mm, d / "model.joblib")
        doc = {"version": d.name, "kind": mm.kind, "trained_at": time.time(), "threshold": mm.threshold,
               "scores": mm.scores, "importance": mm.importance, "features": mm.features, **meta}
        (d / "meta.json").write_text(json.dumps(doc, indent=2, default=str))
        return d.name

    def versions(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "meta.json").exists():
                out.append(json.loads((d / "meta.json").read_text()))
        return out

    def current_version(self) -> str | None:
        p = self.root / "CURRENT"
        return p.read_text().strip() if p.exists() else None

    def load(self, version: str | None = None) -> tuple[MetaModel, dict] | None:
        v = version or self.current_version()
        if not v or not (self.root / v / "model.joblib").exists():
            return None
        return joblib.load(self.root / v / "model.joblib"), json.loads((self.root / v / "meta.json").read_text())

    def meta(self, version: str) -> dict:
        return json.loads((self.root / version / "meta.json").read_text())

    def deploy(self, version: str, metric: str = "mean_ret_taken", force: bool = False) -> tuple[bool, str]:
        new = self.meta(version)
        if not force and not new.get("oos_beats_baseline", False):
            return False, "refused: model did not beat the baseline out-of-sample after costs"
        cur_v = self.current_version()
        if cur_v and not force:
            cur = self.meta(cur_v)
            a, b = new["scores"].get(metric, float("nan")), cur["scores"].get(metric, float("nan"))
            if math.isfinite(b) and (not math.isfinite(a) or a < b):
                return False, f"refused: new {metric}={a:.5f} is worse than current {b:.5f} ({cur_v})"
        (self.root / "CURRENT").write_text(version)
        return True, f"deployed {version}"
