"""Optional local-LLM (Ollama) news/sentiment scorer.

* Talks only to a local endpoint (validated in config) -- no cloud calls.
* Outputs a *structured score only*: ``{"score": -1..1, "confidence": 0..1}``.
* It never places or sizes trades. At most it can be used as one more feature, and
  only if it improves out-of-sample results like any other feature.
"""

from __future__ import annotations

import json

import httpx

from sentinel.monitoring.logging import get_logger

log = get_logger("ml.sentiment")

PROMPT = (
    "You are a financial news sentiment rater. Rate the likely short-term price impact of the "
    "following headline on {asset}. Reply ONLY with JSON: "
    '{{"score": <float from -1 (very bearish) to 1 (very bullish)>, "confidence": <float 0..1>}}.\n'
    "Headline: {text}"
)


class OllamaSentiment:
    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "llama3.1", timeout: float = 30.0,
                 client: httpx.Client | None = None) -> None:
        if not any(h in endpoint for h in ("localhost", "127.0.0.1", "[::1]")):
            raise ValueError("sentiment endpoint must be local")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.client = client or httpx.Client(timeout=timeout)

    def score(self, text: str, asset: str = "BTC") -> dict | None:
        try:
            r = self.client.post(f"{self.endpoint}/api/generate", json={
                "model": self.model, "prompt": PROMPT.format(asset=asset, text=text[:2000]), "format": "json",
                "stream": False, "options": {"temperature": 0}})
            r.raise_for_status()
            doc = json.loads(r.json()["response"])
            score = float(doc["score"])
            conf = float(doc.get("confidence", 0.5))
        except Exception as e:
            log.warning("sentiment scoring failed", extra={"err": str(e)[:200]})
            return None
        return {"score": max(-1.0, min(1.0, score)), "confidence": max(0.0, min(1.0, conf)), "model": self.model}
