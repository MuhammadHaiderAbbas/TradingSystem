"""Structured JSON logging with rotation and automatic secret redaction.

Two layers of redaction run on every record:

1. Every secret value loaded from the vault is registered with :data:`REDACTOR`
   and replaced by ``***`` wherever it appears (message, args, extras, tracebacks).
2. Pattern-based redaction for things that *look* like credentials
   (``api_key=...``, ``"secret": "..."``, ``Authorization: Bearer ...``).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

MASK = "***"

_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}

_PATTERNS = [
    # key=value / key: value  (api_key, secret, password, token, passphrase)
    re.compile(
        r"(?i)\b((?:api[_-]?key|api[_-]?secret|secret|password|passwd|passphrase|token|private[_-]?key)"
        r"\b[\"']?\s*[:=]\s*[\"']?)([^\s\"',;}]+)"
    ),
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer|basic)\s+)([A-Za-z0-9._~+/=-]+)"),
    re.compile(r"(?i)(\bbot)(\d{6,}:[A-Za-z0-9_-]{20,})"),  # telegram bot tokens in URLs
]


class Redactor:
    """Thread-safe registry of secret values to mask."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()
        self._lock = threading.Lock()

    def register(self, value: str | None) -> None:
        if value and len(value) >= 4:
            with self._lock:
                self._secrets.add(value)

    def clear(self) -> None:
        with self._lock:
            self._secrets.clear()

    def redact(self, text: str) -> str:
        if not text:
            return text
        with self._lock:
            secrets = sorted(self._secrets, key=len, reverse=True)
        for s in secrets:
            if s in text:
                text = text.replace(s, MASK)
        for pat in _PATTERNS:
            text = pat.sub(lambda m: m.group(1) + MASK, text)
        return text

    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, str) and _is_secret_key(k) and v not in (None, ""):
                    out[k] = MASK
                else:
                    out[k] = self.redact_obj(v)
            return out
        if isinstance(obj, (list, tuple)):
            return [self.redact_obj(v) for v in obj]
        return obj


def _is_secret_key(k: str) -> bool:
    k = k.lower()
    return any(h in k for h in ("api_key", "apikey", "secret", "password", "passphrase", "token", "private_key"))


REDACTOR = Redactor()

logging.getLogger("sentinel").addHandler(logging.NullHandler())


class JsonFormatter(logging.Formatter):
    def __init__(self, redactor: Redactor = REDACTOR) -> None:
        super().__init__()
        self.redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        try:
            msg = record.getMessage()
        except Exception:  # bad format args must never crash logging
            msg = str(record.msg)
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and not k.startswith("_")}
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        payload = self.redactor.redact_obj(payload)
        return json.dumps(payload, default=str)


class RedactingFilter(logging.Filter):
    """For non-JSON handlers (e.g. third-party): redact the rendered message in place."""

    def __init__(self, redactor: Redactor = REDACTOR) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self.redactor.redact(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True


_configured = False


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = "var/logs",
    max_bytes: int = 10_000_000,
    backup_count: int = 10,
    console: bool = True,
) -> logging.Logger:
    global _configured
    root = logging.getLogger("sentinel")
    root.setLevel(level)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    fmt = JsonFormatter()
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            Path(log_dir) / "sentinel.jsonl", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        fh.addFilter(RedactingFilter())
        root.addHandler(fh)
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        ch.addFilter(RedactingFilter())
        root.addHandler(ch)
    # Third-party libraries (ccxt, httpx) can log request URLs/headers: redact those too.
    for name in ("ccxt", "httpx", "urllib3", "uvicorn"):
        lg = logging.getLogger(name)
        lg.addFilter(RedactingFilter())
        lg.setLevel(max(logging.getLevelName(level), logging.WARNING))
    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("sentinel"):
        name = f"sentinel.{name}"
    return logging.getLogger(name)
