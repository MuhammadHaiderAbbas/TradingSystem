"""Opt-in alerts: Telegram and email. Disabled by default; nothing leaves the machine
unless you enable a channel. Credentials come from the vault
(``telegram.bot_token``, ``email.password``). Messages are redacted before sending."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

import httpx

from sentinel.config import AlertsConfig
from sentinel.monitoring.logging import REDACTOR, get_logger

log = get_logger("alerts")

EVENTS = ("trade_filled", "limit_hit", "halt", "error", "daily_summary", "kill")


class Alerter:
    def __init__(self, cfg: AlertsConfig, telegram_token: str | None = None, email_password: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._tg = telegram_token
        self._pw = email_password
        self.client = client
        self.sent: list[tuple[str, str]] = []

    @property
    def enabled(self) -> bool:
        return (self.cfg.telegram.enabled and bool(self._tg)) or (self.cfg.email.enabled and bool(self._pw))

    def send(self, event: str, text: str) -> bool:
        text = REDACTOR.redact(f"[Sentinel] {event}: {text}")
        ok = False
        if self.cfg.telegram.enabled and self._tg:
            ok |= self._telegram(text)
        if self.cfg.email.enabled and self._pw:
            ok |= self._email(event, text)
        if ok:
            self.sent.append((event, text))
        return ok

    def _telegram(self, text: str) -> bool:
        try:
            c = self.client or httpx.Client(timeout=10)
            r = c.post(f"https://api.telegram.org/bot{self._tg}/sendMessage",
                       json={"chat_id": self.cfg.telegram.chat_id, "text": text[:4000]})
            return r.status_code == 200
        except Exception as e:
            log.warning("telegram alert failed", extra={"err": REDACTOR.redact(str(e))[:200]})
            return False

    def _email(self, event: str, text: str) -> bool:
        e = self.cfg.email
        try:
            msg = EmailMessage()
            msg["Subject"] = f"Sentinel: {event}"
            msg["From"], msg["To"] = e.sender, e.recipient
            msg.set_content(text)
            with smtplib.SMTP(e.smtp_host, e.smtp_port, timeout=15) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(e.sender, self._pw)
                s.send_message(msg)
            return True
        except Exception as ex:
            log.warning("email alert failed", extra={"err": REDACTOR.redact(str(ex))[:200]})
            return False
