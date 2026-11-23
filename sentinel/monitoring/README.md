# Monitoring

* `logging.py`: JSON logs with rotation. Vault secrets are registered for redaction, with pattern redaction on top; third-party loggers are filtered too.
* `alerts.py`: opt-in Telegram and email; messages are redacted before sending.
* `health.py`: heartbeats, error and veto counts, CPU, memory and disk.

Verify: `pytest tests/test_security.py -k redact`.
