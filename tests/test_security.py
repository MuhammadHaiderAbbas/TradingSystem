import json
import logging
import os
import stat

import pytest

from sentinel.core.errors import KeySafetyError, VaultError
from sentinel.core.types import Mode
from sentinel.monitoring.logging import REDACTOR, JsonFormatter, setup_logging
from sentinel.security.key_safety import (
    KeyPermissions,
    enforce_key_safety,
    permissions_from_binance,
    permissions_from_coinbase,
)
from sentinel.security.vault import Secret, Vault, exchange_credentials
from tests.conftest import FAST_KDF

PW = "correct horse battery"


# ------------------------------------------------------------------ vault
def test_vault_roundtrip(tmp_path):
    p = tmp_path / "vault.json"
    v = Vault.create(p, PW, kdf_params=FAST_KDF)
    v.set("binance.api_key", "KEY-1234567890")
    v.set("binance.api_secret", "SECRET-abcdefghij")
    v2 = Vault(p, PW)
    assert v2.get("binance.api_key").reveal() == "KEY-1234567890"
    assert v2.names() == ["binance.api_key", "binance.api_secret"]
    assert exchange_credentials(v2, "binance") == {"apiKey": "KEY-1234567890", "secret": "SECRET-abcdefghij"}


def test_vault_never_plaintext_and_0600(tmp_path):
    p = tmp_path / "vault.json"
    Vault.create(p, PW, kdf_params=FAST_KDF).set("k", "SUPERSECRETVALUE")
    raw = p.read_text()
    assert "SUPERSECRETVALUE" not in raw
    doc = json.loads(raw)
    assert doc["kdf_alg"] == "argon2id" and doc["cipher"] == "fernet"
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


def test_vault_wrong_password(tmp_path):
    p = tmp_path / "vault.json"
    Vault.create(p, PW, kdf_params=FAST_KDF).set("k", "value123")
    with pytest.raises(VaultError, match="wrong master password"):
        Vault(p, "not the password")


def test_vault_tamper_detected(tmp_path):
    p = tmp_path / "vault.json"
    Vault.create(p, PW, kdf_params=FAST_KDF).set("k", "value123")
    doc = json.loads(p.read_text())
    tok = doc["token"]
    doc["token"] = tok[:-5] + ("A" if tok[-5] != "A" else "B") + tok[-4:]
    p.write_text(json.dumps(doc))
    with pytest.raises(VaultError):
        Vault(p, PW)


def test_vault_short_password_and_missing(tmp_path):
    with pytest.raises(VaultError):
        Vault.create(tmp_path / "v.json", "short", kdf_params=FAST_KDF)
    v = Vault.create(tmp_path / "v2.json", PW, kdf_params=FAST_KDF)
    with pytest.raises(VaultError):
        v.get("missing")


def test_vault_change_password(tmp_path):
    p = tmp_path / "vault.json"
    v = Vault.create(p, PW, kdf_params=FAST_KDF)
    v.set("k", "value123")
    v.change_password("another long password")
    assert Vault(p, "another long password").get("k").reveal() == "value123"
    with pytest.raises(VaultError):
        Vault(p, PW)


def test_secret_masked():
    s = Secret("my-very-secret")
    assert "my-very-secret" not in repr(s) and "my-very-secret" not in str(s) and f"{s}" == "***"


# ------------------------------------------------------------------ log redaction
def _fmt(msg, *args, **extra):
    rec = logging.LogRecord("sentinel.t", logging.INFO, __file__, 1, msg, args, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return JsonFormatter().format(rec)


def test_registered_secret_redacted_everywhere(tmp_path):
    v = Vault.create(tmp_path / "v.json", PW, kdf_params=FAST_KDF)
    v.set("binance.api_secret", "S3cr3tValue-XYZ")
    out = _fmt("connecting with %s", "S3cr3tValue-XYZ", context={"nested": ["S3cr3tValue-XYZ"]})
    assert "S3cr3tValue-XYZ" not in out
    assert "***" in out


def test_pattern_redaction():
    out = _fmt("request api_key=abcdef123456 secret: 'zzzzzz999' Authorization: Bearer tok.en.123 password=hunter22")
    for leaked in ("abcdef123456", "zzzzzz999", "tok.en.123", "hunter22"):
        assert leaked not in out


def test_secret_keys_in_extra_masked():
    out = json.loads(_fmt("x", payload={"apiKey": "zzz", "api_secret": "yyy", "symbol": "BTC/USDT"}))
    p = out["extra"]["payload"]
    assert p["api_secret"] == "***" and p["symbol"] == "BTC/USDT"


def test_exception_traceback_redacted():
    REDACTOR.register("TRACEBACK-SECRET")
    try:
        raise RuntimeError("failed with TRACEBACK-SECRET")
    except RuntimeError:
        import sys

        rec = logging.LogRecord("sentinel.t", logging.ERROR, __file__, 1, "boom", (), sys.exc_info())
    assert "TRACEBACK-SECRET" not in JsonFormatter().format(rec)


def test_log_file_json_rotating_and_redacted(tmp_path):
    REDACTOR.register("FILESECRET999")
    lg = setup_logging("INFO", tmp_path, max_bytes=20_000, backup_count=2, console=False)
    for i in range(400):
        lg.info("line %d FILESECRET999", i)
    files = sorted(tmp_path.glob("sentinel.jsonl*"))
    assert len(files) >= 2  # rotated
    text = "".join(f.read_text() for f in files)
    assert "FILESECRET999" not in text
    json.loads(files[0].read_text().splitlines()[0])


# ------------------------------------------------------------------ key safety
def test_withdrawal_key_refused_in_every_mode():
    perms = permissions_from_binance({"enableWithdrawals": True, "ipRestrict": True, "enableSpotAndMarginTrading": True})
    for mode in Mode:
        with pytest.raises(KeySafetyError, match="WITHDRAWAL"):
            enforce_key_safety(perms, mode)


def test_no_ip_whitelist_warns_paper_refuses_live():
    perms = permissions_from_binance({"enableWithdrawals": False, "ipRestrict": False, "enableSpotAndMarginTrading": True})
    assert enforce_key_safety(perms, Mode.PAPER)
    with pytest.raises(KeySafetyError):
        enforce_key_safety(perms, Mode.LIVE)


def test_safe_key_passes_live():
    perms = permissions_from_binance({"enableWithdrawals": False, "ipRestrict": True, "enableSpotAndMarginTrading": True})
    assert enforce_key_safety(perms, Mode.LIVE) == []


def test_unknown_permissions_need_attestation_for_live():
    perms = KeyPermissions(known=False, source="kraken:unsupported")
    with pytest.raises(KeySafetyError):
        enforce_key_safety(perms, Mode.LIVE)
    enforce_key_safety(perms, Mode.LIVE, {"no_withdrawal": True, "ip_whitelist": True})


def test_coinbase_transfer_is_withdrawal():
    with pytest.raises(KeySafetyError):
        enforce_key_safety(permissions_from_coinbase({"can_trade": True, "can_transfer": True}), Mode.PAPER)
