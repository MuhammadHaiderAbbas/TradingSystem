import base64
import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from sentinel.cli import app as cli_app
from sentinel.core.modes import KILL_KEY
from sentinel.dashboard.app import check_bind, create_app, set_password
from sentinel.ops.ops import create_backup, performance_report, prune_backups, restore_backup

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ dashboard
@pytest.fixture
def client(settings, store):
    set_password(store, "a-long-dashboard-pw")
    store.record_equity("PAPER", 1_700_000_000, 10_000, 10_000, 0.0)
    store.record_equity("PAPER", 1_700_003_600, 10_050, 5_000, 0.5)
    return TestClient(create_app(settings, store)), store


def auth(pw="a-long-dashboard-pw", user="admin"):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def test_dashboard_requires_auth(client):
    c, _ = client
    assert c.get("/api/overview").status_code == 401
    assert c.get("/api/overview", headers=auth("wrong-password")).status_code == 401
    assert c.get("/api/overview", headers=auth(user="root")).status_code == 401
    r = c.get("/api/overview", headers=auth())
    assert r.status_code == 200 and r.json()["equity"] == 10_050
    assert r.headers["X-Frame-Options"] == "DENY"


def test_dashboard_endpoints(client):
    c, _ = client
    h = auth()
    for path in ("/", "/api/equity", "/api/compare", "/api/risk", "/api/strategies", "/api/journal",
                 "/api/health", "/api/events"):
        assert c.get(path, headers=h).status_code == 200, path
    assert "KILL SWITCH" in c.get("/", headers=h).text


def test_dashboard_kill_needs_typed_confirmation(client):
    c, store = client
    assert c.post("/api/kill", json={"confirm": "yes"}, headers=auth()).status_code == 400
    assert store.get(KILL_KEY) is None
    r = c.post("/api/kill", json={"confirm": "KILL", "close_positions": True}, headers=auth())
    assert r.status_code == 200 and store.get(KILL_KEY)["executed"] is False


def test_dashboard_strategy_toggle(client):
    c, store = client
    r = c.post("/api/strategies/ema_adx/toggle", headers=auth())
    assert r.json() == {"name": "ema_adx", "enabled": False}
    assert store.get("strategies.disabled") == ["ema_adx"]
    assert c.post("/api/strategies/nope/toggle", headers=auth()).status_code == 404


def test_dashboard_bind_and_password_policy(store):
    check_bind("127.0.0.1", False)
    with pytest.raises(SystemExit):
        check_bind("0.0.0.0", False)
    with pytest.raises(ValueError):
        set_password(store, "short")


# ------------------------------------------------------------------ CLI
@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    cfg = tmp_path / "s.yaml"
    data = yaml.safe_load((ROOT / "config" / "sentinel.example.yaml").read_text())
    data["app"]["home"] = str(tmp_path / "var")
    data["logging"]["dir"] = str(tmp_path / "logs")
    data["ml"]["registry_dir"] = str(tmp_path / "models")
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("SENTINEL_CONFIG", str(cfg))
    monkeypatch.delenv("SENTINEL_HOME", raising=False)
    return cfg


def test_cli_refuses_invalid_config(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("risk: {max_leverage: 20}\n")
    r = CliRunner().invoke(cli_app, ["status", "--config", str(bad)])
    assert r.exit_code == 2 and "refusing to start" in r.output


def test_cli_backtest_status_kill_gate(cli_env):
    runner = CliRunner()
    r = runner.invoke(cli_app, ["backtest", "--demo", "--strategy", "ema_adx", "--save-expectations"])
    assert r.exit_code == 0, r.output
    assert "report:" in r.output and "holdout" in r.output
    r = runner.invoke(cli_app, ["status", "--json"])
    assert r.exit_code == 0 and '"mode": "PAPER"' in r.output
    r = runner.invoke(cli_app, ["kill", "--yes", "--test"])
    assert r.exit_code == 0 and "LOCKED" in r.output
    r = runner.invoke(cli_app, ["paper", "--demo", "--max-bars", "5"])
    assert r.exit_code == 4  # locked by the kill switch
    r = runner.invoke(cli_app, ["live", "--confirm", "I understand that I can lose real money"])
    assert r.exit_code == 5 and "LIVE refused" in r.output
    r = runner.invoke(cli_app, ["gate", "--skip-tests"])
    assert r.exit_code == 6 and "GATE FAILED" in r.output and "[PASS] kill_switch_tested" in r.output
    r = runner.invoke(cli_app, ["strategies"])
    assert "ema_adx" in r.output and "supertrend_mtf" in r.output


# ------------------------------------------------------------------ ops
def test_backup_roundtrip_and_wrong_password(tmp_path, store):
    store.set("x", 1)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("app: {}\n")
    out = create_backup([cfg, tmp_path / "state.db"], tmp_path / "b", "backup-password")
    assert b"app:" not in out.read_bytes()  # encrypted
    names = restore_backup(out, "backup-password", tmp_path / "restore")
    assert any(n.endswith("state.db") for n in names)
    from cryptography.fernet import InvalidToken

    with pytest.raises(InvalidToken):
        restore_backup(out, "wrong-password!", tmp_path / "r2")
    for i in range(3):
        (tmp_path / "b" / f"sentinel-backup-2000010{i}.enc").write_text("x")
    assert prune_backups(tmp_path / "b", keep=2) == 2


def test_performance_report(store, tmp_path):
    import time

    now = time.time()
    store.record_equity("PAPER", now - 7200, 100, 100, 0)
    store.record_equity("PAPER", now - 3600, 101, 101, 0)
    p = performance_report(store, "PAPER", 1, tmp_path / "r.html")
    assert "<svg" in p.read_text()


# ------------------------------------------------------------------ repo hygiene
def test_gitignore_excludes_secrets():
    gi = (ROOT / ".gitignore").read_text()
    for pat in (".env", "vault.json", "var/", "*.key", "backups/"):
        assert pat in gi


def test_no_secrets_committed():
    import re

    pats = [re.compile(r"(?i)(api[_-]?key|api[_-]?secret|password)\s*[:=]\s*['\"][A-Za-z0-9/+=_-]{16,}['\"]"),
            re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"), re.compile(r"\b\d{9,10}:[A-Za-z0-9_-]{35}\b")]
    skip = {".git", "var", "__pycache__", ".pytest_cache", "node_modules"}
    hits = []
    for f in ROOT.rglob("*"):
        if f.is_dir() or any(p in skip for p in f.parts) or f.suffix in (".db", ".pyc", ".parquet", ".enc"):
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{f}: {m.group(0)[:30]}" for p in pats for m in p.finditer(text)]
    assert hits == []


def test_env_example_has_no_values():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.startswith("SENTINEL_MASTER_PASSWORD"):
            assert line.strip() == "SENTINEL_MASTER_PASSWORD="
    assert os.path.exists(ROOT / ".env.example")
