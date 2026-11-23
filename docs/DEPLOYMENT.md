# Deployment & operations

## Option A: your own machine

```bash
pip install -e ".[all]"
cp config/sentinel.example.yaml config/sentinel.yaml
sentinel init && sentinel ops timecheck
sentinel paper            # leave it running; use tmux, systemd or Docker
```

## Option B: Docker Compose (bot + dashboard, optional TimescaleDB)

```bash
cp config/sentinel.example.yaml config/sentinel.yaml
printf 'SENTINEL_MASTER_PASSWORD=...\n' > docker/.env && chmod 600 docker/.env
docker compose -f docker/docker-compose.yml run --rm bot init
docker compose -f docker/docker-compose.yml run --rm -it dashboard dashboard --set-password
docker compose -f docker/docker-compose.yml up -d
```

* Containers run as a non-root user with a read-only root filesystem, `no-new-privileges`, and all capabilities dropped.
* `restart: unless-stopped` restarts the bot after a crash.
* State lives in the `sentinel-data` volume. The image contains no secrets.
* The dashboard is published on `127.0.0.1:8765` only.

## Option C: private VPS

1. `sudo ./scripts/harden_vps.sh`. This sets up a UFW firewall (SSH only), key-only SSH, fail2ban,
   chrony time sync, unattended security upgrades, and a `sentinel` system user. **Read the
   script first. Make sure your SSH key login works before running it.**
2. Install Sentinel under `/opt/sentinel` in a venv owned by `sentinel`.
3. `sudo cp scripts/sentinel.service /etc/systemd/system/ && sudo systemctl enable --now sentinel`
   (it checks the clock before starting and restarts on failure).
4. Reach the dashboard only through an SSH tunnel: `ssh -L 8765:127.0.0.1:8765 you@vps`, then
   open http://127.0.0.1:8765. Alternatively, use a VPN such as WireGuard or Tailscale. Never open port 8765.
5. Whitelist the VPS IP on the exchange API key.

## Backups

`./scripts/backup.sh` (cron, e.g. daily at 02:17) creates an encrypted archive (argon2id + Fernet)
of the config, a consistent snapshot of the state and market DBs, the vault file (still encrypted
with its own password) and the models. It keeps the last 14. Set `BACKUP_REMOTE=user@host:/path` to rsync
the archives off-machine. Test restoring regularly (see the runbook).

## Update process: test, then paper, then live

`./scripts/update.sh <git-ref>` checks out the new version in a separate worktree and runs the full
test suite. Run the new version in **PAPER**. Any change to strategy, risk, cost or ML config
changes the config fingerprint and **invalidates the readiness gate**, so LIVE stays locked until
the new configuration has passed `sentinel gate` again. Never deploy straight to live.
