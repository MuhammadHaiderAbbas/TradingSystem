#!/usr/bin/env bash
# Encrypted backup of config + state + market DB + vault + models.
# Schedule it, e.g. crontab:  17 2 * * *  cd /opt/sentinel && ./scripts/backup.sh >> var/logs/backup.log 2>&1
# Requires SENTINEL_BACKUP_PASSWORD in the environment (or prompts). Copy the .enc files
# off-machine (e.g. rsync to another host); they are useless without the password.
set -euo pipefail
cd "$(dirname "$0")/.."
KEEP="${KEEP:-14}"
sentinel ops backup --keep "$KEEP"
if [[ -n "${BACKUP_REMOTE:-}" ]]; then
  rsync -a --chmod=F600 "${SENTINEL_HOME:-var}/backups/" "$BACKUP_REMOTE"
fi
