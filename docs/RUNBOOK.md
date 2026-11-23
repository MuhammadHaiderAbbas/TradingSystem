# Runbook

First rule in any incident: **`sentinel kill`** (or the dashboard button) if in doubt. It cancels
every order, closes positions (unless `--no-close`) and locks the bot. Investigate, then
`sentinel unlock` (enforces the cooldown).

## The exchange is down / API errors

Symptoms: `api_errors` halt, `error` events, stale `runner` heartbeat in `sentinel status`.

1. Check the exchange status page. Sentinel auto-halts after `circuit_breakers.max_api_errors`
   errors in 10 minutes and keeps existing protective stops at the exchange.
2. If positions are open and you can't see them, log in to the exchange UI and act manually.
3. When the exchange is back: `sentinel status` → `sentinel recon list` (no mismatches?) →
   `sentinel unlock`.

## The bot halted

`sentinel status` shows `halt.kind`:

| kind | meaning | action |
|---|---|---|
| `daily_loss` / `weekly_loss` | loss limit breached | review trades in the dashboard journal; unlock only next period |
| `drawdown` | max drawdown breached | **stop**. Re-validate the strategy (walk-forward, holdout) before resuming |
| `consecutive_losses` | N losers in a row | check for regime change or data problems |
| `api_errors` | exchange trouble | see above |
| `reconciliation` | position mismatch | see below |
| `protection_failed` | a stop couldn't be placed, position flattened | check exchange order types and min sizes |
| `kill` | kill switch | investigate the reason in `events` |

Halts block new entries only; exits and protective stops keep working.

## Positions mismatch (reconciliation error)

1. `sentinel recon list` shows internal vs venue quantity per symbol.
2. Common causes: a manual trade on the same account, a fill that arrived during a restart,
   deposits or withdrawals of the base asset, fees charged in the base asset.
3. Fix the account (or accept the venue as the truth), then
   `sentinel recon resolve <id> --note "what happened"`, then `sentinel unlock`.
4. Use a dedicated sub-account for Sentinel to avoid this entirely.

## An API key leaked

1. **Revoke the key in the exchange UI immediately.** Because Sentinel refuses keys with
   withdrawal permission, a leaked key can trade but can't withdraw. Still: check open orders and
   recent trades for anything you didn't place.
2. `sentinel kill` locks the bot.
3. Create a new trade-only, IP-whitelisted key; `sentinel vault set <venue>.api_key` / `.api_secret`;
   `sentinel keycheck`.
4. If the vault master password may also be compromised: rotate it (re-create the vault), rotate
   every secret in it, and rotate the backup password.
5. Find the cause: shell history, a committed `.env`, logs (they are redacted, but check), backups.

## The machine crashed / restarted

Sentinel restarts automatically (systemd `Restart=on-failure` or Docker `restart: unless-stopped`).
On start it restores the paper broker state, reloads known order ids (no duplicates), refreshes open
orders from the venue and reconciles. Check `sentinel status` and the `startup` event.

## Clock skew

Exchanges reject signed requests with skewed timestamps. `sentinel ops timecheck` must report
`ok`; install chrony/NTP (the VPS hardening script does).

## Restoring a backup

```bash
SENTINEL_BACKUP_PASSWORD=... sentinel ops restore var/backups/sentinel-backup-<ts>.enc --dest restore/
```
Stop the bot, copy the restored files into place, start the bot, then check `sentinel status`.
