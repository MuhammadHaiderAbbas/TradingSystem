# Dashboard

FastAPI with a single static page (no build step). Uses HTTP Basic auth against an argon2 hash, binds to localhost only, and sends security headers.
Shows equity, PnL, positions and orders, the risk panel, backtest vs paper vs live drift, strategy toggles, the trade journal, health and events. The KILL SWITCH requires typing `KILL`.

Run: `sentinel dashboard --set-password && sentinel dashboard`. Verify: `pytest tests/test_dashboard_cli_ops.py`.
