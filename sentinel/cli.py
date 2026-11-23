"""Sentinel command-line interface (typer).

Core: init, backtest, paper, live, status, kill, unlock, gate.
Plus: data, vault, keycheck, ml, dashboard, report, recon, cap, ops, strategies, demo.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import typer

from sentinel import __version__
from sentinel.config import Settings, StrategyEntry, dump_default_config, load_config
from sentinel.core.errors import ConfigError, KeySafetyError, ModeLockedError, SentinelError
from sentinel.core.modes import HALT_KEY, KILL_KEY, LIVE_CONFIRMATION, ModeManager
from sentinel.core.state import StateStore
from sentinel.core.types import Mode
from sentinel.monitoring.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Sentinel: private, paper-first trading bot.")
data_app = typer.Typer(no_args_is_help=True, help="Market data: download, check, report.")
vault_app = typer.Typer(no_args_is_help=True, help="Encrypted secrets vault.")
ml_app = typer.Typer(no_args_is_help=True, help="Machine-learning filter (meta-labelling).")
recon_app = typer.Typer(no_args_is_help=True, help="Reconciliation errors.")
cap_app = typer.Typer(no_args_is_help=True, help="LIVE capital cap.")
ops_app = typer.Typer(no_args_is_help=True, help="Operations: backups, time sync.")
funds_app = typer.Typer(no_args_is_help=True, help="Deposits, withdrawals and profit (money stays at your exchange).")
for sub, name in ((data_app, "data"), (vault_app, "vault"), (ml_app, "ml"), (recon_app, "recon"), (cap_app, "cap"),
                  (ops_app, "ops"), (funds_app, "funds")):
    app.add_typer(sub, name=name)

log = get_logger("cli")
CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to YAML config (default: $SENTINEL_CONFIG or "
                                                         "config/sentinel.yaml)")


def _echo(msg: str, err: bool = False) -> None:
    typer.echo(msg, err=err)


def _load(config: Path | None) -> tuple[Settings, StateStore]:
    try:
        s = load_config(config)
    except ConfigError as e:
        _echo(f"Configuration error - refusing to start.\n{e}", err=True)
        raise typer.Exit(2) from e
    setup_logging(s.logging.level, s.logging.dir, s.logging.max_bytes, s.logging.backup_count, console=False)
    store = StateStore(s.home / "state.db")
    return s, store


def _demo_settings(s: Settings, strategy: str | None = None, params: dict | None = None):
    from sentinel.data.synthetic import demo_universe

    data = demo_universe(timeframe="1h")
    upd = {"market": s.market.model_copy(update={"symbols": list(data), "timeframe": "1h", "venue": "sim"})}
    if strategy:
        upd["strategies"] = [StrategyEntry(name=strategy, params=params or {})]
        upd["combiner"] = s.combiner.model_copy(update={"method": "none"})
    return s.model_copy(update=upd), data


def _load_market(s: Settings, symbols: list[str] | None = None) -> dict:
    from sentinel.data.storage import OHLCVStore

    store = OHLCVStore(s.data.db_path)
    out = {}
    for sym in symbols or s.market.symbols:
        df = store.load(sym, s.market.timeframe, start=s.data.history_start)
        if len(df) < 500:
            _echo(f"Not enough stored data for {sym} {s.market.timeframe} ({len(df)} bars). "
                  f"Run `sentinel data download` or use --demo.", err=True)
            raise typer.Exit(1)
        out[sym] = df.drop(columns=["flags"], errors="ignore")
    return out


def _print_json(d) -> None:
    _echo(json.dumps(d, indent=2, default=str))


# ===================================================================== init
@app.command()
def init(config: Path | None = CONFIG_OPT, force: bool = typer.Option(False, help="Overwrite existing config"),
         no_vault: bool = typer.Option(False, help="Skip vault creation")) -> None:
    """Create config, state directories and the encrypted vault."""
    cfg = Path(config or os.environ.get("SENTINEL_CONFIG", "config/sentinel.yaml"))
    example = Path(__file__).resolve().parents[1] / "config" / "sentinel.example.yaml"
    if cfg.exists() and not force:
        _echo(f"config exists: {cfg} (use --force to overwrite)")
    else:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        if example.exists():
            shutil.copy(example, cfg)
        else:
            cfg.write_text(dump_default_config())
        _echo(f"wrote {cfg}")
    s, store = _load(cfg)
    for d in (s.home, Path(s.logging.dir), Path(s.ml.registry_dir), s.home / "reports", s.home / "backups"):
        Path(d).mkdir(parents=True, exist_ok=True)
    os.chmod(s.home, 0o700)
    if not no_vault:
        from sentinel.security.vault import Vault, keyring_store

        vp = s.home / "vault.json"
        if vp.exists():
            _echo(f"vault exists: {vp}")
        else:
            pw = os.environ.get("SENTINEL_MASTER_PASSWORD") or getpass.getpass("New vault master password (>= 8 chars): ")
            if not os.environ.get("SENTINEL_MASTER_PASSWORD") and getpass.getpass("Repeat: ") != pw:
                _echo("passwords differ", err=True)
                raise typer.Exit(1)
            Vault.create(vp, pw)
            _echo(f"created encrypted vault {vp}")
            if keyring_store(pw):
                _echo("master password stored in the OS keyring")
    store.set("mode.current", Mode.PAPER.value)
    _echo("Sentinel initialised. Default mode: PAPER. LIVE is locked until `sentinel gate` passes.")


# ===================================================================== vault
def _vault(s: Settings, create: bool = False):
    from sentinel.security.vault import Vault, resolve_master_password

    vp = s.home / "vault.json"
    if not vp.exists() and not create:
        _echo("no vault: run `sentinel init`", err=True)
        raise typer.Exit(1)
    return Vault(vp, resolve_master_password())


@vault_app.command("set")
def vault_set(name: str = typer.Argument(..., help="e.g. binance.api_key, binance.api_secret, telegram.bot_token"),
              config: Path | None = CONFIG_OPT) -> None:
    """Store a secret (value is prompted, never echoed or logged)."""
    s, _ = _load(config)
    v = _vault(s)
    v.set(name, getpass.getpass(f"value for {name}: "))
    _echo(f"stored {name}")


@vault_app.command("list")
def vault_list(config: Path | None = CONFIG_OPT) -> None:
    """List secret names (never values)."""
    s, _ = _load(config)
    for n in _vault(s).names():
        _echo(n)


@vault_app.command("delete")
def vault_delete(name: str, config: Path | None = CONFIG_OPT) -> None:
    s, _ = _load(config)
    _vault(s).delete(name)
    _echo(f"deleted {name}")


# ===================================================================== keycheck
@app.command()
def keycheck(config: Path | None = CONFIG_OPT,
             attest: bool = typer.Option(False, help="Operator attestation for venues that don't report permissions")) -> None:
    """Query the exchange for API key permissions. Refuses keys with withdrawal rights."""
    from sentinel.data.downloader import make_exchange
    from sentinel.security.key_safety import enforce_key_safety, fetch_permissions
    from sentinel.security.vault import exchange_credentials

    s, store = _load(config)
    creds = exchange_credentials(_vault(s), s.market.venue)
    if not creds:
        _echo(f"no credentials for {s.market.venue} in the vault (vault set {s.market.venue}.api_key ...)", err=True)
        raise typer.Exit(1)
    perms = fetch_permissions(make_exchange(s.market.venue, s.market.testnet, creds))
    att = None
    if attest:
        ok1 = typer.confirm("I confirmed in the exchange UI that this key has NO withdrawal permission")
        ok2 = typer.confirm("I confirmed the key is restricted to my server's IP address")
        att = {"no_withdrawal": ok1, "ip_whitelist": ok2, "ts": time.time()}
        store.set("keycheck.attestation", att)
        if ok1 and ok2 and not perms.known:
            perms.can_withdraw, perms.ip_restricted, perms.can_trade = False, True, True
    try:
        warnings = enforce_key_safety(perms, s.app.mode, att or store.get("keycheck.attestation"))
    except KeySafetyError as e:
        store.set("keycheck.result", perms.to_dict() | {"ok": False, "ts": time.time()})
        _echo(f"UNSAFE KEY: {e}", err=True)
        raise typer.Exit(3) from e
    store.set("keycheck.result", perms.to_dict() | {"ok": True, "ts": time.time(), "warnings": warnings})
    _print_json(perms.to_dict() | {"warnings": warnings})


# ===================================================================== data
@data_app.command("download")
def data_download(config: Path | None = CONFIG_OPT, symbol: list[str] | None = typer.Option(None),
                  timeframe: list[str] | None = typer.Option(None, help="1m 5m 15m 1h 4h 1d (repeatable)"),
                  since: str | None = None, funding: bool = False) -> None:
    """Download historical OHLCV (public endpoints, no keys) with resume support."""
    from sentinel.data.downloader import HistoricalDownloader, make_exchange
    from sentinel.data.storage import OHLCVStore

    s, _ = _load(config)
    ex = make_exchange(s.market.venue, testnet=False)  # history from the real public API
    dl = HistoricalDownloader(ex, OHLCVStore(s.data.db_path))
    for sym in symbol or s.market.symbols:
        for tf in timeframe or [s.market.timeframe]:
            n = dl.download(sym, tf, since or s.data.history_start)
            _echo(f"{sym} {tf}: +{n} bars")
        if funding:
            _echo(f"{sym} funding: +{dl.download_funding(sym, 0)} rows")


@data_app.command("report")
def data_report_cmd(config: Path | None = CONFIG_OPT) -> None:
    """Coverage, gaps and last update per symbol."""
    import pandas as pd

    from sentinel.data.report import data_report
    from sentinel.data.storage import OHLCVStore

    s, _ = _load(config)
    rep = data_report(OHLCVStore(s.data.db_path), s.data.outlier_zscore)
    if rep.empty:
        _echo("no data stored yet")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        _echo(rep.to_string(index=False))


# ===================================================================== backtest
@app.command()
def backtest(
    config: Path | None = CONFIG_OPT,
    strategy: str | None = typer.Option(None, help="Strategy name (default: config strategies)"),
    params: str | None = typer.Option(None, help='JSON params, e.g. \'{"fast": 10}\''),
    demo: bool = typer.Option(False, help="Use the bundled synthetic dataset (no data/API keys needed)"),
    report: bool = typer.Option(True, help="Write an HTML report"),
    walk_forward: bool = typer.Option(False, "--walk-forward", help="Walk-forward optimisation"),
    monte_carlo: bool = typer.Option(True, help="Monte Carlo trade resampling"),
    sensitivity: str | None = typer.Option(None, help="Two params for a heatmap, e.g. fast,slow"),
    holdout: bool = typer.Option(False, help="Evaluate on the untouched holdout (do this ONCE, at the end)"),
    save_expectations: bool = typer.Option(False, help="Save metrics as the paper-trading expectation (gate)"),
    out: Path | None = typer.Option(None, help="Report path"),
) -> None:
    """Run an honest backtest (next-bar fills, fees, slippage, risk manager) + robustness tests."""
    from sentinel.backtest import robustness as rb
    from sentinel.backtest.report import render_report

    s, store = _load(config)
    p = json.loads(params) if params else None
    if demo:
        s, data = _demo_settings(s, strategy, p)
    else:
        if strategy:
            s = s.model_copy(update={"strategies": [StrategyEntry(name=strategy, params=p or {})],
                                     "combiner": s.combiner.model_copy(update={"method": "none"})})
        data = _load_market(s)
    entry = next(e for e in s.strategies if e.enabled)
    name, prm = entry.name, entry.params
    dev, hold = rb.split_holdout(data, s.backtest.holdout_fraction)
    _echo(f"Backtesting {name} {prm or ''} on {', '.join(data)} ({s.market.timeframe}); holdout "
          f"({s.backtest.holdout_fraction:.0%}) from {hold['__cut__']} is excluded from everything below.")
    multi = len([e for e in s.strategies if e.enabled]) > 1
    if multi:
        from sentinel.backtest.engine import run_backtest

        res = run_backtest(dev, s)
    else:
        res = rb.backtest_window(dev, s, name, prm, None, None)
    wf = mc = sens = None
    union_len = len(next(iter(dev.values())))
    if walk_forward and not multi:
        _echo("walk-forward optimisation ...")
        wf = rb.walk_forward(dev, s, name, train_bars=union_len // 4, test_bars=union_len // 8, max_combos=20)
    trade_r = rb.trade_equity_returns(res)
    if monte_carlo and len(trade_r) >= 2:
        mc = rb.monte_carlo(trade_r, 2000, method="bootstrap")
        mc_dd = rb.monte_carlo(trade_r, 2000, method="shuffle")
        mc.max_drawdown = mc_dd.max_drawdown
        mc.method = "bootstrap (returns) + shuffle (drawdown)"
    if sensitivity and not multi:
        x, y = sensitivity.split(",")
        _echo(f"sensitivity {x} x {y} ...")
        sens = rb.sensitivity(dev, s, name, x.strip(), y.strip(), base_params=prm)
    ho_res, n_evals = None, None
    if holdout and not multi:
        ho_res, n_evals = rb.evaluate_holdout(hold, s, name, prm, store)
    periods = rb.by_period(res, next(iter(dev.values()))["close"], window_bars=24 * 30 if s.market.timeframe == "1h" else 30)
    per_sym = rb.per_symbol(dev, s, name, prm) if (len(dev) > 1 and not multi) else None
    warn = rb.overfit_warnings(res.metrics, wf, sens, mc, n_evals)
    m = res.metrics
    _echo(json.dumps({k: m.get(k) for k in ("total_return", "benchmark_total_return", "cagr", "sharpe", "sortino",
                                            "max_drawdown", "trades", "win_rate", "profit_factor", "expectancy",
                                            "fees_paid")}, indent=2, default=str))
    if wf:
        _echo(f"walk-forward: mean IS {wf.objective} {wf.is_mean:.2f} | mean OOS {wf.oos_mean:.2f} | "
              f"stitched OOS return {wf.oos_metrics['total_return']:.2%}")
    if ho_res:
        _echo(f"holdout: return {ho_res.metrics['total_return']:.2%}, sharpe {ho_res.metrics['sharpe']:.2f}, "
              f"trades {ho_res.metrics['trades']} (holdout configs evaluated so far: {n_evals})")
    for w in warn.warnings:
        _echo(f"[{w.level.upper()}] {w.message}")
    if report:
        path = out or (s.home / "reports" / f"backtest_{name}_{time.strftime('%Y%m%d_%H%M%S')}.html")
        render_report(res, f"Backtest: {name}" + (" (synthetic demo data)" if demo else ""), path, warn.to_list(), wf,
                      mc, sens, periods, per_sym, ho_res,
                      notes="SYNTHETIC DATA: results say nothing about real markets." if demo else "")
        _echo(f"report: {path}")
    if save_expectations:
        oos_exp = None
        if wf is not None and len(wf.oos_metrics):
            oos_exp = wf.oos_metrics.get("expectancy_pct")
        elif ho_res is not None:
            oos_exp = ho_res.metrics.get("expectancy_pct")
        span_w = max((res.equity.index[-1] - res.equity.index[0]).total_seconds() / (7 * 86400), 1e-9)
        exp = {"strategy": name, "params": prm, "expectancy_pct": m.get("expectancy_pct"), "win_rate": m.get("win_rate"),
               "max_drawdown": m.get("max_drawdown"), "sharpe": m.get("sharpe"),
               "trades_per_week": m.get("trades", 0) / span_w, "oos_expectancy_pct": oos_exp, "ts": time.time(),
               "demo": demo}
        store.set("backtest.expectations", exp)
        _echo("saved backtest expectations for the readiness gate" +
              ("" if oos_exp is not None else " (no OOS expectancy: re-run with --walk-forward or --holdout)"))


# ===================================================================== paper / live
def _paper_demo(s: Settings, store: StateStore, max_bars: int, history_bars: int = 1500) -> None:
    from sentinel.core.runner import Runner
    from sentinel.data.live_feed import ReplayFeed
    from sentinel.data.timeframes import tf_seconds
    from sentinel.execution.paper import PaperBroker

    s, data = _demo_settings(s)
    split = len(next(iter(data.values()))) - max_bars
    history = {k: v.iloc[:split].tail(history_bars) for k, v in data.items()}
    live = {k: v.iloc[split:] for k, v in data.items()}
    sim_now = [float(next(iter(live.values())).index[0].timestamp())]
    bar_s = tf_seconds(s.market.timeframe)
    broker = PaperBroker(s.backtest.initial_cash, s.costs, bar_seconds=bar_s, clock=lambda: sim_now[0])
    runner = Runner(s, broker, store, Mode.PAPER, history=history, clock=lambda: sim_now[0])

    def on_bar(sym, bar):
        sim_now[0] = bar.ts + bar_s
        runner.on_bar(sym, bar)

    _echo(f"paper (demo replay): {max_bars} synthetic bars x {len(live)} symbols ...")
    feed = ReplayFeed(live, s.market.timeframe, on_bar)
    try:
        feed.run()
    finally:
        runner.stop()
    bal = broker.balance()
    _echo(f"done: equity {bal['equity']:.2f}, positions {broker.positions()}, trades {len(store.trades('PAPER'))}")


def _live_loop(s: Settings, store: StateStore, mode: Mode, broker, history: dict, use_ws: bool) -> None:
    from apscheduler.schedulers.background import BackgroundScheduler

    from sentinel.core.runner import Runner
    from sentinel.data.downloader import make_exchange
    from sentinel.data.live_feed import PollingFeed, WebSocketFeed
    from sentinel.ops.ops import performance_report

    alerter = _alerter(s)
    runner = Runner(s, broker, store, mode, history=history, alerter=alerter)
    if mode is Mode.LIVE:
        from sentinel.gate.readiness import current_cap

        runner.trader.capital_cap = current_cap(store)
        _echo(f"LIVE capital cap: {runner.trader.capital_cap}")
    lock = threading.Lock()

    def on_bar(sym, bar):
        with lock:
            runner.on_bar(sym, bar)

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(lambda: _locked(lock, runner.tick), "interval", seconds=2, id="kill_and_funds")
    sched.add_job(lambda: _locked(lock, lambda: runner.maybe_reconcile(force=True)), "interval",
                  minutes=s.execution.reconcile_interval_min, id="reconcile")
    sched.add_job(lambda: performance_report(store, mode.value, 1, s.home / "reports" / "daily.html") and (
        alerter.send("daily_summary", f"equity {broker.balance()['equity']:.2f}") if alerter else None),
        "cron", **_cron(s.alerts.daily_summary_cron), id="daily")
    if s.ml.enabled:
        sched.add_job(lambda: _retrain(s, store), "cron", **_cron(s.ml.retrain_cron), id="retrain")
    sched.start()
    runner.maybe_reconcile(force=True)
    ex = make_exchange(s.market.venue, s.market.testnet)
    if use_ws:
        import asyncio

        import ccxt.pro as ccxtpro  # type: ignore

        feed = WebSocketFeed(getattr(ccxtpro, s.market.venue)({"enableRateLimit": True}), s.market.symbols,
                             s.market.timeframe, on_bar)
        asyncio.run(feed.run())
    else:
        feed = PollingFeed(ex, s.market.symbols, s.market.timeframe, on_bar)
        watchdog = threading.Thread(target=_watch_feed, args=(feed, runner), daemon=True)
        watchdog.start()
        try:
            feed.run()
        finally:
            runner.stop()


def _locked(lock, fn):
    with lock:
        return fn()


def _watch_feed(feed, runner) -> None:
    while True:
        time.sleep(30)
        runner.check_feed(feed.seconds_since_heartbeat())


def _cron(expr: str) -> dict:
    m, h, dom, mon, dow = expr.split()
    return {"minute": m, "hour": h, "day": dom, "month": mon, "day_of_week": dow}


def _retrain(s: Settings, store: StateStore) -> None:
    from sentinel.ml.pipeline import train_and_evaluate

    try:
        train_and_evaluate(_load_market(s), s, store=store)
    except Exception as e:  # never let retraining crash the trading loop
        log.error("scheduled retrain failed", extra={"err": str(e)})


def _alerter(s: Settings):
    from sentinel.monitoring.alerts import Alerter

    if not (s.alerts.telegram.enabled or s.alerts.email.enabled):
        return None
    v = _vault(s)
    tok = v.get_optional("telegram.bot_token")
    pw = v.get_optional("email.password")
    return Alerter(s.alerts, tok.reveal() if tok else None, pw.reveal() if pw else None)


def _history(s: Settings, ex) -> dict:
    import pandas as pd

    from sentinel.data.storage import OHLCVStore

    st = OHLCVStore(s.data.db_path)
    out = {}
    for sym in s.market.symbols:
        df = st.load(sym, s.market.timeframe)
        if len(df) < 500:
            rows = ex.fetch_ohlcv(sym, s.market.timeframe, limit=1000)
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
            df = df.iloc[:-1]  # drop the forming candle
        out[sym] = df.drop(columns=["flags"], errors="ignore")
    return out


@app.command()
def paper(config: Path | None = CONFIG_OPT,
          demo: bool = typer.Option(False, help="Replay synthetic bars offline (no network, no keys)"),
          max_bars: int = typer.Option(500, help="Demo: number of bars to replay"),
          ws: bool = typer.Option(False, help="Use WebSocket feed (ccxt.pro) instead of REST polling")) -> None:
    """Paper trade: live public prices, simulated fills, real risk manager. No API keys needed."""
    s, store = _load(config)
    try:
        ModeManager(s, store).require(Mode.PAPER)
    except SentinelError as e:
        _echo(str(e), err=True)
        raise typer.Exit(4) from e
    if demo:
        _paper_demo(s, store, max_bars)
        return
    from sentinel.data.downloader import make_exchange
    from sentinel.data.timeframes import tf_seconds
    from sentinel.execution.paper import PaperBroker

    ex = make_exchange(s.market.venue, testnet=False)  # public market data only

    def quote(sym):
        t = ex.fetch_ticker(sym)
        return float(t["bid"]), float(t["ask"])

    broker = PaperBroker(s.backtest.initial_cash, s.costs, bar_seconds=tf_seconds(s.market.timeframe),
                         quote_source=quote)
    _echo(f"PAPER trading {s.market.symbols} on {s.market.venue} public data. Ctrl+C to stop.")
    _live_loop(s, store, Mode.PAPER, broker, _history(s, ex), ws)


@app.command()
def live(config: Path | None = CONFIG_OPT,
         confirm: str = typer.Option("", help=f"Must be exactly: '{LIVE_CONFIRMATION}'"),
         ws: bool = typer.Option(False)) -> None:
    """LIVE trading. Locked until the readiness gate passes; starts with a capital cap."""
    from sentinel.data.downloader import make_exchange
    from sentinel.execution.ccxt_broker import CcxtBroker
    from sentinel.security.key_safety import enforce_key_safety
    from sentinel.security.vault import exchange_credentials

    s, store = _load(config)
    try:
        ModeManager(s, store).require(Mode.LIVE, confirm)
        creds = exchange_credentials(_vault(s), s.market.venue)
        ex = make_exchange(s.market.venue, s.market.testnet, creds)
        broker = CcxtBroker(ex, s.market.symbols, store, s.market.quote_currency)
        enforce_key_safety(broker.key_permissions(), Mode.LIVE, store.get("keycheck.attestation"))
    except (SentinelError, KeySafetyError, ModeLockedError) as e:
        _echo(f"LIVE refused: {e}", err=True)
        raise typer.Exit(5) from e
    store.log_event("live_start", "LIVE mode started", level="WARNING", mode="LIVE")
    _live_loop(s, store, Mode.LIVE, broker, _history(s, ex), ws)


# ===================================================================== status / kill / unlock
@app.command()
def status(config: Path | None = CONFIG_OPT, as_json: bool = typer.Option(False, "--json")) -> None:
    """Mode, locks, equity, positions, health."""
    from sentinel.gate.readiness import current_cap
    from sentinel.monitoring.health import system_health

    s, store = _load(config)
    mode = store.get("mode.current") or s.app.mode.value
    eq = store.equity_curve(mode)
    bstate = store.get(f"paper.broker_state.{mode}") or {}
    ok, why = ModeManager(s, store).live_unlocked()
    d = {
        "version": __version__, "mode": mode, "kill_switch": store.get(KILL_KEY), "halt": store.get(HALT_KEY),
        "equity": eq[-1]["equity"] if eq else None, "equity_points": len(eq),
        "positions": {k: v["qty"] for k, v in (bstate.get("positions") or {}).items()},
        "open_orders": len(store.orders(mode, open_only=True)), "closed_trades": len(store.trades(mode)),
        "live_unlocked": ok, "live_lock_reason": None if ok else why, "live_capital_cap": current_cap(store),
        "unresolved_reconciliation_errors": len(store.unresolved_recon_errors()),
        "health": system_health(store, s.data.stale_feed_seconds)["components"],
    }
    if as_json:
        _print_json(d)
        return
    for k, v in d.items():
        _echo(f"{k:34s} {json.dumps(v, default=str) if isinstance(v, (dict, list)) else v}")


@app.command()
def kill(config: Path | None = CONFIG_OPT,
         close: bool = typer.Option(True, "--close/--no-close", help="Also close all positions at market"),
         test: bool = typer.Option(False, help="Drill: records that the kill switch was tested"),
         direct: bool = typer.Option(True, help="If no runner is active, execute immediately against the venue"),
         yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """KILL SWITCH: cancel all orders, optionally close positions, lock the bot."""
    s, store = _load(config)
    if not yes and not typer.confirm(f"Engage kill switch (close positions: {close})?"):
        raise typer.Exit(1)
    mode = Mode(store.get("mode.current") or s.app.mode.value)
    state = {"ts": time.time(), "reason": "CLI kill" + (" (test)" if test else ""), "close_positions": close,
             "executed": False, "test": test}
    store.set(KILL_KEY, state)
    store.log_event("kill", state["reason"], level="CRITICAL", mode=mode.value, data=state)
    runner_active = _runner_alive(store)
    if runner_active:
        _echo("kill switch engaged; the running bot will execute it within ~2 seconds")
        for _ in range(15):
            time.sleep(1)
            if (store.get(KILL_KEY) or {}).get("executed"):
                _echo(f"executed: {store.get(KILL_KEY).get('result')}")
                break
        return
    if direct:
        res = _direct_kill(s, store, mode, close)
        state.update(executed=True, executed_ts=time.time(), result=res)
        store.set(KILL_KEY, state)
        if test:
            store.set("kill.tested", {"ts": time.time(), "mode": mode.value, "result": res})
        _echo(f"kill executed directly: {res}")
    _echo("bot LOCKED. Investigate, then `sentinel unlock`.")


def _runner_alive(store: StateStore) -> bool:
    hb = store.heartbeats().get("runner")
    if not hb or hb.get("stopped") or time.time() - hb["ts"] > 600:
        return False
    pid = hb.get("pid")
    if not pid or pid == os.getpid():
        return False
    try:
        os.kill(int(pid), 0)  # signal 0: existence check only
    except (ProcessLookupError, PermissionError, ValueError):
        return False
    return True


def _direct_kill(s: Settings, store: StateStore, mode: Mode, close: bool) -> dict:
    from sentinel.execution.order_manager import OrderManager
    from sentinel.execution.paper import PaperBroker
    from sentinel.risk.manager import RiskManager

    if mode is Mode.LIVE:
        from sentinel.data.downloader import make_exchange
        from sentinel.execution.ccxt_broker import CcxtBroker
        from sentinel.security.vault import exchange_credentials

        ex = make_exchange(s.market.venue, s.market.testnet, exchange_credentials(_vault(s), s.market.venue))
        broker = CcxtBroker(ex, s.market.symbols, store, s.market.quote_currency)
    else:
        broker = PaperBroker(s.backtest.initial_cash, s.costs)
        st = store.get(f"paper.broker_state.{mode.value}")
        if st:
            broker.restore_state(st)
    risk = RiskManager(s.risk, store, mode)
    om = OrderManager(broker, risk, s.execution, store, mode)
    if isinstance(broker, PaperBroker):
        # no live quotes here: close at the last mark (+ slippage) so the paper book is flat
        for sym in broker.positions():
            m = broker.portfolio.marks.get(sym)
            if m:
                broker.quotes[sym] = (m, m)
    res = om.kill(close_positions=close, reason="CLI kill (direct)")
    if isinstance(broker, PaperBroker):
        store.set(f"paper.broker_state.{mode.value}", broker.to_state())
    return res


@app.command()
def unlock(config: Path | None = CONFIG_OPT, yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Manual re-enable after a halt or kill (enforces the cooldown)."""
    from sentinel.risk.manager import RiskManager

    s, store = _load(config)
    k, h = store.get(KILL_KEY), store.get(HALT_KEY)
    if not k and not h:
        _echo("nothing to unlock")
        return
    _echo(f"kill: {k}\nhalt: {h}")
    unresolved = store.unresolved_recon_errors()
    if unresolved:
        _echo(f"{len(unresolved)} unresolved reconciliation errors: resolve them first (`sentinel recon`)", err=True)
        raise typer.Exit(1)
    if not yes and not typer.confirm("Have you investigated the cause and want to re-enable trading?"):
        raise typer.Exit(1)
    mode = Mode(store.get("mode.current") or s.app.mode.value)
    ts = max((k or {}).get("ts", 0), (h or {}).get("ts", 0))
    if time.time() - ts < s.risk.cooldown_minutes * 60:
        _echo(f"cooldown active: {(s.risk.cooldown_minutes * 60 - (time.time() - ts)) / 60:.1f} min left", err=True)
        raise typer.Exit(1)
    store.delete(KILL_KEY)
    RiskManager(s.risk, store, mode).reenable("manual unlock via CLI", force_cooldown=False)
    store.log_event("unlock", "manual re-enable via CLI", level="WARNING")
    _echo("unlocked")


# ===================================================================== gate / cap / recon
@app.command()
def gate(config: Path | None = CONFIG_OPT, skip_tests: bool = typer.Option(False, help="Don't run pytest "
                                                                                         "(the check then fails)")) -> None:
    """Run the LIVE readiness gate. ALL checks must pass to unlock LIVE."""
    from sentinel.gate.readiness import evaluate_gate, record_gate

    s, store = _load(config)
    rep = evaluate_gate(s, store, run_test_suite=not skip_tests)
    for c in rep.checks:
        _echo(f"[{'PASS' if c.passed else 'FAIL'}] {c.name:24s} {c.detail}")
    record_gate(s, store, rep)
    _echo("\nGATE PASSED: LIVE may be enabled (with capital cap)." if rep.passed else "\nGATE FAILED: LIVE stays locked.")
    if not rep.passed:
        raise typer.Exit(6)


@cap_app.command("show")
def cap_show(config: Path | None = CONFIG_OPT) -> None:
    s, store = _load(config)
    _print_json(store.get("live.capital_cap"))


@cap_app.command("raise")
def cap_raise(amount: float, config: Path | None = CONFIG_OPT) -> None:
    """Raise (or lower) the LIVE capital cap, in limited steps."""
    from sentinel.gate.readiness import raise_cap

    s, store = _load(config)
    ok, msg = raise_cap(s, store, amount)
    _echo(msg, err=not ok)
    if not ok:
        raise typer.Exit(1)


@recon_app.command("list")
def recon_list(config: Path | None = CONFIG_OPT) -> None:
    s, store = _load(config)
    _print_json(store.unresolved_recon_errors())


@recon_app.command("resolve")
def recon_resolve(err_id: int, note: str = typer.Option(..., help="What you found and fixed"),
                  config: Path | None = CONFIG_OPT) -> None:
    s, store = _load(config)
    store.resolve_recon_error(err_id, note)
    _echo(f"resolved {err_id}")


# ===================================================================== ml
@ml_app.command("train")
def ml_train(config: Path | None = CONFIG_OPT, demo: bool = False, strategy: str | None = None,
             no_deploy: bool = False) -> None:
    """Train + evaluate the meta-label filter; deploys only if it beats the baseline OOS after costs."""
    from sentinel.ml.pipeline import train_and_evaluate

    s, store = _load(config)
    if demo:
        s, data = _demo_settings(s, strategy)
    else:
        data = _load_market(s)
    v = train_and_evaluate(data, s, strategy, store=store, deploy=not no_deploy)
    _print_json({k: v[k] for k in ("version", "beats_baseline", "deployed", "message", "scores", "oos_baseline",
                                   "oos_filtered", "importance")})


@ml_app.command("status")
def ml_status(config: Path | None = CONFIG_OPT) -> None:
    from sentinel.ml.pipeline import ml_allowed
    from sentinel.ml.registry import ModelRegistry

    s, store = _load(config)
    ok, why = ml_allowed(s, store)
    reg = ModelRegistry(s.ml.registry_dir)
    _print_json({"ml_active": ok, "reason": why, "current": reg.current_version(),
                 "versions": [v["version"] for v in reg.versions()], "last_verdict": store.get("ml.verdict")})


# ===================================================================== dashboard / report
@app.command()
def dashboard(config: Path | None = CONFIG_OPT, host: str | None = None, port: int | None = None,
              set_password: bool = typer.Option(False, "--set-password"),
              allow_remote: bool = typer.Option(False, help="Allow non-loopback bind (Docker only)")) -> None:
    """Local, password-protected dashboard (default http://127.0.0.1:8765)."""
    from sentinel.dashboard.app import PW_KEY, check_bind, create_app
    from sentinel.dashboard.app import set_password as _setpw

    s, store = _load(config)
    if set_password:
        pw = os.environ.get("SENTINEL_DASHBOARD_PASSWORD") or getpass.getpass("New dashboard password (>= 10 chars): ")
        _setpw(store, pw)
        _echo("dashboard password set")
        return
    if not store.get(PW_KEY):
        _echo("no dashboard password: run `sentinel dashboard --set-password` first", err=True)
        raise typer.Exit(1)
    h, p = host or s.dashboard.host, port or s.dashboard.port
    check_bind(h, allow_remote)
    import uvicorn

    _echo(f"dashboard on http://{h}:{p} (user: {s.dashboard.username})")
    uvicorn.run(create_app(s, store), host=h, port=p, log_level="warning", server_header=False)


@app.command()
def report(config: Path | None = CONFIG_OPT, period: str = typer.Option("daily", help="daily | weekly"),
           mode: str | None = None) -> None:
    """Write a daily/weekly HTML performance report from the paper/live log."""
    from sentinel.ops.ops import performance_report

    s, store = _load(config)
    m = mode or store.get("mode.current") or s.app.mode.value
    days = 1 if period == "daily" else 7
    out = performance_report(store, m, days, s.home / "reports" / f"{period}_{m}_{time.strftime('%Y%m%d')}.html")
    _echo(f"report: {out}")


# ===================================================================== ops
@ops_app.command("timecheck")
def ops_timecheck(max_offset: float = 1.0) -> None:
    """Check the system clock against NTP (exchanges reject requests with skewed timestamps)."""
    from sentinel.ops.ops import check_time_sync

    r = check_time_sync(max_offset)
    _print_json(r)
    if not r["ok"]:
        raise typer.Exit(1)


@ops_app.command("backup")
def ops_backup(config: Path | None = CONFIG_OPT, keep: int = 14) -> None:
    """Encrypted backup of config + state/market DBs + vault file (password: $SENTINEL_BACKUP_PASSWORD)."""
    from sentinel.ops.ops import create_backup, prune_backups

    s, store = _load(config)
    pw = os.environ.get("SENTINEL_BACKUP_PASSWORD") or getpass.getpass("Backup password: ")
    paths = [Path(config or os.environ.get("SENTINEL_CONFIG", "config/sentinel.yaml")), s.home / "state.db",
             Path(s.data.db_path), s.home / "vault.json", Path(s.ml.registry_dir)]
    out = create_backup(paths, s.home / "backups", pw)
    pruned = prune_backups(s.home / "backups", keep)
    _echo(f"backup: {out} (pruned {pruned} old)")


@ops_app.command("restore")
def ops_restore(path: Path, dest: Path = typer.Option(Path("restore"))) -> None:
    from sentinel.ops.ops import restore_backup

    pw = os.environ.get("SENTINEL_BACKUP_PASSWORD") or getpass.getpass("Backup password: ")
    names = restore_backup(path, pw, dest)
    _echo(f"restored {len(names)} files into {dest}")


# ===================================================================== funds
def _funds_ctx(config):
    from sentinel.funds import Funds

    s, store = _load(config)
    mode = Mode(store.get("mode.current") or s.app.mode.value)
    return s, store, mode, Funds(store, mode.value, s.funds)


def _balances(s: Settings, store: StateStore, mode: Mode) -> tuple[float | None, float | None]:
    st = store.get(f"paper.broker_state.{mode.value}") if mode is Mode.PAPER else None
    if st:  # PAPER: the saved paper account is the source of truth
        return st["cash"] + sum(p["qty"] * st["marks"].get(sym, p["avg_price"]) for sym, p in st["positions"].items()), \
            st["cash"]
    eq = store.equity_curve(mode.value)
    if eq:
        return eq[-1]["equity"], eq[-1]["cash"]
    return (s.backtest.initial_cash, s.backtest.initial_cash) if mode is Mode.PAPER else (None, None)


def _apply_offline(s: Settings, store: StateStore, mode: Mode, funds) -> None:
    """No bot running: apply recorded flows now (PAPER: to the saved paper account; both: risk references)."""
    if _runner_alive(store):
        _echo("the running bot will apply this within a few seconds")
        return
    from sentinel.execution.paper import PaperBroker
    from sentinel.risk.manager import RiskManager

    risk = RiskManager(s.risk, store, mode)
    broker = None
    if mode is Mode.PAPER:
        broker = PaperBroker(s.backtest.initial_cash, s.costs)
        st = store.get(f"paper.broker_state.{mode.value}")
        if st:
            broker.restore_state(st)
        funds.ensure_opening(broker.balance()["equity"])
    last = store.equity_curve(mode.value)
    ts = (last[-1]["ts"] + 1) if last else None  # stay on the bot's clock (paper may replay old data)
    done = funds.apply_pending(broker, risk, ts=ts)
    if broker is not None and done:
        store.set(f"paper.broker_state.{mode.value}", broker.to_state())
        if last:
            bal = broker.balance()
            store.record_equity(mode.value, ts, bal["equity"], bal["cash"], broker.portfolio.exposure_pct())
    funds.refresh_ready(_balances(s, store, mode)[1] or 0.0)


def _money(v, cur):
    return "–" if v is None else f"{v:,.2f} {cur}"


@funds_app.command("status")
def funds_status(config: Path | None = CONFIG_OPT, as_json: bool = typer.Option(False, "--json")) -> None:
    """Invested capital vs profit, what you can withdraw now, open withdrawal requests."""
    s, store, mode, funds = _funds_ctx(config)
    eq, cash = _balances(s, store, mode)
    sm = funds.summary(eq, cash, store.equity_curve(mode.value))
    sm["open_withdrawals"] = funds.flows(("requested", "ready"), "withdrawal")
    if as_json:
        _print_json(sm)
        return
    c = sm["currency"]
    _echo(f"mode {mode.value} · funds stay at your exchange; Sentinel only records and reserves")
    for label, key in (("opening balance", "opening"), ("deposits", "deposits"), ("withdrawals", "withdrawals"),
                       ("net invested", "net_invested"), ("account value", "equity"), ("free cash", "cash")):
        _echo(f"{label:28s} {_money(sm[key], c)}")
    _echo(f"{'profit / loss':28s} {_money(sm['profit'], c)} ({sm['profit_pct']:+.2%} on invested)")
    if sm["time_weighted_return"] is not None:
        _echo(f"{'time-weighted return':28s} {sm['time_weighted_return']:+.2%}")
    _echo(f"{'reserved for withdrawal':28s} {_money(sm['reserved_for_withdrawal'], c)}")
    _echo(f"{'withdrawable now':28s} {_money(sm['withdrawable_now'], c)} (profit part: "
          f"{_money(sm['withdrawable_profit'], c)})")
    for w in sm["open_withdrawals"]:
        _echo(f"  withdrawal #{w['id']}: {_money(w['amount'], c)} [{w['status']}] {w['note']}")
    if sm["suggestion"]:
        _echo(f"suggestion: {sm['suggestion']}")


@funds_app.command("deposit")
def funds_deposit(amount: float | None = typer.Argument(None, help="Amount that arrived"),
                  config: Path | None = CONFIG_OPT, note: str = "",
                  how: bool = typer.Option(False, "--how", help="Show how to deposit from a UK bank")) -> None:
    """Record a deposit (PAPER: simulated). On a live venue the read-only sync records deposits automatically."""
    s, store, mode, funds = _funds_ctx(config)
    if how:
        _echo(funds.deposit_instructions(s.market.venue))
        return
    if not amount:
        _echo("give an amount, e.g. `sentinel funds deposit 500` (or --how for instructions)", err=True)
        raise typer.Exit(1)
    fid = funds.deposit(amount, "paper" if mode is Mode.PAPER else "manual", note)
    _echo(f"deposit #{fid}: {_money(amount, s.funds.currency)} recorded ({mode.value})")
    _apply_offline(s, store, mode, funds)


@funds_app.command("withdraw")
def funds_withdraw(amount: float, config: Path | None = CONFIG_OPT,
                   profit_only: bool = typer.Option(False, "--profit-only", help="Refuse if it would touch capital"),
                   note: str = "") -> None:
    """Request a withdrawal: reserves the cash so the bot stops trading it, then tells you how to send it
    to your bank on the exchange (you approve it there with 2FA)."""
    from sentinel.funds import FundsError

    s, store, mode, funds = _funds_ctx(config)
    eq, cash = _balances(s, store, mode)
    if eq is None:
        _echo("no account balance known yet: start the bot once (or run `sentinel funds sync`)", err=True)
        raise typer.Exit(1)
    try:
        fid, chk = funds.request_withdrawal(amount, eq, cash, profit_only, note)
    except FundsError as e:
        _echo(f"refused: {e}", err=True)
        raise typer.Exit(1) from e
    _echo(f"withdrawal #{fid}: {_money(amount, s.funds.currency)} reserved [{'ready' if chk.ready else 'requested'}]")
    _echo(chk.message)
    if mode is Mode.PAPER:
        _echo(f"PAPER mode: simulate the bank payout with `sentinel funds confirm {fid}`")
    else:
        _echo(funds.instructions(s.market.venue, amount))


@funds_app.command("confirm")
def funds_confirm(flow_id: int, config: Path | None = CONFIG_OPT, ref: str = typer.Option(None, help="Venue ref")) -> None:
    """Mark a withdrawal as sent to your bank (PAPER: simulates the payout)."""
    from sentinel.funds import FundsError

    s, store, mode, funds = _funds_ctx(config)
    try:
        funds.confirm_withdrawal(flow_id, ref)
    except FundsError as e:
        _echo(f"refused: {e}", err=True)
        raise typer.Exit(1) from e
    _echo(f"withdrawal #{flow_id} completed")
    _apply_offline(s, store, mode, funds)


@funds_app.command("cancel")
def funds_cancel(flow_id: int, config: Path | None = CONFIG_OPT) -> None:
    """Cancel an open withdrawal request and release the reserved cash."""
    from sentinel.funds import FundsError

    s, store, mode, funds = _funds_ctx(config)
    try:
        funds.cancel_withdrawal(flow_id)
    except FundsError as e:
        _echo(f"refused: {e}", err=True)
        raise typer.Exit(1) from e
    _echo(f"withdrawal #{flow_id} cancelled; cash released back to trading")


@funds_app.command("history")
def funds_history(config: Path | None = CONFIG_OPT) -> None:
    s, store, mode, funds = _funds_ctx(config)
    for f in funds.flows():
        _echo(f"#{f['id']:<4} {time.strftime('%Y-%m-%d %H:%M', time.gmtime(f['ts']))}  {f['kind']:10s} "
              f"{f['amount']:>12,.2f} {f['currency']}  {f['status']:9s} {f['source']:8s} {f['note'] or ''}")


@funds_app.command("sync")
def funds_sync(config: Path | None = CONFIG_OPT) -> None:
    """Read deposits/withdrawals from the exchange (read-only key permission is enough)."""
    from sentinel.data.downloader import make_exchange
    from sentinel.security.vault import exchange_credentials

    s, store, mode, funds = _funds_ctx(config)
    ex = make_exchange(s.market.venue, s.market.testnet, exchange_credentials(_vault(s), s.market.venue))
    _print_json(funds.sync_from_exchange(ex))
    _apply_offline(s, store, mode, funds)


@funds_app.command("export")
def funds_export(config: Path | None = CONFIG_OPT, year: int | None = None,
                 out: Path | None = None) -> None:
    """CSV of deposits, withdrawals and trades for your records (UK: HMRC capital gains). Not tax advice."""
    s, store, mode, funds = _funds_ctx(config)
    p = funds.export_csv(out or s.home / "reports" / f"funds_{mode.value}_{year or 'all'}.csv", year)
    _echo(f"wrote {p}")


# ===================================================================== misc
@app.command()
def strategies() -> None:
    """List available strategies (built-ins + plugins) and their parameters."""
    from sentinel.strategies import registry

    for n in registry.available():
        cls = registry.get(n)
        ps = ", ".join(f"{k}={p.default}" for k, p in cls.params.items())
        _echo(f"{n:16s} {cls.description}\n{'':16s} params: {ps or '-'}  regimes: {cls.regimes or 'all'}")


@app.command()
def demo(config: Path | None = CONFIG_OPT, quick: bool = typer.Option(False, help="Skip slow steps")) -> None:
    """End-to-end offline demo on synthetic data: backtest + report, ML check, paper replay, gate."""
    from sentinel.backtest import robustness as rb
    from sentinel.backtest.report import render_report
    from sentinel.gate.readiness import evaluate_gate
    from sentinel.ml.pipeline import train_and_evaluate

    s0, _ = _load(config)
    home = Path(os.environ.get("SENTINEL_DEMO_HOME", s0.home / "demo"))
    s0 = s0.model_copy(update={"app": s0.app.model_copy(update={"home": str(home)}),
                               "ml": s0.ml.model_copy(update={"registry_dir": str(home / "models")})})
    home.mkdir(parents=True, exist_ok=True)
    store = StateStore(home / "state.db")
    s, data = _demo_settings(s0, "ema_adx")
    _echo("== 1/4 backtest (synthetic data, 2 years of 1h bars, 2 symbols) ==")
    dev, hold = rb.split_holdout(data, s.backtest.holdout_fraction)
    res = rb.backtest_window(dev, s, "ema_adx", {}, None, None)
    wf = None if quick else rb.walk_forward(dev, s, "ema_adx", 3500, 1750, max_combos=8)
    tr = rb.trade_equity_returns(res)
    mc = rb.monte_carlo(tr, 1000, method="bootstrap") if len(tr) > 1 else None
    sens = None if quick else rb.sensitivity(dev, s, "ema_adx", "fast", "slow", [8, 12, 16, 24], [36, 48, 72, 96])
    warn = rb.overfit_warnings(res.metrics, wf, sens, mc)
    rp = render_report(res, "Demo backtest: EMA crossover + ADX (synthetic data)", home / "reports" / "demo_backtest.html",
                       warn.to_list(), wf, mc, sens, rb.by_period(res, dev[next(iter(dev))]["close"], 720),
                       rb.per_symbol(dev, s, "ema_adx", {}),
                       notes="SYNTHETIC DATA: a regime-switching random walk. Results say nothing about real markets.")
    m = res.metrics
    _echo(f"return {m['total_return']:.2%} vs buy&hold {m['benchmark_total_return']:.2%}, sharpe {m['sharpe']:.2f}, "
          f"max DD {m['max_drawdown']:.2%}, trades {m['trades']}\nreport: {rp}")
    _echo("== 2/4 ML meta-label filter (deploys only if it beats the baseline OOS) ==")
    try:
        v = train_and_evaluate(data, s.model_copy(update={"strategies": [StrategyEntry(name="rsi_bb")]}), store=store)
        _echo(f"beats baseline: {v['beats_baseline']} -> {v['message']}")
    except ValueError as e:
        _echo(f"ML skipped: {e}")
    _echo("== 3/4 paper trading (replay of the last 300 synthetic bars) ==")
    store.delete(KILL_KEY)
    store.delete(HALT_KEY)
    _paper_demo(s0, store, 300)
    _echo("== 4/4 LIVE readiness gate (expected to FAIL: no 8 weeks of paper trading) ==")
    rep = evaluate_gate(s, store, run_test_suite=False)
    for c in rep.checks:
        _echo(f"[{'PASS' if c.passed else 'FAIL'}] {c.name:24s} {c.detail}")
    _echo(f"\nDemo state in {home}. Open {rp} in a browser.")


def main() -> None:
    try:
        app()
    except SentinelError as e:
        _echo(f"error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
