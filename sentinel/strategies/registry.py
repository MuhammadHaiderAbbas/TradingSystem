"""Strategy plugin system.

Strategies are discovered from three places, so new ones drop in without touching
the core:

1. built-ins in :mod:`sentinel.strategies.builtin`
2. any module in ``sentinel/strategies/plugins/`` (or a directory listed in
   ``SENTINEL_STRATEGY_PATH``) that defines ``Strategy`` subclasses
3. installed packages exposing the ``sentinel.strategies`` entry-point group
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import pkgutil
from importlib.metadata import entry_points
from pathlib import Path

from sentinel.monitoring.logging import get_logger
from sentinel.strategies.base import Strategy
from sentinel.strategies.builtin import BUILTIN

log = get_logger("strategies")

_REGISTRY: dict[str, type[Strategy]] = {}
_loaded = False


def register(cls: type[Strategy]) -> type[Strategy]:
    if not (inspect.isclass(cls) and issubclass(cls, Strategy)) or inspect.isabstract(cls):
        raise TypeError(f"{cls!r} is not a concrete Strategy")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate strategy name {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def _scan_module(mod) -> None:
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, Strategy) and obj is not Strategy and not inspect.isabstract(obj) \
                and obj.__module__ == mod.__name__:
            register(obj)


def _load_dir(path: Path) -> None:
    for f in sorted(path.glob("*.py")):
        if f.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"sentinel_plugin_{f.stem}", f)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _scan_module(mod)


def load_all(force: bool = False) -> dict[str, type[Strategy]]:
    global _loaded
    if _loaded and not force:
        return _REGISTRY
    for cls in BUILTIN:
        register(cls)
    from sentinel.strategies import plugins

    for info in pkgutil.iter_modules(plugins.__path__):
        _scan_module(importlib.import_module(f"sentinel.strategies.plugins.{info.name}"))
    for d in filter(None, os.environ.get("SENTINEL_STRATEGY_PATH", "").split(os.pathsep)):
        if Path(d).is_dir():
            _load_dir(Path(d))
    try:
        for ep in entry_points(group="sentinel.strategies"):
            try:
                register(ep.load())
            except Exception as e:
                log.error("failed to load strategy plugin", extra={"plugin": ep.name, "err": str(e)})
    except Exception:
        pass
    _loaded = True
    return _REGISTRY


def get(name: str) -> type[Strategy]:
    reg = load_all()
    if name not in reg:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(reg)}")
    return reg[name]


def create(name: str, **params) -> Strategy:
    return get(name)(**params)


def available() -> list[str]:
    return sorted(load_all())
