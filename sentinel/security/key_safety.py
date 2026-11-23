"""API-key safety checker.

On startup Sentinel asks the exchange what the key can do and **refuses to run if
withdrawals are enabled**. It warns when no IP whitelist is set, and in LIVE mode a
missing whitelist or an unknown permission set is also fatal.

ccxt has no unified "key permissions" call, so each venue is handled explicitly:

* Binance: ``GET /sapi/v1/account/apiRestrictions`` -> enableWithdrawals, ipRestrict
* Coinbase Advanced: ``GET /api/v3/brokerage/key_permissions`` -> can_transfer
* Kraken: not exposed by the API; the operator must attest via ``sentinel keycheck --attest``
* Alpaca: trading keys cannot withdraw; IP whitelisting is not offered
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sentinel.core.errors import KeySafetyError
from sentinel.core.types import Mode
from sentinel.monitoring.logging import get_logger

log = get_logger("security.keys")


@dataclass
class KeyPermissions:
    known: bool  # False if the venue doesn't expose permissions
    can_trade: bool = False
    can_withdraw: bool = True  # pessimistic default
    ip_restricted: bool = False
    source: str = ""
    raw: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


def permissions_from_binance(resp: dict) -> KeyPermissions:
    return KeyPermissions(
        known=True,
        can_trade=bool(resp.get("enableSpotAndMarginTrading") or resp.get("enableFutures")),
        can_withdraw=bool(resp.get("enableWithdrawals")) or bool(resp.get("enableInternalTransfer")),
        ip_restricted=bool(resp.get("ipRestrict")),
        source="binance.apiRestrictions",
    )


def permissions_from_coinbase(resp: dict) -> KeyPermissions:
    return KeyPermissions(
        known=True,
        can_trade=bool(resp.get("can_trade")),
        can_withdraw=bool(resp.get("can_transfer")),
        ip_restricted=False,  # not reported by the endpoint
        source="coinbase.key_permissions",
    )


def fetch_permissions(exchange) -> KeyPermissions:  # exchange: ccxt.Exchange
    """Query a ccxt exchange object for key permissions."""
    eid = getattr(exchange, "id", "")
    try:
        if eid in ("binance", "binanceus") and hasattr(exchange, "sapi_get_account_apirestrictions"):
            return permissions_from_binance(exchange.sapi_get_account_apirestrictions())
        if eid in ("coinbase", "coinbaseadvanced") and hasattr(exchange, "v3PrivateGetBrokerageKeyPermissions"):
            return permissions_from_coinbase(exchange.v3PrivateGetBrokerageKeyPermissions())
    except Exception as e:  # network / auth errors: be pessimistic
        log.warning("key permission query failed", extra={"venue": eid, "error": str(e)})
        return KeyPermissions(known=False, source=f"{eid}:error")
    return KeyPermissions(known=False, source=f"{eid}:unsupported")


def enforce_key_safety(perms: KeyPermissions, mode: Mode, attested: dict | None = None) -> list[str]:
    """Raise KeySafetyError if the key is unsafe for ``mode``; return warnings otherwise.

    ``attested`` is an operator attestation (``{"no_withdrawal": True, "ip_whitelist": True}``)
    used only for venues that don't report permissions.
    """
    warnings: list[str] = []
    if perms.known:
        if perms.can_withdraw:
            raise KeySafetyError(
                "API key has WITHDRAWAL/transfer permission enabled. Create a trade-only key. Refusing to run."
            )
        if not perms.ip_restricted:
            msg = "API key has no IP whitelist; restrict it to your server's IP"
            if mode is Mode.LIVE:
                raise KeySafetyError(msg + " (required for LIVE)")
            warnings.append(msg)
    else:
        att = attested or {}
        if mode is Mode.LIVE and not (att.get("no_withdrawal") and att.get("ip_whitelist")):
            raise KeySafetyError(
                f"cannot verify key permissions ({perms.source}); LIVE requires an operator attestation "
                "via `sentinel keycheck --attest` after checking the exchange UI"
            )
        warnings.append(f"key permissions not verifiable via API ({perms.source}); relying on attestation")
    for w in warnings:
        log.warning(w)
    return warnings
