"""Exception hierarchy."""


class SentinelError(Exception):
    """Base error."""


class ConfigError(SentinelError):
    """Invalid configuration: Sentinel refuses to start."""


class VaultError(SentinelError):
    """Vault could not be opened or a secret is missing."""


class KeySafetyError(SentinelError):
    """API key permissions are unsafe (e.g. withdrawals enabled)."""


class ModeLockedError(SentinelError):
    """The requested mode is locked (LIVE before the readiness gate passes)."""


class RiskVeto(SentinelError):
    """The risk manager rejected an order."""


class KillSwitchEngaged(SentinelError):
    """The bot is locked by the kill switch."""


class TradingHalted(SentinelError):
    """Trading is halted until a manual re-enable."""


class BrokerError(SentinelError):
    """Generic broker error."""


class TransientBrokerError(BrokerError):
    """Retryable broker error (network, rate limit, 5xx)."""


class ReconciliationError(SentinelError):
    """Internal state disagrees with the exchange."""


class DataQualityError(SentinelError):
    """Market data failed a quality check that cannot be repaired."""
