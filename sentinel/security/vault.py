"""Encrypted secrets vault.

* Key derivation: argon2id (memory-hard) from a master password + random salt.
* Encryption: Fernet (AES-128-CBC + HMAC-SHA256) over a JSON map of secrets.
* The vault file is written atomically with mode 0600.
* The master password comes from (in order): an explicit argument, the OS keyring
  (if the ``keyring`` package and a backend are available), the
  ``SENTINEL_MASTER_PASSWORD`` environment variable, or an interactive prompt.
* Every decrypted value is registered with the log redactor, so it can never
  appear in logs, and is wrapped in :class:`Secret`, whose repr is masked.
"""

from __future__ import annotations

import base64
import getpass
import json
import os
import tempfile
from pathlib import Path

from argon2.low_level import Type, hash_secret_raw
from cryptography.fernet import Fernet, InvalidToken

from sentinel.core.errors import VaultError
from sentinel.monitoring.logging import REDACTOR

KEYRING_SERVICE = "sentinel-trader"
KEYRING_USER = "vault-master"
VAULT_VERSION = 1
# argon2id parameters (RFC 9106 "second recommended" profile, 64 MiB)
ARGON_TIME = 3
ARGON_MEMORY_KIB = 64 * 1024
ARGON_PARALLELISM = 4


class Secret:
    """A string wrapper that never reveals itself in repr/str/format."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value
        REDACTOR.register(value)

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return "***"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)


def derive_key(password: str, salt: bytes, time_cost: int = ARGON_TIME, memory_kib: int = ARGON_MEMORY_KIB,
               parallelism: int = ARGON_PARALLELISM) -> bytes:
    raw = hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=salt,
        time_cost=time_cost,
        memory_cost=memory_kib,
        parallelism=parallelism,
        hash_len=32,
        type=Type.ID,
    )
    return base64.urlsafe_b64encode(raw)


def _keyring_get() -> str | None:
    try:
        import keyring  # type: ignore
        from keyring.backends import fail  # type: ignore

        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:
        return None


def keyring_store(password: str) -> bool:
    """Store the master password in the OS keyring. Returns False if unavailable."""
    try:
        import keyring  # type: ignore
        from keyring.backends import fail  # type: ignore

        if isinstance(keyring.get_keyring(), fail.Keyring):
            return False
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, password)
        return True
    except Exception:
        return False


def resolve_master_password(explicit: str | None = None, interactive: bool = True) -> str:
    if explicit:
        return explicit
    pw = _keyring_get() or os.environ.get("SENTINEL_MASTER_PASSWORD")
    if pw:
        return pw
    if interactive:
        return getpass.getpass("Sentinel vault master password: ")
    raise VaultError("no master password available (keyring, SENTINEL_MASTER_PASSWORD, or prompt)")


class Vault:
    """File-backed encrypted key/value store for secrets."""

    def __init__(self, path: str | Path, password: str, *, kdf_params: dict | None = None) -> None:
        self.path = Path(path)
        self._password = password
        self._kdf = kdf_params or {"t": ARGON_TIME, "m": ARGON_MEMORY_KIB, "p": ARGON_PARALLELISM}
        self._secrets: dict[str, str] = {}
        self._salt: bytes = b""
        if self.path.exists():
            self._load()
        if len(password) < 8:
            raise VaultError("master password must be at least 8 characters")

    # ---- persistence ---------------------------------------------------------------
    @classmethod
    def create(cls, path: str | Path, password: str, *, kdf_params: dict | None = None) -> Vault:
        p = Path(path)
        if p.exists():
            raise VaultError(f"vault already exists at {p}")
        if len(password) < 8:
            raise VaultError("master password must be at least 8 characters")
        v = cls.__new__(cls)
        v.path = p
        v._password = password
        v._kdf = kdf_params or {"t": ARGON_TIME, "m": ARGON_MEMORY_KIB, "p": ARGON_PARALLELISM}
        v._secrets = {}
        v._salt = os.urandom(16)
        v._save()
        return v

    def _fernet(self) -> Fernet:
        return Fernet(derive_key(self._password, self._salt, self._kdf["t"], self._kdf["m"], self._kdf["p"]))

    def _load(self) -> None:
        try:
            doc = json.loads(self.path.read_text())
            self._salt = base64.b64decode(doc["salt"])
            self._kdf = doc["kdf"]
            token = doc["token"].encode()
        except (OSError, KeyError, ValueError) as e:
            raise VaultError(f"vault file is corrupt: {self.path}") from e
        try:
            plain = self._fernet().decrypt(token)
        except InvalidToken as e:
            raise VaultError("wrong master password or tampered vault") from e
        self._secrets = json.loads(plain)
        for v in self._secrets.values():
            REDACTOR.register(v)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        token = self._fernet().encrypt(json.dumps(self._secrets).encode())
        doc = {
            "version": VAULT_VERSION,
            "kdf_alg": "argon2id",
            "kdf": dict(self._kdf),
            "cipher": "fernet",
            "salt": base64.b64encode(self._salt).decode(),
            "token": token.decode(),
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".vault-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ---- API -------------------------------------------------------------------------
    def set(self, name: str, value: str) -> None:
        if not name or not value:
            raise VaultError("secret name and value must be non-empty")
        REDACTOR.register(value)
        self._secrets[name] = value
        self._save()

    def get(self, name: str) -> Secret:
        if name not in self._secrets:
            raise VaultError(f"secret '{name}' not found in vault")
        return Secret(self._secrets[name])

    def get_optional(self, name: str) -> Secret | None:
        return Secret(self._secrets[name]) if name in self._secrets else None

    def delete(self, name: str) -> None:
        self._secrets.pop(name, None)
        self._save()

    def names(self) -> list[str]:
        return sorted(self._secrets)

    def change_password(self, new_password: str) -> None:
        if len(new_password) < 8:
            raise VaultError("master password must be at least 8 characters")
        self._password = new_password
        self._salt = os.urandom(16)
        self._save()

    def __repr__(self) -> str:
        return f"Vault(path={self.path!s}, secrets={len(self._secrets)})"


def exchange_credentials(vault: Vault, venue: str) -> dict[str, str]:
    """Return ccxt-style credential dict for a venue (values revealed only here)."""
    creds: dict[str, str] = {}
    for field, ccxt_name in (("api_key", "apiKey"), ("api_secret", "secret"), ("passphrase", "password")):
        s = vault.get_optional(f"{venue}.{field}")
        if s is not None:
            creds[ccxt_name] = s.reveal()
    return creds
