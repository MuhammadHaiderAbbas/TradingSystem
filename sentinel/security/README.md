# Security

* `vault.py`: argon2id-derived Fernet encryption, atomic 0600 writes, OS keyring support, a `Secret` type that never prints its value.
* `key_safety.py`: queries key permissions (Binance, Coinbase; attestation for venues that don't expose them). It refuses any key with withdrawal or transfer permission, and LIVE also requires an IP whitelist.

Verify: `pytest tests/test_security.py`; `sentinel keycheck`.
