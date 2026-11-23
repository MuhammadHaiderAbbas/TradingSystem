# Funds: deposits, withdrawals, profit

Money never moves through Sentinel. Funds stay at the exchange or broker, and withdrawals are executed by you on the venue with 2FA.
This module records cash flows (manually, or read-only from the venue's funding history), applies them to the
accounting (in PAPER it also moves simulated cash), and shifts the risk manager's reference equities so deposits and
withdrawals are never treated as profit or loss. It also reserves cash for withdrawal requests (the Trader excludes reserved cash),
separates invested capital from profit (including time-weighted return), suggests profit-taking, and exports a CSV.

A test guarantees that no code path calls a venue withdrawal or transfer API.

Run: `sentinel funds status`. Guide: `docs/FUNDING_UK.md`. Verify: `pytest tests/test_funds.py`.
