# Deposits, withdrawals & profit (UK)

How money gets from your bank to the bot and back, and what Sentinel does and **doesn't** do.

```
UK bank ──Faster Payments──► Kraken / Coinbase GBP balance ──► Sentinel trades BTC/GBP, ETH/GBP
   ▲                                                                    │
   └───────── you withdraw GBP on the exchange (2FA) ◄──── Sentinel reserves the cash & tells you
```

## The rules

* **Your money never passes through Sentinel.** It stays in your own exchange account, in your name.
  There is no "payment gateway" to hack, and no balance Sentinel could lose.
* **Sentinel can't withdraw, by design.** It uses a *trade-only* API key and refuses to start with a key
  that has withdrawal permission. If that key ever leaked, an attacker could trade, but could not take your
  money. Every withdrawal is done by you, on the exchange, confirmed with 2FA.
* **Deposits aren't profit and withdrawals aren't losses.** Sentinel tracks invested capital separately.
  Your risk limits (daily loss, drawdown) measure trading only, so taking £500 out never triggers a halt.
* **Profit is not guaranteed.** The bot only trades strategies that pass the backtest → paper → readiness-gate
  path, and even then it can lose money. See the [risk disclosure](RISK_DISCLOSURE.md).

## One-time setup

1. **Open a UK-supported exchange account** in your own name and complete verification (KYC). Kraken and
   Coinbase both support GBP deposits and withdrawals to UK bank accounts by Faster Payments. Check each
   venue's current fees and limits; they change.
2. **Use the UK profile:**
   ```bash
   cp config/uk-kraken.example.yaml config/sentinel.yaml   # or set venue: coinbase
   ```
   Set `funds.bank_label` to something you'll recognise, e.g. `"Monzo ****1234"`. It's only a label; never
   put account numbers in the config (Sentinel refuses secrets in config anyway).
3. **Create the API key** on the exchange with **only** these permissions:
   * *Query funds* and *Query ledger entries*: lets Sentinel read balances and your deposit/withdrawal history
   * *Create & modify orders*, *Cancel/close orders*: lets it trade
   * **NOT** *Withdraw funds*, and no transfer permissions
   * Restrict it to your server's IP address if the exchange allows it.
4. Store it in the encrypted vault and check it:
   ```bash
   sentinel vault set kraken.api_key
   sentinel vault set kraken.api_secret
   sentinel keycheck --attest     # Kraken doesn't report permissions via API: you confirm them
   ```

## Depositing

```bash
sentinel funds deposit --how      # shows the steps
```

Send GBP from your bank to the exchange's deposit details (include any reference it gives you). When it
arrives, Sentinel's read-only sync (every 15 minutes) records it, or record it yourself:

```bash
sentinel funds deposit 1000 --note "from Monzo"
sentinel funds status
```

In **PAPER** mode `funds deposit` simulates the money arriving, so you can rehearse everything first.

## Withdrawing your profit

```bash
sentinel funds status                       # profit, and how much is withdrawable right now
sentinel funds withdraw 250 --profit-only   # refuses if it would touch your invested capital
```

What happens next:

1. Sentinel **reserves** £250: the bot will not open new trades with it.
2. If that cash is tied up in an open position, the request waits as `requested` and becomes `ready` when the
   position closes (or close it yourself, or use `sentinel kill`).
3. **You** open the exchange: Funding → Withdraw → GBP → your bank → £250 → approve with 2FA.
4. Sentinel spots the withdrawal in the exchange history and marks it `completed`, or run
   `sentinel funds confirm <id>`. Changed your mind? `sentinel funds cancel <id>`.

The dashboard's **Funds** panel does the same (record deposit, request withdrawal, confirm, cancel). It also
suggests taking profit once profit exceeds `funds.profit_take_pct` (default 10%) of what you invested.

## Reading the numbers

| Figure | Meaning |
|---|---|
| Net invested | opening balance + deposits − withdrawals |
| Profit / loss | account value − net invested |
| Return on invested | profit ÷ net invested |
| Time-weighted return | the strategy's performance with your deposits/withdrawals removed |
| Withdrawable now | free cash (not in positions), minus reservations and `keep_minimum_balance` |

## Tax (UK)

Crypto disposals are generally subject to Capital Gains Tax in the UK, and HMRC expects records of every
acquisition and disposal. Export everything once a year:

```bash
sentinel funds export --year 2026
```

That gives a CSV of deposits, withdrawals and every closed trade. It's a record, **not tax advice**. HMRC's
share-pooling rules need proper calculation, so use a crypto tax tool or an accountant.

## Limits of this design

* Only cash in `funds.currency` (GBP) is tracked as deposits/withdrawals. Crypto sent in or out of the account
  shows up as unexplained balance changes and will trip reconciliation, so keep a dedicated account (or
  sub-account) for the bot.
* The exchange's withdrawal fee is matched approximately (within 2%) when pairing a request with what actually
  left the account.
* Taking money from other people (friends, clients) is a regulated activity in the UK (FCA). This system is for
  your own money only.
