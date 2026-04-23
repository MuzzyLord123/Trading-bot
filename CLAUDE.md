# Trading 212 Stock Bot — Agent Guide

Read this first. It summarises what the project is, what rules the owner has
set, and how to make changes without breaking them.

## What this is

A Trading 212 Invest stock/ETF trading bot. Yahoo Finance (`yfinance`) for
historical candles and signals; Trading 212 REST API for live execution.
Three run modes: backtest, paper, live. Streamlit dashboard for monitoring.

**Platform target: macOS.** The owner is developing on a MacBook Pro with
zsh.

## Hard rules (do not violate)

1. **No emojis anywhere.** Not in code, not in UI, not in commit messages,
   not in documentation. Plain ASCII only. If you find one, remove it.
2. **Trading 212 only.** No crypto, no Kraken, no ccxt, no Binance, no
   Coinbase, no Fear & Greed index, no CoinDesk/Cointelegraph feeds. If
   you see any of those words outside a deliberate "legacy keys we ignore"
   shim, treat it as a bug and delete it.
3. **Every feature must actually work.** Do not mark a task complete based
   on imports or syntax alone — exercise the real call path end to end.
   There is a 95-case pytest suite; keep it green.
4. **No parameter tuning disguised as optimisation.** Fitting the strategy
   to past data is overfitting and will not make the live bot profitable.
   Prefer infrastructure that lets the owner evaluate changes honestly
   (walk-forward, per-window stats, cost-aware r:r) over "better numbers
   on this one backtest".
5. **Opt-in for new risk behaviours.** Scale-out, ATR stops, consecutive-
   loss circuit breaker are all off by default. Keep them that way so
   existing configs behave identically after an upgrade.

## Development loop

The owner uses zsh on macOS. All commands assume you are inside the repo
directory with the venv activated:

```bash
cd ~/trading-bot
source .venv/bin/activate
```

Common tasks:

```bash
python -m pytest                                  # run the suite (should be 95+ passing)
python backtest.py --days 365                     # backtest
python backtest.py --days 365 --walk-forward 4    # walk-forward OOS report
python run.py                                     # paper/live depending on config
python launch_dashboard.py                        # Streamlit on :8501, opens Safari on mac
python scripts/sweep.py                           # parameter sweep on synthetic data
```

## Git workflow

- The owner's current feature branch is `claude/analyze-trading-bot-dY50P`.
- Push to that branch unless told otherwise.
- After any non-trivial change: run the full test suite, then commit with
  a descriptive message explaining **why**, not just what. Then `git push`.
- Do **not** open pull requests unless asked.

## Architecture at a glance

```
run.py / backtest.py / launch_dashboard.py      # entry points
dashboard.py                                     # Streamlit UI
bot/
  engine.py        event loop for paper + live; handles tick, open, scale-out, close
  backtest.py      historical replay + walk-forward + BacktestResult.stats
  risk.py          sizing, stops, circuit breaker, scale-out trigger
  portfolio.py     cash, positions, MFE/MAE tracking
  execution.py     shared buy_fill / sell_fill helpers used by engine + backtest
  stocks.py        YFinanceSource + Trading212Broker -> StocksExchange
  news_feed.py     yfinance-backed per-ticker news for the dashboard
  news_stocks.py   earnings calendar + bearish-gap detector
  universe.py      SP500 / NASDAQ100 token expansion
  indicators.py    atr, adx, rsi, ema, macd, bollinger, supertrend
  strategies/
    ma_crossover.py / rsi_reversion.py / macd.py / bollinger.py
    ensemble.py    weighted voter
    filtered.py    regime gates: trend / ADX / ATR / volume / earnings / news-gap
  factory.py       trivial builder for StocksExchange (single-broker now)
  config.py        YAML loader with validate() — legacy ccxt keys are silently
                   dropped so old user configs still load
  notifications.py optional Telegram sender
  logger.py        rich console + append-only CsvLogger
```

## Configuration

- `config.example.yaml` is the canonical starting point. It is copied to
  `config.yaml` (gitignored) by the user.
- `.env.example` -> `.env` for secrets. Only these are read:
  `TRADING212_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Legacy ccxt-style keys (`EXCHANGE_API_KEY`, `requires_password`,
  `enable_rate_limit`) are ignored — don't re-introduce them.

## Things that frequently trip people up

- **Streamlit does not auto-open a browser** because `.streamlit/config.toml`
  has `server.headless = true`. Use `python launch_dashboard.py`; it
  opens Safari on macOS via `open -a Safari`, default browser elsewhere.
- **macOS has no `python`** command by default, only `python3`. After
  `source .venv/bin/activate` both names resolve inside the venv.
- **The user keeps their local `config.yaml` forever.** Don't assume it
  matches `config.example.yaml`. For branding/display strings, hard-code
  rather than reading from the user's config.
- **Universe expansion can fail offline.** `SP500` and `NASDAQ100` tokens
  hit Wikipedia; if that 403s, the code falls back to a cached copy. This
  is normal, not a bug.
- **Market-hours gate:** the engine skips ticks outside 07:00-21:00 UTC
  (covers US session + LSE). Don't remove this; it exists so paper/live
  runs don't accumulate bad data overnight.

## Testing conventions

- pytest, single `tests/` directory.
- Every new feature should land with unit tests. Current count: 95.
- Integration tests that need yfinance mock it via
  `patch.dict(sys.modules, {"yfinance": fake})`.
- Do not add tests that hit the network.
