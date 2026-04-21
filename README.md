# Advanced Trading Bot

A configurable, multi-strategy crypto trading bot with backtesting, paper
trading, and live trading support. Built on [ccxt](https://github.com/ccxt/ccxt)
so it works against 100+ exchanges.

> **Risk warning**: Trading is high risk. You can lose all your capital. This
> software is provided as-is with no warranty. Start with paper trading and
> only risk what you can afford to lose.

## Features

- **Multi-strategy engine**: MA crossover, RSI mean reversion, MACD,
  Bollinger bands, and an ensemble voter that combines them.
- **Risk management first**: volatility-aware position sizing, per-trade
  stop-loss and take-profit, trailing stops, daily loss limit, and a global
  max-drawdown kill switch.
- **Three run modes**:
  - `backtest` – replay historical candles, produce performance stats.
  - `paper` – live prices, simulated fills, no real money.
  - `live` – real orders via the exchange API.
- **Exchange agnostic** via ccxt (Kraken, Binance, Coinbase, Bybit, …).
- **Structured logging** of every trade and equity snapshot to CSV.
- **Configuration by YAML** – no code changes needed to retune.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
cp .env.example .env
# edit config.yaml and .env

# 1. Backtest the strategy on recent history
python backtest.py --days 180

# 2. Run in paper mode (real prices, fake money)
python run.py

# 3. Go live (only after you've tested the above!)
#    Edit config.yaml: trading.mode = live, then:
python run.py
```

## Safety defaults

- `trading.mode` defaults to `paper`. Live trading requires an explicit
  config change **and** the presence of API credentials in `.env`.
- Live mode prompts for confirmation on startup.
- A daily loss limit and a global max drawdown automatically halt trading.

## Project layout

```
bot/
  engine.py          # main event loop (paper + live)
  backtest.py        # historical replay engine
  exchange.py        # ccxt wrapper with retry logic
  portfolio.py       # cash, positions, equity tracking
  risk.py            # sizing, stops, kill switches
  indicators.py      # technical indicators (pandas/numpy)
  data.py            # OHLCV fetch + caching
  logger.py          # rich console + CSV loggers
  notifications.py   # optional Telegram alerts
  strategies/
    base.py
    ma_crossover.py
    rsi_reversion.py
    macd.py
    bollinger.py
    ensemble.py
run.py               # entry point for paper/live
backtest.py          # entry point for backtests
tests/               # pytest suite
```

## Strategies

Each strategy consumes an OHLCV DataFrame and returns `+1` (long), `-1`
(short/exit), or `0` (no signal). The ensemble requires `min_agreement`
votes in the same direction before acting.

Add your own by subclassing `bot/strategies/base.py::Strategy` and
registering the name in `bot/strategies/__init__.py`.

## Going live – the sensible path

1. Backtest over at least 6 months. Check the stats (Sharpe, max drawdown,
   win rate). If drawdown is greater than you can stomach, re-tune.
2. Paper trade for at least a week. Verify the bot's behaviour matches
   expectations on live prices.
3. Create an exchange API key with **trade** permissions but **no
   withdrawal** permissions. Set IP allowlisting if your exchange
   supports it.
4. Put your credentials in `.env`, set `trading.mode: live`, start small.

## Disclaimer

No strategy wins forever. Markets change. The defaults here are
illustrative, not recommendations. You are responsible for your own
trades.
