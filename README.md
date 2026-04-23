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

## Requirements

- **Python 3.10 or newer** on every supported platform.
  - **macOS**: Python 3 is *not* preinstalled. Install via
    [Homebrew](https://brew.sh) with `brew install python`, or grab the
    `.pkg` from [python.org/downloads/macos/](https://www.python.org/downloads/macos/).
  - **Linux**: install `python3` and `python3-venv` through your package
    manager (e.g. `sudo apt install python3 python3-venv`).
  - **Windows**: install from [python.org/downloads/windows/](https://www.python.org/downloads/windows/)
    and tick *"Add python.exe to PATH"* on the first installer screen.
- Git (to clone the repo).
- ~200 MB of disk for the virtualenv and cached market data.

Check your version:

```bash
python3 --version   # macOS / Linux
py --version        # Windows (py launcher)
```

Most macOS and Linux systems expose the interpreter as **`python3`**, not
`python` — use whichever one responds to `--version`. Once the virtualenv
is activated (step 2 below) plain `python` works inside it on all
platforms.

## Quick start

> **Do these from inside the cloned repo**, not your home directory. If
> you haven't cloned yet:
> ```bash
> git clone <repo-url>
> cd trading-bot
> ```

Then pick the block that matches your platform.

### macOS / Linux (bash or zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
cp .env.example .env
# edit config.yaml and .env in your editor of choice

python backtest.py --days 180        # 1. Backtest
python run.py                        # 2. Paper trade
# 3. Edit config.yaml -> trading.mode: live, then re-run
python run.py
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item config.example.yaml config.yaml
Copy-Item .env.example .env
# edit config.yaml and .env (e.g. `notepad config.yaml`)

python backtest.py --days 180
python run.py
# edit config.yaml -> trading.mode: live, then re-run
python run.py
```

> If PowerShell blocks the activation script with a *"running scripts is
> disabled"* error, run this once (as your own user, not admin):
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### Windows (Command Prompt / cmd.exe)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

copy config.example.yaml config.yaml
copy .env.example .env
REM edit config.yaml and .env (e.g. `notepad config.yaml`)

python backtest.py --days 180
python run.py
REM edit config.yaml -> trading.mode: live, then re-run
python run.py
```

## Command reference

| Task                       | Command                                           |
| -------------------------- | ------------------------------------------------- |
| Backtest (default 180 days)| `python backtest.py`                              |
| Backtest, custom window    | `python backtest.py --days 365`                   |
| Walk-forward (4 windows)   | `python backtest.py --days 365 --walk-forward 4`  |
| Paper / live trading       | `python run.py`                                   |
| Live without confirm prompt| `python run.py --yes`                             |
| Streamlit dashboard        | `streamlit run dashboard.py`                      |
| Parameter sweep            | `python scripts/sweep.py`                         |
| Compare strategies         | `python scripts/compare_strategies.py`            |
| Run tests                  | `python -m pytest`                                |
| Run one test file          | `python -m pytest tests/test_risk.py`             |

The quote style in arguments is the same across platforms (`--days 180`, no
escaping needed). File paths use forward slashes in config files; Python
normalises them to backslashes on Windows automatically.

### Leaving / re-entering the virtualenv

| Action  | macOS / Linux             | PowerShell                      | CMD                            |
| ------- | ------------------------- | ------------------------------- | ------------------------------ |
| Enter   | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1`| `.venv\Scripts\activate.bat`   |
| Leave   | `deactivate`              | `deactivate`                    | `deactivate`                   |

## Safety defaults

- `trading.mode` defaults to `paper`. Live trading requires an explicit
  config change **and** the presence of API credentials in `.env`.
- Live mode prompts for confirmation on startup.
- A daily loss limit and a global max drawdown automatically halt trading.

## Project layout

```
bot/
  engine.py          # main event loop (paper + live)
  backtest.py        # historical replay + walk-forward
  exchange.py        # ccxt wrapper with retry logic
  execution.py       # shared fee/slippage/fill helpers
  portfolio.py       # cash, positions, MFE/MAE tracking
  risk.py            # sizing, stops, kill switches, circuit breaker
  indicators.py      # technical indicators (pandas/numpy)
  logger.py          # rich console + CSV loggers
  notifications.py   # optional Telegram alerts
  news.py            # crypto news feed + classifier
  news_stocks.py     # earnings calendar + gap detector
  sentiment.py       # fear & greed index
  stocks.py          # Trading 212 + yfinance adapter
  universe.py        # S&P 500 / Nasdaq-100 expansion
  strategies/
    base.py
    ma_crossover.py
    rsi_reversion.py
    macd.py
    bollinger.py
    ensemble.py
    filtered.py      # trend / ADX / ATR / volume gates
run.py               # entry point for paper/live
backtest.py          # entry point for backtests
dashboard.py         # Streamlit monitoring UI
scripts/
  sweep.py           # parameter sweep
  compare_strategies.py
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

## Platform notes

- **Line endings**: the repo uses LF. On Windows, set
  `git config --global core.autocrlf input` to avoid CRLF being committed
  back. Python reads both fine regardless.
- **Long paths on Windows**: the bot writes to `logs/`, `reports/`, and
  `cache/` under the project directory. If you hit a *"path too long"*
  error, enable long paths:
  `git config --global core.longpaths true` and reboot after enabling
  `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled`.
- **Stopping the bot**: `Ctrl+C` works on all three platforms and triggers
  a graceful shutdown with a final notification.
- **Running as a background service**: use `launchd` (macOS), `systemd`
  (Linux) or Task Scheduler (Windows). The bot has no daemonisation
  built in — run it from a supervisor of your choice.
- **Telegram alerts**: the `requests`-based notifier works on all
  platforms. No firewall exemptions needed — it only makes outbound HTTPS
  calls.

## Disclaimer

No strategy wins forever. Markets change. The defaults here are
illustrative, not recommendations. You are responsible for your own
trades.
