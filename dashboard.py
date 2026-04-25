"""Local web dashboard for the trading bot.

Run with:
    streamlit run dashboard.py

Opens http://localhost:8501 in your browser. The bot itself runs
separately (``python main.py`` for live/paper). This dashboard is a
read/write UI over the config file and the CSV logs the bot writes.

Pages (sidebar):
  * Overview  - current equity, today's P&L, open positions, equity curve.
  * Trades    - full trade history with per-symbol breakdown.
  * Config    - edit config + "Save & Run Backtest" button.
  * News      - live company-news headlines from Yahoo Finance.
  * Backtest  - latest backtest results.
  * Settings  - API keys, stored in .env.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
CONFIG_EXAMPLE_PATH = ROOT / "config.example.yaml"
ENV_PATH = ROOT / ".env"
TRADES_PATH = ROOT / "logs" / "trades.csv"
EQUITY_PATH = ROOT / "logs" / "equity.csv"
BACKTEST_EQUITY = ROOT / "reports" / "backtest_equity.csv"
BACKTEST_TRADES = ROOT / "reports" / "backtest_trades.csv"
BACKTEST_STATS = ROOT / "reports" / "backtest_stats.json"
BACKTEST_LOG = ROOT / "reports" / "last_backtest.log"
STATE_PATH = ROOT / "state" / "portfolio.json"

API_KEY_FIELDS = [
    ("TRADING212_API_KEY", "Trading 212 API key"),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token"),
    ("TELEGRAM_CHAT_ID", "Telegram chat ID"),
]


# ---------------------------------------------------------------------------
# Page setup + custom CSS.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Bot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      /* Cards */
      div[data-testid="stMetric"] {
          background: #161B22;
          border: 1px solid #2d333b;
          border-radius: 8px;
          padding: 14px 16px;
      }
      div[data-testid="stMetricValue"] { font-weight: 600; }
      div[data-testid="stMetricDelta"] { font-size: 0.85rem; }
      /* Tabs */
      button[data-baseweb="tab"] {
          padding: 0.5rem 1.1rem;
          font-weight: 500;
      }
      /* Headings */
      h1 { font-weight: 700; letter-spacing: -0.02em; }
      h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
      /* Subtle divider */
      hr { border-color: #2d333b; margin: 1.2rem 0; }
      /* Dataframes */
      div[data-testid="stDataFrame"] { border-radius: 8px; }
      /* Sidebar */
      section[data-testid="stSidebar"] { background: #0a0c10; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV written by the bot, tolerant of mid-life schema changes.

    If an older run wrote fewer columns than a newer one (e.g. before
    bars_held / mfe / mae were added), pandas' strict parser blows up with
    "Expected N fields, saw M". Fall back to a lenient parse that skips
    malformed lines so the dashboard never crashes on one bad log file -
    CsvLogger now rotates stale files on header change, but users with
    pre-existing logs still get a graceful view.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, on_bad_lines="skip", engine="python")
        except Exception:
            return pd.DataFrame()


def format_currency(value: float, ccy: str = "GBP") -> str:
    sign = "£" if ccy == "GBP" else ""
    return f"{sign}{value:,.2f}"


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def write_env(updates: dict[str, str]) -> None:
    existing: list[str] = []
    if ENV_PATH.exists():
        existing = ENV_PATH.read_text().splitlines()
    seen_keys: set[str] = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen_keys.add(key)
                continue
        out.append(line)
    for key, val in updates.items():
        if key not in seen_keys:
            out.append(f"{key}={val}")
    ENV_PATH.write_text("\n".join(out) + "\n")


def mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return value[:4] + "•" * (len(value) - 8) + value[-4:]


# ---------------------------------------------------------------------------
# Data loading (cached to keep the UI snappy).
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10)
def load_trades() -> pd.DataFrame:
    df = read_csv(TRADES_PATH)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


@st.cache_data(ttl=10)
def load_equity() -> pd.DataFrame:
    df = read_csv(EQUITY_PATH)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def _numeric_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "pnl" not in trades.columns:
        return pd.DataFrame(columns=trades.columns)
    df = trades.copy()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    return df.dropna(subset=["pnl"])


def pnl_stats(trades: pd.DataFrame) -> dict[str, float]:
    closed = _numeric_pnl(trades)
    if closed.empty:
        return {"count": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "best": 0.0, "worst": 0.0}
    wins = closed[closed["pnl"] > 0]
    return {
        "count": int(len(closed)),
        "win_rate": 100.0 * len(wins) / len(closed),
        "total_pnl": float(closed["pnl"].sum()),
        "avg_pnl": float(closed["pnl"].mean()),
        "best": float(closed["pnl"].max()),
        "worst": float(closed["pnl"].min()),
    }


@st.cache_data(ttl=30)
def load_state() -> dict[str, Any]:
    """Return the engine's persisted portfolio state, or empty dict if
    nothing saved yet. Cached so a 1-second auto-refresh doesn't re-parse
    the file on every rerun."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=60)
def fetch_live_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    """Batch quote lookup via yfinance. Cached for a minute so
    auto-refresh tabs don't hammer Yahoo."""
    if not symbols:
        return {}
    try:
        import yfinance as yf
    except ImportError:
        return {}
    try:
        quotes = yf.Tickers(" ".join(symbols))
    except Exception:
        return {}
    out: dict[str, float] = {}
    for sym in symbols:
        try:
            info = quotes.tickers[sym].fast_info
            last = getattr(info, "last_price", None) or info.get("last_price") if hasattr(info, "get") else None
            if last is not None and not pd.isna(last):
                out[sym] = float(last)
        except Exception:
            continue
    return out


def per_symbol_stats(trades: pd.DataFrame) -> pd.DataFrame:
    closed = _numeric_pnl(trades)
    if closed.empty:
        return pd.DataFrame()
    grp = closed.groupby("symbol")["pnl"].agg(
        trades="count",
        total="sum",
        avg="mean",
        wins=lambda s: int((s > 0).sum()),
        losses=lambda s: int((s <= 0).sum()),
    ).reset_index()
    grp["win_rate"] = (grp["wins"] / grp["trades"] * 100).round(1)
    return grp.sort_values("total", ascending=False)


# ---------------------------------------------------------------------------
# Sidebar: bot status, quick actions, navigation.
# ---------------------------------------------------------------------------
cfg = load_yaml(CONFIG_PATH)
ccy = cfg.get("trading", {}).get("quote_currency", "GBP")
starting_capital = float(cfg.get("trading", {}).get("starting_capital", 0.0))
mode = cfg.get("trading", {}).get("mode", "paper").upper()
exchange_name = "Trading 212"

with st.sidebar:
    st.markdown("### Trading Bot")

    # Live status pill - one glance and you know if the bot is running.
    equity_df = load_equity()
    if not equity_df.empty:
        last_seen = equity_df["timestamp"].iloc[-1]
        age = datetime.now(timezone.utc) - last_seen.to_pydatetime()
        if age < timedelta(minutes=5):
            st.success(
                f"Active · last tick {int(age.total_seconds())}s ago",
                icon=None,
            )
        elif age < timedelta(hours=1):
            st.warning(f"Stale · last tick {int(age.total_seconds() / 60)} min ago")
        else:
            st.error(f"Offline · last tick {age.days}d {age.seconds // 3600}h ago")
    else:
        st.info("Bot has not run yet. Start it with `python run.py`.")

    st.markdown("---")
    st.markdown("**At a glance**")
    symbols = cfg.get("trading", {}).get("symbols", [])
    st.caption(f"Mode: **{mode}** · Exchange: **{exchange_name}**")
    st.caption(f"Universe: **{len(symbols)}** symbols")
    st.caption(f"Starting capital: **{format_currency(starting_capital, ccy)}**")

    st.markdown("---")
    st.markdown("**Quick actions**")
    qa_cols = st.columns(2)
    if qa_cols[0].button("Refresh", use_container_width=True,
                         help="Reload all CSV / state caches and rerun the page."):
        st.cache_data.clear()
        st.rerun()
    if qa_cols[1].button("Reset caches", use_container_width=True,
                         help="Delete on-disk yfinance and news caches; the next "
                              "backtest or news refresh will re-fetch from source."):
        from pathlib import Path as _P
        for sub in ("ohlcv", "news"):
            d = ROOT / "cache" / sub
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        f.unlink(missing_ok=True)
        st.cache_data.clear()
        st.toast("Caches cleared.")
        st.rerun()

    auto_refresh = st.toggle(
        "Auto-refresh", value=False,
        help="Re-run the page on an interval so the Overview tab stays live.",
    )
    refresh_interval = st.slider(
        "Interval (seconds)", min_value=5, max_value=120, value=15, step=5,
        disabled=not auto_refresh,
    )

    st.markdown("---")
    st.markdown("**Tools**")
    st.caption(
        "Run from the terminal:\n"
        "- `python scripts/check_t212.py` - verify your API key\n"
        "- `python scripts/recommend_config.py` - compare strategies\n"
        "- `python backtest.py --walk-forward 4` - out-of-sample report\n"
        "- `python run.py` - start paper / live trading"
    )

    st.markdown("---")
    st.caption(
        "Config changes need a bot restart to take effect. "
        "Backtest runs use the saved config immediately."
    )


# ---------------------------------------------------------------------------
# Header: title + mode pill + first-run guidance.
# ---------------------------------------------------------------------------
mode_color = {"PAPER": "#2d8a4f", "LIVE": "#a02c2c"}.get(mode, "#3a3f47")
mode_label = "LIVE TRADING" if mode == "LIVE" else f"{mode} MODE"
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.5rem;">
      <h1 style="margin:0;">Trading Bot Dashboard</h1>
      <span style="background:{mode_color};color:white;padding:4px 10px;
                   border-radius:12px;font-size:0.78rem;font-weight:600;
                   letter-spacing:0.05em;">{mode_label}</span>
    </div>
    <div style="color:#8b949e;margin-bottom:1.2rem;">
      Trading 212 · {len(symbols)} symbols · {format_currency(starting_capital, ccy)} starting capital
    </div>
    """,
    unsafe_allow_html=True,
)

# First-run guidance: if no config or no equity log yet, show actionable
# steps in a banner so newcomers don't stare at empty tables and wonder
# what to do.
if not cfg:
    st.warning(
        "**No config.yaml found.** "
        "Copy `config.example.yaml` to `config.yaml`, edit it (start with "
        "`mode: paper`), then run `python run.py` to start the bot.",
        icon=None,
    )
elif equity_df.empty:
    st.info(
        "**Welcome.** Your config is loaded but the bot hasn't run yet. "
        "Open the Backtest tab to validate the strategy on history, or run "
        "`python run.py` from the terminal to start paper trading. "
        "This dashboard updates automatically once the bot starts writing logs.",
        icon=None,
    )

# Tabs in workflow order: see (Overview) -> what happened (Trades) ->
# why (News) -> tune (Backtest) -> change (Config) -> auth (Settings).
# Bound to named variables so re-ordering above doesn't break the body
# below; the body only references the names.
overview_tab, trades_tab, news_tab, backtest_tab, config_tab, settings_tab = st.tabs(
    ["Overview", "Trades", "News", "Backtest", "Config", "Settings"]
)


# -------------------------- Overview tab ---------------------------------
with overview_tab:
    trades = load_trades()
    equity = load_equity()
    stats = pnl_stats(trades)

    if equity.empty:
        current_equity = starting_capital
        day_pnl = 0.0
        week_pnl = 0.0
    else:
        current_equity = float(equity["equity"].iloc[-1])
        today_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        week_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_day = equity[equity["timestamp"] >= today_cutoff]
        recent_week = equity[equity["timestamp"] >= week_cutoff]
        day_pnl = float(recent_day["equity"].iloc[-1] - recent_day["equity"].iloc[0]) if len(recent_day) >= 2 else 0.0
        week_pnl = float(recent_week["equity"].iloc[-1] - recent_week["equity"].iloc[0]) if len(recent_week) >= 2 else 0.0

    total_pnl = current_equity - starting_capital if starting_capital else stats["total_pnl"]
    pnl_pct = (total_pnl / starting_capital * 100) if starting_capital else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity", format_currency(current_equity, ccy), f"{pnl_pct:+.2f}%")
    col2.metric("24h P&L", format_currency(day_pnl, ccy),
                f"{(day_pnl / current_equity * 100):+.2f}%" if current_equity else None)
    col3.metric("7d P&L", format_currency(week_pnl, ccy),
                f"{(week_pnl / current_equity * 100):+.2f}%" if current_equity else None)
    col4.metric("Total P&L", format_currency(total_pnl, ccy))
    col5.metric("Win rate", f"{stats['win_rate']:.1f}%", f"{stats['count']} trades")

    st.markdown("---")

    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.subheader(
            "Equity curve",
            help="Account equity over time. Each point is a tick where the "
                 "engine wrote a snapshot. Flat sections are usually market-"
                 "closed periods.",
        )
        if not equity.empty:
            chart_df = equity.set_index("timestamp")[["equity"]]
            st.line_chart(chart_df, height=340)
        else:
            st.info(
                "Equity curve appears once the bot starts ticking. "
                "Run `python run.py` from the terminal to start paper trading, "
                "or open the **Backtest** tab to validate the strategy on history first."
            )

    with col_r:
        st.subheader(
            "P&L stats",
            help="Realised P&L from closed trades only. Open positions are "
                 "shown below with live unrealised P&L.",
        )
        if stats["count"]:
            st.metric("Best trade", format_currency(stats["best"], ccy))
            st.metric("Worst trade", format_currency(stats["worst"], ccy))
            st.metric("Avg per trade", format_currency(stats["avg_pnl"], ccy))
        else:
            st.caption("No closed trades yet — see Trades tab once the bot has fills.")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(
            "Recent trades",
            help="Last 12 events from logs/trades.csv. See the Trades tab "
                 "for the full history with per-symbol breakdown.",
        )
        if trades.empty:
            st.caption("No trades yet — recent fills will appear here once the bot runs.")
        else:
            recent = trades.tail(12).iloc[::-1]
            st.dataframe(recent, use_container_width=True, hide_index=True, height=360)

    with col_b:
        st.subheader(
            "Open positions",
            help="Read live from state/portfolio.json (the engine's "
                 "authoritative snapshot). Prices are pulled fresh from "
                 "Yahoo Finance and cached for 60s.",
        )
        state = load_state()
        positions = state.get("positions") or []
        if not positions:
            st.caption("No open positions — entries will appear here as the engine fills them.")
        else:
            # Live-price every currently held symbol in one batched
            # yfinance call, then derive unrealised P&L locally.
            live_prices = fetch_live_prices(tuple(p["symbol"] for p in positions))
            rows = []
            total_unrealised = 0.0
            for p in positions:
                sym = p["symbol"]
                entry = float(p["entry_price"])
                amount = float(p["amount"])
                live = live_prices.get(sym)
                if live is None:
                    unrealised = 0.0
                    live_display = "-"
                    pct_display = "-"
                else:
                    unrealised = (live - entry) * amount
                    total_unrealised += unrealised
                    live_display = f"{live:,.4f}"
                    pct_display = f"{((live / entry) - 1) * 100:+.2f}%"
                rows.append({
                    "symbol": sym,
                    "qty": round(amount, 4),
                    "entry": round(entry, 4),
                    "last": live_display,
                    "stop": round(float(p.get("stop_loss", 0.0)), 4),
                    "tp": round(float(p.get("take_profit", 0.0)), 4),
                    "pnl": round(unrealised, 2),
                    "pnl%": pct_display,
                    "mfe": round(float(p.get("mfe", 0.0)), 4),
                    "mae": round(float(p.get("mae", 0.0)), 4),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, hide_index=True, height=360,
            )
            st.caption(
                f"Unrealised total: {format_currency(total_unrealised, ccy)} "
                f"across {len(positions)} position(s). State snapshot from "
                f"{state.get('saved_at', 'unknown')}"
            )


# -------------------------- Trades tab -----------------------------------
with trades_tab:
    trades = load_trades()
    if trades.empty:
        st.info(
            "No trades logged yet. The trade log populates as the engine "
            "opens and closes positions. To get started:\n\n"
            "- Open the **Backtest** tab to dry-run the strategy on history.\n"
            "- Run `python run.py` to start paper trading on live prices.\n"
            "- Once trading.mode is set to `live` (Settings tab) the bot "
            "starts placing real orders on Trading 212."
        )
    else:
        stats = pnl_stats(trades)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total closed", stats["count"],
                    help="Number of round-trip trades the bot has completed.")
        col2.metric("Win rate", f"{stats['win_rate']:.1f}%",
                    help="Percentage of closed trades with positive P&L.")
        col3.metric("Total P&L", format_currency(stats["total_pnl"], ccy),
                    help="Sum of P&L across all closed trades, net of fees and slippage.")
        col4.metric("Avg per trade", format_currency(stats["avg_pnl"], ccy),
                    help="Mean P&L per closed trade. Useful for sizing future trades.")

        st.markdown("---")

        col_sym, col_dist = st.columns([1, 1])
        with col_sym:
            st.subheader("Per-symbol performance")
            sym_df = per_symbol_stats(trades)
            if sym_df.empty:
                st.caption("No closed trades.")
            else:
                display_df = sym_df.rename(
                    columns={
                        "trades": "Trades",
                        "total": f"P&L ({ccy})",
                        "avg": f"Avg ({ccy})",
                        "wins": "Wins",
                        "losses": "Losses",
                        "win_rate": "WR %",
                    }
                )
                display_df[f"P&L ({ccy})"] = display_df[f"P&L ({ccy})"].round(2)
                display_df[f"Avg ({ccy})"] = display_df[f"Avg ({ccy})"].round(2)
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=360)

        with col_dist:
            st.subheader("P&L distribution")
            closed = _numeric_pnl(trades)
            if closed.empty:
                st.caption("No closed trades.")
            else:
                hist_df = pd.DataFrame({"P&L": closed["pnl"]})
                st.bar_chart(
                    hist_df["P&L"].value_counts(bins=20, sort=False).sort_index(),
                    height=340,
                )

        st.markdown("---")

        st.subheader("Trade history")
        symbols_available = sorted(trades["symbol"].dropna().unique().tolist())
        selected = st.multiselect("Filter symbols", symbols_available, default=symbols_available)
        view = trades[trades["symbol"].isin(selected)] if selected else trades
        st.dataframe(
            view.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download trades CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="trades.csv",
            mime="text/csv",
        )


# -------------------------- Config tab -----------------------------------
def _run_backtest_subprocess(days: int = 365) -> tuple[bool, str]:
    """Run ``python backtest.py --days N`` as a subprocess, streaming
    output to BACKTEST_LOG. Returns (success, final JSON string)."""
    BACKTEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = BACKTEST_LOG.open("w")
    try:
        result = subprocess.run(
            [sys.executable, "backtest.py", "--days", str(days)],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        log_handle.close()
        return False, "backtest timed out after 30 minutes"
    log_handle.close()
    return result.returncode == 0, BACKTEST_LOG.read_text()[-4000:]


with config_tab:
    if not cfg:
        st.error(
            "No config.yaml found. Copy config.example.yaml to config.yaml to begin."
        )
    else:
        st.info(
            "Config changes require a bot restart. The "
            "**Save & Run Backtest** button saves and re-runs the backtest "
            "with the new config – no bot restart needed for that."
        )

        with st.form("cfg"):
            with st.expander("Trading", expanded=True):
                trading = cfg.setdefault("trading", {})
                c1, c2 = st.columns(2)
                trading["mode"] = c1.selectbox(
                    "Mode", ["paper", "live"],
                    index=["paper", "live"].index(trading.get("mode", "paper")),
                )
                trading["starting_capital"] = c2.number_input(
                    "Starting capital", min_value=0.0,
                    value=float(trading.get("starting_capital", 500.0)), step=100.0,
                )
                trading["timeframe"] = st.text_input(
                    "Timeframe", value=trading.get("timeframe", "1h"),
                    help="e.g. 1h, 1d, 15m",
                )
                symbols_str = st.text_area(
                    "Symbols (one per line)",
                    value="\n".join(trading.get("symbols", [])),
                    height=160,
                    help="Tokens: SP500, NASDAQ100 auto-expand. Others treated literally.",
                )
                trading["symbols"] = [s.strip() for s in symbols_str.splitlines() if s.strip()]

            with st.expander("Risk", expanded=False):
                risk = cfg.setdefault("risk", {})
                c1, c2, c3 = st.columns(3)
                with c1:
                    risk["risk_per_trade"] = st.number_input(
                        "Risk per trade", min_value=0.0, max_value=0.5,
                        value=float(risk.get("risk_per_trade", 0.01)),
                        step=0.005, format="%.4f",
                    )
                    risk["max_position_pct"] = st.number_input(
                        "Max position %", min_value=0.0, max_value=1.0,
                        value=float(risk.get("max_position_pct", 0.25)),
                        step=0.05, format="%.2f",
                    )
                    risk["max_open_positions"] = st.number_input(
                        "Max open positions", min_value=1, max_value=50,
                        value=int(risk.get("max_open_positions", 3)), step=1,
                    )
                with c2:
                    risk["stop_loss_pct"] = st.number_input(
                        "Stop loss %", min_value=0.0, max_value=0.5,
                        value=float(risk.get("stop_loss_pct", 0.03)),
                        step=0.005, format="%.3f",
                    )
                    risk["take_profit_pct"] = st.number_input(
                        "Take profit %", min_value=0.0, max_value=1.0,
                        value=float(risk.get("take_profit_pct", 0.06)),
                        step=0.005, format="%.3f",
                    )
                    risk["trailing_stop_pct"] = st.number_input(
                        "Trailing stop %", min_value=0.0, max_value=0.5,
                        value=float(risk.get("trailing_stop_pct", 0.02)),
                        step=0.005, format="%.3f",
                    )
                with c3:
                    risk["daily_loss_limit_pct"] = st.number_input(
                        "Daily loss limit %", min_value=0.0, max_value=1.0,
                        value=float(risk.get("daily_loss_limit_pct", 0.05)),
                        step=0.01, format="%.3f",
                    )
                    risk["max_drawdown_pct"] = st.number_input(
                        "Max drawdown %", min_value=0.0, max_value=1.0,
                        value=float(risk.get("max_drawdown_pct", 0.20)),
                        step=0.05, format="%.2f",
                    )

            with st.expander("Strategy", expanded=False):
                strategy = cfg.setdefault("strategy", {})
                options = ["ensemble", "macd", "ma_crossover", "rsi_reversion", "bollinger"]
                current_name = strategy.get("name", "ensemble")
                strategy["name"] = st.selectbox(
                    "Strategy", options,
                    index=options.index(current_name) if current_name in options else 0,
                )
                ensemble = strategy.setdefault("ensemble", {})
                c1, c2 = st.columns(2)
                ensemble["min_agreement"] = c1.number_input(
                    "Ensemble min agreement", min_value=1, max_value=10,
                    value=int(ensemble.get("min_agreement", 2)), step=1,
                )
                ensemble["min_score"] = c2.number_input(
                    "Ensemble min score", min_value=0.0, max_value=10.0,
                    value=float(ensemble.get("min_score", 1.5)), step=0.1,
                )
                flt = strategy.setdefault("filter", {})
                flt["enabled"] = st.checkbox(
                    "Regime filter enabled", value=bool(flt.get("enabled", True)),
                )
                c1, c2 = st.columns(2)
                flt["trend_ema"] = c1.number_input(
                    "Trend EMA length", min_value=5, max_value=500,
                    value=int(flt.get("trend_ema", 100)), step=5,
                )
                flt["min_adx"] = c2.number_input(
                    "Min ADX", min_value=0.0, max_value=100.0,
                    value=float(flt.get("min_adx", 15.0)), step=1.0,
                )

            with st.expander("News (live only)", expanded=False):
                news = cfg.setdefault("news", {})
                news["enabled"] = st.checkbox(
                    "News feed enabled", value=bool(news.get("enabled", False)),
                )
                news["mode"] = st.selectbox(
                    "News mode", ["defensive", "aggressive"],
                    index=["defensive", "aggressive"].index(news.get("mode", "defensive")),
                )

            st.markdown("---")
            backtest_days = st.slider(
                "Backtest days", min_value=30, max_value=730,
                value=365, step=30,
                help="Used by the Save & Run Backtest button below.",
            )

            col_save, col_run = st.columns(2)
            save_only = col_save.form_submit_button(
                "Save only", use_container_width=True,
            )
            save_and_run = col_run.form_submit_button(
                "Save & Run Backtest", use_container_width=True, type="primary",
            )

        if save_only or save_and_run:
            try:
                save_yaml(CONFIG_PATH, cfg)
                st.success("Config saved.")
            except Exception as exc:
                st.error(f"Save failed: {exc}")
                save_and_run = False

        if save_and_run:
            with st.status(f"Running backtest for {backtest_days} days…", expanded=True) as status:
                st.write("Spawning `python backtest.py` as subprocess…")
                success, tail = _run_backtest_subprocess(days=backtest_days)
                if success:
                    status.update(label="Backtest complete", state="complete")
                else:
                    status.update(label="Backtest failed", state="error")
                st.code(tail, language="text")

            st.cache_data.clear()
            # Surface the freshly-written backtest summary directly.
            tr_bt = read_csv(BACKTEST_TRADES)
            if not tr_bt.empty:
                s = pnl_stats(tr_bt)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trades", s["count"])
                c2.metric("Win rate", f"{s['win_rate']:.1f}%")
                c3.metric("Total P&L", format_currency(s["total_pnl"], ccy))
                c4.metric("Avg per trade", format_currency(s["avg_pnl"], ccy))

        with st.expander("Raw YAML (read-only)"):
            st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")


# -------------------------- News tab ------------------------------------
with news_tab:
    st.subheader("Company news")
    symbols = list((cfg or {}).get("trading", {}).get("symbols", []))
    # Expand universe tokens (SP500 / NASDAQ100) exactly like the bot does so
    # the dashboard shows news for every ticker the engine would actually trade.
    try:
        from bot.universe import expand_universe_tokens
        expanded_symbols = expand_universe_tokens(symbols) if symbols else []
    except Exception as exc:  # universe fetch can fail without net
        expanded_symbols = symbols
        st.caption(f"Universe expansion failed ({exc}); using raw symbols list.")

    if not expanded_symbols:
        st.info(
            "No symbols configured. Add some to `trading.symbols` in the Config tab "
            "(e.g. `SP500`, `NASDAQ100`, or explicit tickers) and the news feed "
            "will populate automatically."
        )
    else:
        st.caption(
            f"Watching **{len(expanded_symbols)}** tickers. "
            "News comes from Yahoo Finance (free, unauthenticated) and is cached "
            "locally for 15 minutes."
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            scope = st.radio(
                "Scope",
                ["Full universe", "Open positions only", "Pick a ticker"],
                horizontal=True,
            )
        with col2:
            per_symbol = st.number_input(
                "Items / ticker", min_value=1, max_value=20, value=3, step=1
            )
        with col3:
            total_cap = st.number_input(
                "Total shown", min_value=10, max_value=500, value=100, step=10
            )

        target_symbols: list[str] = []
        if scope == "Full universe":
            # 500 symbols × a network round-trip each = slow; cap the live
            # fetch set and rely on the cache for the rest.
            batch_size = st.slider(
                "Batch size (tickers to refresh this load)",
                min_value=10, max_value=min(500, len(expanded_symbols)),
                value=min(50, len(expanded_symbols)), step=10,
                help=(
                    "Yahoo is rate-limited. Fetch news for this many tickers now; "
                    "cached news for the rest is still shown below."
                ),
            )
            target_symbols = expanded_symbols[:batch_size]
        elif scope == "Open positions only":
            trades_now = load_trades()
            if trades_now.empty:
                st.info("No open positions – no news to fetch.")
            else:
                opens = set(trades_now.loc[trades_now["side"] == "buy", "symbol"])
                closes = trades_now.loc[trades_now["side"] == "sell", "symbol"]
                open_now = sorted(opens - set(closes))
                target_symbols = open_now
                st.caption(
                    f"{len(open_now)} open position(s): {', '.join(open_now) or '—'}"
                )
        else:
            pick = st.selectbox("Ticker", expanded_symbols)
            target_symbols = [pick]

        refresh = st.button("Refresh now (bypass cache)")

        if target_symbols:
            with st.spinner(f"Fetching news for {len(target_symbols)} ticker(s)…"):
                from bot.news_feed import fetch_news_bulk, clear_cache
                if refresh:
                    for s in target_symbols:
                        clear_cache(s)
                items = fetch_news_bulk(
                    target_symbols,
                    per_symbol_limit=int(per_symbol),
                    total_limit=int(total_cap),
                )

            if not items:
                st.info(
                    "No news returned. Yahoo may be rate-limiting – wait a minute "
                    "and click refresh, or narrow the scope."
                )
            else:
                st.caption(f"Showing **{len(items)}** headline(s).")
                for item in items:
                    when = item.published_dt.strftime("%Y-%m-%d %H:%M UTC")
                    title_md = f"**[{item.title}]({item.link})**" if item.link else f"**{item.title}**"
                    st.markdown(f"`{item.symbol}` · {when} · *{item.publisher or 'unknown'}*")
                    st.markdown(title_md)
                    if item.summary:
                        st.caption(item.summary)
                    st.divider()


# -------------------------- Backtest tab --------------------------------
with backtest_tab:
    st.subheader("Latest backtest")
    st.caption(
        "Replays the strategy in your **current** config against historical "
        "yfinance data and writes the equity curve and trade log to "
        "`reports/`. Edit strategy / risk settings on the Config tab and "
        "rerun here to compare."
    )
    eq_bt = read_csv(BACKTEST_EQUITY)
    tr_bt = read_csv(BACKTEST_TRADES)

    col1, col2 = st.columns([1, 3])
    with col1:
        quick_days = st.number_input(
            "Days", min_value=30, max_value=730, value=365, step=30,
            key="quick_days",
            help="Length of the historical window. yfinance caps hourly bars "
                 "at 730 days.",
        )
        if st.button("Run backtest", use_container_width=True, type="primary",
                     help="Spawns `python backtest.py --days N` as a subprocess. "
                          "Results overwrite reports/backtest_*.csv."):
            with st.status(f"Running {quick_days}-day backtest…", expanded=True) as status:
                ok, tail = _run_backtest_subprocess(days=int(quick_days))
                if ok:
                    status.update(label="Backtest complete", state="complete")
                else:
                    status.update(label="Backtest failed", state="error")
                st.code(tail[-2000:], language="text")
            st.cache_data.clear()
            st.rerun()

    with col2:
        if eq_bt.empty and tr_bt.empty:
            st.info("No backtest output yet. Click **Run backtest** to generate.")

    if not tr_bt.empty:
        stats = pnl_stats(tr_bt)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", stats["count"])
        c2.metric("Win rate", f"{stats['win_rate']:.1f}%")
        c3.metric("Total P&L", format_currency(stats["total_pnl"], ccy))
        c4.metric("Best", format_currency(stats["best"], ccy))
        c5.metric("Worst", format_currency(stats["worst"], ccy))

        # Full stats dict from backtest.py run (sharpe/sortino/calmar/
        # max_dd/benchmark/excess return). Only present if the last run
        # wrote reports/backtest_stats.json.
        if BACKTEST_STATS.exists():
            try:
                full_stats = json.loads(BACKTEST_STATS.read_text())
            except (json.JSONDecodeError, OSError):
                full_stats = {}
            if full_stats:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Total return", f"{full_stats.get('total_return_pct', 0):.2f}%",
                          f"vs BH {full_stats.get('excess_return_pct', 0):+.2f}%")
                r2.metric("Max drawdown", f"{full_stats.get('max_drawdown_pct', 0):.2f}%")
                r3.metric("Sharpe", f"{full_stats.get('sharpe', 0):.2f}")
                r4.metric("Sortino", f"{full_stats.get('sortino', 0):.2f}")
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Calmar", f"{full_stats.get('calmar', 0):.2f}")
                q2.metric("Ulcer", f"{full_stats.get('ulcer_index', 0):.2f}")
                q3.metric("Longest win streak", full_stats.get("longest_win_streak", 0))
                q4.metric("Longest loss streak", full_stats.get("longest_loss_streak", 0))

    if not eq_bt.empty:
        eq_bt["timestamp"] = pd.to_datetime(eq_bt["timestamp"], utc=True, errors="coerce")
        st.subheader("Equity curve")
        st.line_chart(eq_bt.set_index("timestamp")[["equity"]], height=340)

    if not tr_bt.empty:
        st.subheader("Per-symbol")
        sym_df = per_symbol_stats(tr_bt)
        if not sym_df.empty:
            display_df = sym_df.rename(columns={
                "trades": "Trades", "total": f"P&L ({ccy})",
                "avg": f"Avg ({ccy})", "wins": "Wins",
                "losses": "Losses", "win_rate": "WR %",
            })
            display_df[f"P&L ({ccy})"] = display_df[f"P&L ({ccy})"].round(2)
            display_df[f"Avg ({ccy})"] = display_df[f"Avg ({ccy})"].round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Trade history")
        st.dataframe(tr_bt.sort_values("timestamp", ascending=False),
                     use_container_width=True, hide_index=True)

    # ----- Compare strategies (walk-forward) ---------------------------
    st.markdown("---")
    st.subheader("Compare strategies (walk-forward)")
    st.caption(
        "Runs six pre-built trading philosophies through the walk-forward "
        "harness on **your configured universe** and ranks them by "
        "**stability**, not headline return. Stability = how many windows "
        "were profitable, then how shallow the worst-window drawdown was. "
        "Picking the highest-return preset would be curve-fitting; picking "
        "the most consistent one is the closest thing to an honest answer "
        "to 'which config should I use?'"
    )
    cmp_col1, cmp_col2, cmp_col3 = st.columns([1, 1, 2])
    with cmp_col1:
        cmp_days = st.number_input(
            "Days", min_value=90, max_value=730, value=365, step=30,
            key="cmp_days",
            help="Length of the historical window per evaluation. 365 days "
                 "is the default; shorter windows risk being unrepresentative.",
        )
    with cmp_col2:
        cmp_windows = st.number_input(
            "Windows", min_value=2, max_value=12, value=4, step=1,
            key="cmp_windows",
            help="Number of non-overlapping walk-forward windows. More "
                 "windows = stronger consistency signal but each window "
                 "covers less history.",
        )
    with cmp_col3:
        st.caption(
            "Each preset takes 30-90 seconds depending on universe size. "
            "With parallelism on (default) the six presets run concurrently "
            "on your CPU cores, so total wall time is roughly the slowest "
            "single preset rather than their sum."
        )
    cmp_parallel = st.toggle(
        "Parallel (process pool)", value=True,
        help="Run the six presets concurrently in subprocesses. Each preset's "
             "walk-forward is fully independent so this is a clean speedup. "
             "Turn off if your sandbox forbids subprocess spawning, or for "
             "deterministic-order log output during debugging.",
    )

    if st.button("Compare strategies", type="primary",
                 help="Runs scripts/recommend_config.py-equivalent inline. "
                      "Does NOT modify config.yaml. Read the table, decide, "
                      "edit Config tab manually if you want to adopt the winner."):
        try:
            from scripts.recommend_config import PRESETS, evaluate_presets, _rank
            from bot.config import Config as _Config
            base_cfg = _Config.load(CONFIG_PATH)
        except Exception as exc:
            st.error(f"Could not load base config: {exc}")
        else:
            mode_label = "parallel" if cmp_parallel else "serial"
            with st.status(
                f"Comparing {len(PRESETS)} presets over {cmp_days}d "
                f"in {cmp_windows} windows ({mode_label})…",
                expanded=True,
            ) as status:
                done: list[str] = []
                placeholder = st.empty()
                def _on_progress(key: str) -> None:
                    done.append(key)
                    placeholder.write(
                        f"Done: {', '.join(done)} ({len(done)}/{len(PRESETS)})"
                    )
                results = evaluate_presets(
                    PRESETS, base_cfg, int(cmp_days), int(cmp_windows),
                    parallel=cmp_parallel, progress=_on_progress,
                )
                status.update(label="Comparison complete", state="complete")

            ranked = _rank(results)
            if not ranked or "error" in ranked[0]:
                st.error("All presets failed to evaluate. Check the universe "
                         "and yfinance connectivity.")
                if ranked and "error" in ranked[0]:
                    st.code(ranked[0]["error"])
            else:
                rows = []
                for i, r in enumerate(ranked, start=1):
                    if "error" in r:
                        rows.append({
                            "Rank": i, "Preset": r.get("preset", "?"),
                            "Description": "ERROR: " + r["error"][:60],
                            "Profitable windows": "-", "Worst window %": "-",
                            "Avg Sharpe": "-", "Avg return %": "-",
                            "Avg trades": "-",
                        })
                        continue
                    rows.append({
                        "Rank": i,
                        "Preset": r["preset"],
                        "Description": r["description"][:90],
                        "Profitable windows": f"{r['profitable_windows']}/{r['n_windows']}",
                        "Worst window %": f"{r['worst_return_pct']:+.2f}",
                        "Avg Sharpe": f"{r['avg_sharpe']:.2f}",
                        "Avg return %": f"{r['avg_return_pct']:+.2f}",
                        "Avg trades": f"{r['trades_avg']:.1f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                winner = ranked[0]
                consistency = winner["profitable_windows"] / max(1, winner["n_windows"])
                st.success(
                    f"**Recommendation: `{winner['preset']}`** — "
                    f"{winner['profitable_windows']}/{winner['n_windows']} profitable "
                    f"windows ({consistency:.0%}), worst window "
                    f"{winner['worst_return_pct']:+.2f}%, avg Sharpe "
                    f"{winner['avg_sharpe']:.2f}, headline return "
                    f"{winner['avg_return_pct']:+.2f}%/window."
                )
                st.warning(
                    "**This does not adopt the preset.** It tells you which "
                    "philosophy held up best on the last "
                    f"{cmp_days} days. Markets change; re-run quarterly. "
                    "To adopt, copy the preset's config from "
                    "`scripts/recommend_config.py` into your `config.yaml` "
                    "by hand and restart the bot."
                )


# -------------------------- Settings tab -------------------------------
with settings_tab:
    st.subheader("API keys")
    st.caption(
        "Stored in .env (gitignored). Required for live trading only. "
        "Leave blank to keep an existing value."
    )
    current_env = read_env()
    with st.form("env_form"):
        new_values: dict[str, str] = {}
        for key, label in API_KEY_FIELDS:
            current = current_env.get(key, "")
            placeholder = mask(current) if current else "not set"
            entry = st.text_input(
                label,
                key=f"env_{key}",
                type="password",
                placeholder=placeholder,
                help=f"Environment variable: {key}",
            )
            if entry:
                new_values[key] = entry
        submitted = st.form_submit_button("Save API keys", type="primary")
        if submitted:
            if not new_values:
                st.info("No new values provided — nothing changed.")
            else:
                try:
                    write_env(new_values)
                    st.success(
                        f"Saved {len(new_values)} key(s) to .env. Restart the bot "
                        "for changes to take effect."
                    )
                except Exception as exc:
                    st.error(f"Could not write .env: {exc}")

    st.markdown("---")
    st.subheader("Universe")
    st.caption(
        "In config.yaml -> `trading.symbols`, these tokens auto-expand:"
    )
    st.code(
        "- SP500       # ~500 S&P 500 constituents\n"
        "- NASDAQ100   # ~100 largest Nasdaq non-financials",
        language="yaml",
    )
    col_sp, col_nd = st.columns(2)
    for col, name, path in [
        (col_sp, "S&P 500", ROOT / "cache" / "sp500.csv"),
        (col_nd, "Nasdaq-100", ROOT / "cache" / "nasdaq100.csv"),
    ]:
        with col:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    refreshed = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    st.metric(f"{name} cached", f"{len(df)} tickers",
                              delta=f"refreshed {refreshed}", delta_color="off")
                except Exception as exc:
                    st.error(f"Could not read: {exc}")
            else:
                st.caption(f"{name} cache empty.")


# ---------------------------------------------------------------------------
# Auto-refresh hook. Placed at the very end so every other tab has rendered
# before the sleep kicks in; Streamlit reruns the whole script when we call
# st.rerun(), so we avoid flashing half-drawn pages.
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
