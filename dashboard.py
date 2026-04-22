"""Local web dashboard for the trading bot.

Run with:
    streamlit run dashboard.py

Opens http://localhost:8501 in your browser. The bot itself runs
separately (``python main.py`` for live/paper). This dashboard is a
read/write UI over the config file and the CSV logs the bot writes.

Key pages:
  * Overview  – current equity, today's P&L, open positions, equity curve.
  * Trades    – full trade history with filters.
  * Config    – form to edit the main config fields and save the YAML.
  * News      – recent news-driven actions parsed from the bot log.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
CONFIG_EXAMPLE_PATH = ROOT / "config.example.yaml"
STOCKS_EXAMPLE_PATH = ROOT / "config.stocks.example.yaml"
ENV_PATH = ROOT / ".env"
TRADES_PATH = ROOT / "logs" / "trades.csv"
EQUITY_PATH = ROOT / "logs" / "equity.csv"
BACKTEST_EQUITY = ROOT / "reports" / "backtest_equity.csv"
BACKTEST_TRADES = ROOT / "reports" / "backtest_trades.csv"

API_KEY_FIELDS = [
    ("TRADING212_API_KEY", "Trading 212 API key"),
    ("EXCHANGE_API_KEY", "Crypto exchange API key"),
    ("EXCHANGE_API_SECRET", "Crypto exchange API secret"),
    ("EXCHANGE_API_PASSWORD", "Crypto exchange API password"),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token"),
    ("TELEGRAM_CHAT_ID", "Telegram chat ID"),
]


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
    """Merge ``updates`` into ``.env`` preserving comments and order."""
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

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")


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
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def format_currency(value: float, ccy: str = "GBP") -> str:
    sign = "£" if ccy == "GBP" else ""
    return f"{sign}{value:,.2f}"


def kpi_card(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label, value, delta)


# ---------------------------------------------------------------------------
# Data loading (cached to keep the UI snappy).
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10)
def load_trades() -> pd.DataFrame:
    df = read_csv(TRADES_PATH)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


@st.cache_data(ttl=10)
def load_equity() -> pd.DataFrame:
    df = read_csv(EQUITY_PATH)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def pnl_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "pnl" not in trades.columns:
        return {"count": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "best": 0.0, "worst": 0.0}
    closed = trades[trades["pnl"].notna() & (trades["pnl"] != "")].copy()
    closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce")
    closed = closed.dropna(subset=["pnl"])
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


# ---------------------------------------------------------------------------
# Layout.
# ---------------------------------------------------------------------------
cfg = load_yaml(CONFIG_PATH)
ccy = cfg.get("trading", {}).get("quote_currency", "GBP")
starting_capital = float(cfg.get("trading", {}).get("starting_capital", 0.0))

st.title("Trading Bot Dashboard")
mode = cfg.get("trading", {}).get("mode", "paper").upper()
exchange_name = cfg.get("exchange", {}).get("name", "—")
st.caption(f"{mode} mode · {exchange_name}")

tabs = st.tabs(["Overview", "Trades", "Config", "News", "Backtest", "Settings"])


# -------------------------- Overview tab ---------------------------------
with tabs[0]:
    trades = load_trades()
    equity = load_equity()
    stats = pnl_stats(trades)

    if equity.empty:
        current_equity = starting_capital
        day_pnl = 0.0
    else:
        current_equity = float(equity["equity"].iloc[-1])
        today_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent = equity[equity["timestamp"] >= today_cutoff]
        if len(recent) >= 2:
            day_pnl = float(recent["equity"].iloc[-1] - recent["equity"].iloc[0])
        else:
            day_pnl = 0.0

    total_pnl = current_equity - starting_capital if starting_capital else stats["total_pnl"]
    pnl_pct = (total_pnl / starting_capital * 100) if starting_capital else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Current Equity", format_currency(current_equity, ccy),
                 f"{pnl_pct:+.2f}% from start")
    with col2:
        kpi_card("24h P&L", format_currency(day_pnl, ccy))
    with col3:
        kpi_card("Total P&L", format_currency(total_pnl, ccy))
    with col4:
        kpi_card("Win rate", f"{stats['win_rate']:.1f}%",
                 f"{stats['count']} closed trades")

    st.divider()

    if not equity.empty:
        st.subheader("Equity curve")
        chart_df = equity.set_index("timestamp")[["equity"]]
        st.line_chart(chart_df, height=320)
    else:
        st.info("No equity history yet. Once the bot has run a few ticks, this chart will populate.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Recent trades")
        if trades.empty:
            st.caption("No trades yet.")
        else:
            recent = trades.tail(10).iloc[::-1]
            st.dataframe(recent, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Open positions (live / paper)")
        if trades.empty:
            st.caption("No positions.")
        else:
            # Infer open positions from the trade log: a symbol whose latest
            # row is a buy with no matching sell after it.
            open_rows = []
            for sym, grp in trades.groupby("symbol"):
                grp = grp.sort_values("timestamp")
                last = grp.iloc[-1]
                if str(last.get("side", "")).lower() == "buy" or str(last.get("reason", "")) == "entry":
                    open_rows.append(last)
            if open_rows:
                st.dataframe(pd.DataFrame(open_rows)[["symbol", "price", "amount", "timestamp"]],
                             use_container_width=True, hide_index=True)
            else:
                st.caption("No open positions.")


# -------------------------- Trades tab -----------------------------------
with tabs[1]:
    trades = load_trades()
    if trades.empty:
        st.info("No trade history yet. Run the bot in paper or live mode to populate this.")
    else:
        stats = pnl_stats(trades)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total closed", stats["count"])
        col2.metric("Win rate", f"{stats['win_rate']:.1f}%")
        col3.metric("Total P&L", format_currency(stats["total_pnl"], ccy))
        col4.metric("Avg per trade", format_currency(stats["avg_pnl"], ccy))

        symbols = sorted(trades["symbol"].dropna().unique().tolist())
        selected = st.multiselect("Filter symbols", symbols, default=symbols)
        view = trades[trades["symbol"].isin(selected)] if selected else trades
        st.dataframe(
            view.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# -------------------------- Config tab -----------------------------------
with tabs[2]:
    if not cfg:
        st.error(
            "No config.yaml found. Create one from config.example.yaml "
            "(crypto) or config.stocks.example.yaml (stocks)."
        )
    else:
        st.warning("Config changes require a bot restart to take effect.")

        with st.form("cfg"):
            st.subheader("Trading")
            trading = cfg.setdefault("trading", {})
            trading["mode"] = st.selectbox(
                "Mode", ["paper", "live"],
                index=["paper", "live"].index(trading.get("mode", "paper")),
            )
            trading["starting_capital"] = st.number_input(
                "Starting capital", min_value=0.0,
                value=float(trading.get("starting_capital", 500.0)), step=100.0,
            )
            symbols_str = st.text_area(
                "Symbols (one per line)",
                value="\n".join(trading.get("symbols", [])),
                height=160,
            )
            trading["symbols"] = [s.strip() for s in symbols_str.splitlines() if s.strip()]
            trading["timeframe"] = st.text_input("Timeframe", value=trading.get("timeframe", "1h"))

            st.subheader("Risk")
            risk = cfg.setdefault("risk", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                risk["risk_per_trade"] = st.number_input(
                    "Risk per trade", min_value=0.0, max_value=0.5,
                    value=float(risk.get("risk_per_trade", 0.01)), step=0.005, format="%.4f",
                )
                risk["max_position_pct"] = st.number_input(
                    "Max position %", min_value=0.0, max_value=1.0,
                    value=float(risk.get("max_position_pct", 0.25)), step=0.05, format="%.2f",
                )
                risk["max_open_positions"] = st.number_input(
                    "Max open positions", min_value=1, max_value=50,
                    value=int(risk.get("max_open_positions", 3)), step=1,
                )
            with col2:
                risk["stop_loss_pct"] = st.number_input(
                    "Stop loss %", min_value=0.0, max_value=0.5,
                    value=float(risk.get("stop_loss_pct", 0.03)), step=0.005, format="%.3f",
                )
                risk["take_profit_pct"] = st.number_input(
                    "Take profit %", min_value=0.0, max_value=1.0,
                    value=float(risk.get("take_profit_pct", 0.06)), step=0.005, format="%.3f",
                )
                risk["trailing_stop_pct"] = st.number_input(
                    "Trailing stop %", min_value=0.0, max_value=0.5,
                    value=float(risk.get("trailing_stop_pct", 0.02)), step=0.005, format="%.3f",
                )
            with col3:
                risk["daily_loss_limit_pct"] = st.number_input(
                    "Daily loss limit %", min_value=0.0, max_value=1.0,
                    value=float(risk.get("daily_loss_limit_pct", 0.05)), step=0.01, format="%.3f",
                )
                risk["max_drawdown_pct"] = st.number_input(
                    "Max drawdown %", min_value=0.0, max_value=1.0,
                    value=float(risk.get("max_drawdown_pct", 0.20)), step=0.05, format="%.2f",
                )

            st.subheader("Strategy")
            strategy = cfg.setdefault("strategy", {})
            options = ["ensemble", "macd", "ma_crossover", "rsi_reversion", "bollinger"]
            current_name = strategy.get("name", "ensemble")
            strategy["name"] = st.selectbox(
                "Strategy",
                options,
                index=options.index(current_name) if current_name in options else 0,
            )

            ensemble = strategy.setdefault("ensemble", {})
            col_a, col_b = st.columns(2)
            with col_a:
                ensemble["min_agreement"] = st.number_input(
                    "Ensemble min agreement", min_value=1, max_value=10,
                    value=int(ensemble.get("min_agreement", 2)), step=1,
                )
            with col_b:
                ensemble["min_score"] = st.number_input(
                    "Ensemble min score", min_value=0.0, max_value=10.0,
                    value=float(ensemble.get("min_score", 1.5)), step=0.1,
                )

            flt = strategy.setdefault("filter", {})
            flt["enabled"] = st.checkbox("Regime filter enabled",
                                         value=bool(flt.get("enabled", True)))
            col_x, col_y = st.columns(2)
            with col_x:
                flt["trend_ema"] = st.number_input(
                    "Trend EMA length", min_value=10, max_value=500,
                    value=int(flt.get("trend_ema", 100)), step=10,
                )
            with col_y:
                flt["min_adx"] = st.number_input(
                    "Min ADX", min_value=0.0, max_value=100.0,
                    value=float(flt.get("min_adx", 15.0)), step=1.0,
                )

            st.subheader("News (live mode only)")
            news = cfg.setdefault("news", {})
            news["enabled"] = st.checkbox("News feed enabled",
                                          value=bool(news.get("enabled", False)))
            news["mode"] = st.selectbox(
                "News mode", ["defensive", "aggressive"],
                index=["defensive", "aggressive"].index(news.get("mode", "defensive")),
            )

            submitted = st.form_submit_button("Save config")
            if submitted:
                try:
                    save_yaml(CONFIG_PATH, cfg)
                    st.success("Saved. Restart the bot for changes to take effect.")
                except Exception as exc:
                    st.error(f"Could not save: {exc}")

        with st.expander("Raw YAML (read-only)"):
            st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")


# -------------------------- News tab ------------------------------------
with tabs[3]:
    st.subheader("News-driven actions")
    news_trades_df = load_trades()
    if news_trades_df.empty:
        st.info("No trades yet, so no news-driven actions to show.")
    else:
        news_only = news_trades_df[
            news_trades_df["reason"].astype(str).str.contains("news", na=False)
        ]
        if news_only.empty:
            st.caption(
                "No news-triggered trades yet. Enable `news.enabled: true` "
                "in config and run in live/paper mode."
            )
        else:
            st.dataframe(news_only.sort_values("timestamp", ascending=False),
                         use_container_width=True, hide_index=True)


# -------------------------- Settings tab -------------------------------
with tabs[5]:
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
        submitted = st.form_submit_button("Save API keys")
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

    st.divider()
    st.subheader("Universe expansion")
    st.caption(
        "In config.yaml under `trading.symbols`, you can use these tokens "
        "to auto-expand to a whole index:"
    )
    st.code("- SP500     # all ~500 S&P 500 stocks", language="yaml")
    sp_cache = ROOT / "cache" / "sp500.csv"
    if sp_cache.exists():
        try:
            sp_df = pd.read_csv(sp_cache)
            st.metric("S&P 500 cached", f"{len(sp_df)} tickers",
                      delta=f"refreshed {datetime.fromtimestamp(sp_cache.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
            with st.expander("Show cached tickers"):
                st.dataframe(sp_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not read S&P 500 cache: {exc}")
    else:
        st.caption("S&P 500 cache empty. It will populate on next bot start / backtest.")


# -------------------------- Backtest tab --------------------------------
with tabs[4]:
    st.subheader("Latest backtest")
    eq_bt = read_csv(BACKTEST_EQUITY)
    tr_bt = read_csv(BACKTEST_TRADES)
    if eq_bt.empty and tr_bt.empty:
        st.info("No backtest output yet. Run `python backtest.py --days 365` to populate.")
    else:
        if not eq_bt.empty:
            eq_bt["timestamp"] = pd.to_datetime(eq_bt["timestamp"], utc=True, errors="coerce")
            st.line_chart(eq_bt.set_index("timestamp")[["equity"]], height=320)
        if not tr_bt.empty:
            stats = pnl_stats(tr_bt)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trades", stats["count"])
            col2.metric("Win rate", f"{stats['win_rate']:.1f}%")
            col3.metric("Total P&L", format_currency(stats["total_pnl"], ccy))
            col4.metric("Best / Worst",
                        f"{format_currency(stats['best'], ccy)} / {format_currency(stats['worst'], ccy)}")
            st.dataframe(tr_bt.sort_values("timestamp", ascending=False),
                         use_container_width=True, hide_index=True)
