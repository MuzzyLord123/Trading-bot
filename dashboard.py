"""Local web dashboard for the trading bot.

Run with:
    streamlit run dashboard.py

Opens http://localhost:8501 in your browser. The bot itself runs
separately (``python main.py`` for live/paper). This dashboard is a
read/write UI over the config file and the CSV logs the bot writes.

Pages (sidebar):
  * Overview  – current equity, today's P&L, open positions, equity curve.
  * Trades    – full trade history with per-symbol breakdown.
  * Config    – edit config + "Save & Run Backtest" button.
  * News      – news-triggered trades.
  * Backtest  – latest backtest results.
  * Settings  – API keys, stored in .env.
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
BACKTEST_LOG = ROOT / "reports" / "last_backtest.log"

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
    page_icon="📈",
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
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
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
exchange_name = cfg.get("exchange", {}).get("name", "—")

with st.sidebar:
    st.markdown("### 📈 Trading Bot")
    equity_df = load_equity()
    if not equity_df.empty:
        last_seen = equity_df["timestamp"].iloc[-1]
        age = datetime.now(timezone.utc) - last_seen.to_pydatetime()
        if age < timedelta(minutes=5):
            st.success(f"✓ Bot active\n\nLast tick: {int(age.total_seconds())}s ago")
        elif age < timedelta(hours=1):
            st.warning(f"⚠ Stale\n\nLast tick: {int(age.total_seconds() / 60)} min ago")
        else:
            st.error(f"✗ Bot offline\n\nLast tick: {age.days}d {age.seconds // 3600}h ago")
    else:
        st.info("No equity log — bot hasn't run yet.")

    st.markdown("---")
    st.caption(f"**Mode:** {mode}")
    st.caption(f"**Exchange:** {exchange_name}")
    symbols = cfg.get("trading", {}).get("symbols", [])
    st.caption(f"**Universe:** {len(symbols)} symbols")

    st.markdown("---")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(
        "Config changes need a bot restart to take effect. "
        "Backtest runs use the saved config immediately."
    )

st.title("Trading Bot Dashboard")
st.caption(f"Running on **{exchange_name}** in **{mode}** mode")

tabs = st.tabs(["Overview", "Trades", "Config", "News", "Backtest", "Settings"])


# -------------------------- Overview tab ---------------------------------
with tabs[0]:
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
        st.subheader("Equity curve")
        if not equity.empty:
            chart_df = equity.set_index("timestamp")[["equity"]]
            st.line_chart(chart_df, height=340)
        else:
            st.info("No equity history yet. The chart populates once the bot runs.")

    with col_r:
        st.subheader("P&L stats")
        if stats["count"]:
            st.metric("Best trade", format_currency(stats["best"], ccy))
            st.metric("Worst trade", format_currency(stats["worst"], ccy))
            st.metric("Avg per trade", format_currency(stats["avg_pnl"], ccy))
        else:
            st.caption("No closed trades yet.")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Recent trades")
        if trades.empty:
            st.caption("No trades yet.")
        else:
            recent = trades.tail(12).iloc[::-1]
            st.dataframe(recent, use_container_width=True, hide_index=True, height=360)

    with col_b:
        st.subheader("Open positions")
        if trades.empty:
            st.caption("No positions.")
        else:
            open_rows = []
            for sym, grp in trades.groupby("symbol"):
                grp = grp.sort_values("timestamp")
                last = grp.iloc[-1]
                if str(last.get("reason", "")) == "entry":
                    open_rows.append(last)
            if open_rows:
                st.dataframe(
                    pd.DataFrame(open_rows)[["symbol", "price", "amount", "timestamp"]],
                    use_container_width=True, hide_index=True, height=360,
                )
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


with tabs[2]:
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
                "💾 Save only", use_container_width=True,
            )
            save_and_run = col_run.form_submit_button(
                "🚀 Save & Run Backtest", use_container_width=True, type="primary",
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
                    status.update(label="Backtest complete ✓", state="complete")
                else:
                    status.update(label="Backtest failed ✗", state="error")
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
with tabs[3]:
    st.subheader("News-driven actions")
    news_trades_df = load_trades()
    if news_trades_df.empty:
        st.info("No trades yet — no news-driven actions to show.")
    else:
        news_only = news_trades_df[
            news_trades_df["reason"].astype(str).str.contains("news", na=False)
        ]
        if news_only.empty:
            st.caption(
                "No news-triggered trades yet. Enable `news.enabled: true` "
                "in Config and run in live/paper mode."
            )
        else:
            st.dataframe(news_only.sort_values("timestamp", ascending=False),
                         use_container_width=True, hide_index=True)


# -------------------------- Backtest tab --------------------------------
with tabs[4]:
    st.subheader("Latest backtest")
    eq_bt = read_csv(BACKTEST_EQUITY)
    tr_bt = read_csv(BACKTEST_TRADES)

    col1, col2 = st.columns([1, 3])
    with col1:
        quick_days = st.number_input("Days", min_value=30, max_value=730,
                                     value=365, step=30, key="quick_days")
        if st.button("▶ Run backtest", use_container_width=True, type="primary"):
            with st.status(f"Running {quick_days}-day backtest…", expanded=True) as status:
                ok, tail = _run_backtest_subprocess(days=int(quick_days))
                if ok:
                    status.update(label="Backtest complete ✓", state="complete")
                else:
                    status.update(label="Backtest failed ✗", state="error")
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
        "In config.yaml → `trading.symbols`, these tokens auto-expand:"
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
