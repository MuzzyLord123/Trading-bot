from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import Config
from .execution import buy_fill, sell_fill
from .factory import build_exchange
from .indicators import atr as _atr
from .logger import CsvLogger
from .portfolio import Portfolio, Position
from .risk import RiskManager
from .stocks import StocksExchange
from .strategies import Strategy, StrategyContext, build_strategy_from_config

log = logging.getLogger("bot.backtest")


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list[dict] = field(default_factory=list)
    # Equal-weight buy-and-hold return across the traded universe over the
    # same window. Populated by Backtester.run. Exposes "are we actually
    # beating a do-nothing baseline?" — the single most important sanity
    # check for an active strategy.
    benchmark_return: float = 0.0

    def stats(self, starting_capital: float) -> dict[str, float]:
        if self.equity_curve.empty:
            return {}
        eq = self.equity_curve["equity"]
        returns = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / starting_capital - 1
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = float(dd.min()) if not dd.empty else 0.0
        ann_factor = _annualisation(self.equity_curve["timestamp"])
        sharpe = (
            float(returns.mean() / returns.std() * np.sqrt(ann_factor))
            if returns.std() > 0
            else 0.0
        )
        # Sortino uses only downside deviation, a better fit for strategies
        # with asymmetric return distributions (e.g. trend-following).
        downside = returns[returns < 0]
        sortino = (
            float(returns.mean() / downside.std() * np.sqrt(ann_factor))
            if len(downside) > 1 and downside.std() > 0
            else 0.0
        )
        # Calmar = annualised return / max drawdown. Rewards smooth equity curves.
        years = (
            (self.equity_curve["timestamp"].iloc[-1] - self.equity_curve["timestamp"].iloc[0])
            .total_seconds() / (365 * 24 * 3600)
        )
        ann_return = ((eq.iloc[-1] / starting_capital) ** (1 / years) - 1) if years > 0 else 0.0
        calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0
        # Longest stretch where equity is below its peak (in bars).
        underwater = (eq < peak).astype(int)
        if underwater.any():
            groups = (underwater != underwater.shift()).cumsum()
            time_underwater = int(underwater.groupby(groups).sum().max())
        else:
            time_underwater = 0

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = -sum(t["pnl"] for t in losses)
        avg_win = gross_win / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        # Expectancy = average PnL the strategy generates per trade, derived
        # from the observed win rate and payoff ratio.
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        avg_bars_held = (
            sum(t.get("bars_held", 0) for t in self.trades) / len(self.trades)
            if self.trades
            else 0.0
        )
        median_pnl = float(np.median([t["pnl"] for t in self.trades])) if self.trades else 0.0
        longest_win_streak, longest_loss_streak = _streaks(
            [t["pnl"] for t in self.trades]
        )
        # Ulcer Index — RMS of drawdowns. Unlike max_drawdown it penalises
        # both depth and duration, so strategies that crash fast and recover
        # score better than ones that grind sideways underwater.
        ulcer_index = float(np.sqrt((dd * 100).pow(2).mean())) if not dd.empty else 0.0
        return {
            "final_equity": round(float(eq.iloc[-1]), 2),
            "total_return_pct": round(total_return * 100, 2),
            "annualised_return_pct": round(ann_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "calmar": round(calmar, 2),
            "time_underwater_bars": time_underwater,
            "trades": len(self.trades),
            "win_rate_pct": round(100 * win_rate, 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0,
            "avg_trade_pnl": round(
                sum(t["pnl"] for t in self.trades) / len(self.trades), 2
            )
            if self.trades
            else 0.0,
            "expectancy": round(expectancy, 2),
            "avg_bars_held": round(avg_bars_held, 1),
            "median_trade_pnl": round(median_pnl, 2),
            "longest_win_streak": longest_win_streak,
            "longest_loss_streak": longest_loss_streak,
            "ulcer_index": round(ulcer_index, 2),
            "benchmark_return_pct": round(self.benchmark_return * 100, 2),
            "excess_return_pct": round((total_return - self.benchmark_return) * 100, 2),
        }

    def window_stats(self, n_windows: int = 4) -> list[dict]:
        """Split the equity curve into ``n_windows`` equal slices and report
        per-window return, max drawdown and Sharpe.

        Useful as a cheap stability check: if the overall Sharpe is driven by
        one good window, the strategy is unlikely to generalise.
        """
        if self.equity_curve.empty or n_windows < 1:
            return []
        n = len(self.equity_curve)
        if n < n_windows * 2:
            return []
        size = n // n_windows
        out: list[dict] = []
        for i in range(n_windows):
            start = i * size
            end = n if i == n_windows - 1 else (i + 1) * size
            window = self.equity_curve.iloc[start:end]
            if len(window) < 2:
                continue
            eq = window["equity"]
            ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
            peak = eq.cummax()
            dd = (eq - peak) / peak
            max_dd = float(dd.min()) if not dd.empty else 0.0
            rets = eq.pct_change().dropna()
            ann = _annualisation(window["timestamp"])
            sharpe = (
                float(rets.mean() / rets.std() * np.sqrt(ann))
                if rets.std() > 0
                else 0.0
            )
            out.append(
                {
                    "window": i + 1,
                    "start": str(window["timestamp"].iloc[0]),
                    "end": str(window["timestamp"].iloc[-1]),
                    "return_pct": round(ret * 100, 2),
                    "max_drawdown_pct": round(max_dd * 100, 2),
                    "sharpe": round(sharpe, 2),
                }
            )
        return out


def _streaks(pnls: list[float]) -> tuple[int, int]:
    """Longest run of wins and longest run of losses. Zero-PnL counts as a loss."""
    longest_win = longest_loss = 0
    cur_win = cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        longest_win = max(longest_win, cur_win)
        longest_loss = max(longest_loss, cur_loss)
    return longest_win, longest_loss


def _buy_and_hold_return(data: dict[str, pd.DataFrame]) -> float:
    """Equal-weight buy-and-hold return across the traded universe.

    Gives a cheap baseline: if the strategy's total_return is below this,
    you'd have done better doing nothing. This is the single most honest
    number in the report.
    """
    returns: list[float] = []
    for df in data.values():
        if len(df) < 2:
            continue
        first = float(df["close"].iloc[0])
        last = float(df["close"].iloc[-1])
        if first > 0:
            returns.append(last / first - 1)
    return sum(returns) / len(returns) if returns else 0.0


def _annualisation(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 1.0
    diffs = pd.to_datetime(timestamps).diff().dropna()
    median = diffs.median().total_seconds()
    if median <= 0:
        return 1.0
    return 365 * 24 * 3600 / median


class Backtester:
    """Long-only spot-style backtester. SHORT signals just close longs."""

    def __init__(
        self,
        cfg: Config,
        exchange: StocksExchange,
        strategy: Strategy,
        risk: RiskManager,
    ) -> None:
        self.cfg = cfg
        self.exchange = exchange
        self.strategy = strategy
        self.risk = risk

    def walk_forward(
        self,
        days: int = 180,
        n_windows: int = 4,
        data: dict[str, pd.DataFrame] | None = None,
    ) -> dict:
        """Split history into ``n_windows`` non-overlapping slices and run a
        fresh backtest on each with a reset portfolio and risk state.

        Unlike :meth:`BacktestResult.window_stats` (which slices one continuous
        run), this reruns the strategy independently per window. It's closer to
        a true out-of-sample check: wins in one window don't fund losses in the
        next, and risk halts reset at each boundary.

        Returns a dict with ``windows`` (per-window stats) and ``summary``
        (aggregate and consistency metrics).
        """
        data = data if data is not None else self.load_data(days)
        if not data:
            raise RuntimeError("No market data loaded for walk-forward")
        if n_windows < 2:
            raise ValueError("walk-forward requires at least 2 windows")

        timestamps = _aligned_timestamps(data)
        if len(timestamps) < n_windows * 2:
            raise RuntimeError(
                f"Need at least {n_windows * 2} bars for {n_windows} windows"
            )
        size = len(timestamps) // n_windows

        per_window: list[dict] = []
        for i in range(n_windows):
            start = i * size
            end = len(timestamps) if i == n_windows - 1 else (i + 1) * size
            window_ts = set(timestamps[start:end])
            window_data = {
                sym: df[df["timestamp"].isin(window_ts)].reset_index(drop=True)
                for sym, df in data.items()
            }
            window_data = {k: v for k, v in window_data.items() if not v.empty}
            if not window_data:
                continue
            self.risk.reset()
            sub_result = self.run(data=window_data, write_csv=False)
            stats = sub_result.stats(self.cfg.trading.starting_capital)
            stats["window"] = i + 1
            stats["start"] = str(timestamps[start])
            stats["end"] = str(timestamps[end - 1])
            per_window.append(stats)

        # Aggregate: average return, share of profitable windows, dispersion.
        returns = [w["total_return_pct"] for w in per_window]
        sharpes = [w["sharpe"] for w in per_window]
        summary = {
            "n_windows": len(per_window),
            "avg_return_pct": round(float(np.mean(returns)) if returns else 0.0, 2),
            "median_return_pct": round(float(np.median(returns)) if returns else 0.0, 2),
            "return_std_pct": round(float(np.std(returns)) if returns else 0.0, 2),
            "profitable_windows": sum(1 for r in returns if r > 0),
            "avg_sharpe": round(float(np.mean(sharpes)) if sharpes else 0.0, 2),
            "worst_window_return_pct": round(min(returns), 2) if returns else 0.0,
            "best_window_return_pct": round(max(returns), 2) if returns else 0.0,
        }
        return {"windows": per_window, "summary": summary}

    def load_data(self, days: int) -> dict[str, pd.DataFrame]:
        until = datetime.now(timezone.utc)
        since = until - timedelta(days=days)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        symbols = self.cfg.trading.symbols
        tf = self.cfg.trading.timeframe
        out: dict[str, pd.DataFrame] = {}
        # For large universes, prefer a single batched download.
        if hasattr(self.exchange, "fetch_ohlcv_batch") and len(symbols) > 10:
            log.info("Batch-fetching %d symbols for %d days", len(symbols), days)
            try:
                limit = max(days + 5, 50)
                out = self.exchange.fetch_ohlcv_batch(symbols, tf, limit=limit)
            except Exception as exc:
                log.warning("batch fetch failed, falling back per-symbol: %s", exc)
                out = {}
        for symbol in symbols:
            if symbol in out and not out[symbol].empty:
                continue
            log.info("Fetching %s history for %d days", symbol, days)
            try:
                df = self.exchange.fetch_ohlcv_range(symbol, tf, since_ms, until_ms)
            except Exception as exc:
                log.warning("Fetch failed for %s: %s", symbol, exc)
                continue
            if df.empty:
                log.warning("No data for %s, skipping", symbol)
                continue
            out[symbol] = df
        return out

    def run(
        self,
        days: int = 180,
        write_csv: bool = True,
        data: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        data = data if data is not None else self.load_data(days)
        if not data:
            raise RuntimeError("No market data loaded for backtest")

        merged_index = _aligned_timestamps(data)
        portfolio = Portfolio.new(self.cfg.trading.starting_capital)
        result = BacktestResult(equity_curve=pd.DataFrame())

        cooldown = self.cfg.risk.cooldown_bars_after_loss
        equity_rows: list[dict] = []
        last_prices: dict[str, float] = {}
        cooldown_until: dict[str, int] = {}

        opened_at_bar: dict[str, int] = {}

        for bar_idx, ts in enumerate(merged_index):
            prices: dict[str, float] = {}
            window: dict[str, pd.DataFrame] = {}
            for symbol, df in data.items():
                sub = df[df["timestamp"] <= ts]
                if sub.empty:
                    continue
                window[symbol] = sub
                prices[symbol] = float(sub["close"].iloc[-1])

            if not prices:
                continue
            last_prices = prices

            for symbol, pos in list(portfolio.positions.items()):
                price = prices.get(symbol)
                if price is None:
                    continue
                pos.update_excursion(price)
                pos.update_trailing(price)
                self.risk.maybe_move_to_breakeven(pos, price)
                bars_held = bar_idx - opened_at_bar.get(symbol, bar_idx)
                if self.risk.should_scale_out(pos, price):
                    self._scale_out(portfolio, pos, price, ts, result.trades, bars_held)
                reason = self.risk.should_exit(pos, price, bars_held=bars_held)
                if reason:
                    pnl = self._close(
                        portfolio, pos, price, ts, reason, result.trades, bars_held
                    )
                    opened_at_bar.pop(symbol, None)
                    if pnl < 0:
                        cooldown_until[symbol] = bar_idx + cooldown

            equity = portfolio.equity(prices)
            portfolio.mark_day(equity)
            halt = self.risk.trading_halted(portfolio, equity)

            if not halt:
                for symbol, sub in window.items():
                    ctx = StrategyContext(symbol=symbol, timeframe=self.cfg.trading.timeframe)
                    sig = self.strategy.generate(sub, ctx)
                    pos = portfolio.positions.get(symbol)

                    if pos is not None and sig < 0:
                        bars_held = bar_idx - opened_at_bar.get(symbol, bar_idx)
                        pnl = self._close(
                            portfolio, pos, prices[symbol], ts, "exit_signal",
                            result.trades, bars_held,
                        )
                        opened_at_bar.pop(symbol, None)
                        if pnl < 0:
                            cooldown_until[symbol] = bar_idx + cooldown
                        continue

                    if pos is None and sig > 0 and self.risk.can_open(portfolio):
                        if bar_idx < cooldown_until.get(symbol, 0):
                            continue
                        fill = buy_fill(prices[symbol], 1.0, self.cfg.risk)
                        atr_value = _last_atr(sub, self.cfg.risk) if self.cfg.risk.use_atr_stop else None
                        sizing = self.risk.size(
                            "long", fill.price, equity, portfolio.cash,
                            portfolio=portfolio, atr=atr_value,
                        )
                        if sizing.amount <= 0:
                            continue
                        min_notional = self.cfg.risk.min_notional_value
                        if (
                            min_notional > 0
                            and sizing.amount * prices[symbol] < min_notional
                        ):
                            continue
                        entry_fill = buy_fill(prices[symbol], sizing.amount, self.cfg.risk)
                        if entry_fill.cash_out > portfolio.cash:
                            continue
                        portfolio.cash -= entry_fill.cash_out
                        portfolio.positions[symbol] = Position(
                            symbol=symbol,
                            side="long",
                            amount=entry_fill.amount,
                            entry_price=entry_fill.price,
                            stop_loss=sizing.stop_loss,
                            take_profit=sizing.take_profit,
                            peak_price=entry_fill.price,
                            trailing_stop_pct=self.cfg.risk.trailing_stop_pct,
                            opened_at=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        )
                        opened_at_bar[symbol] = bar_idx

            equity_rows.append({"timestamp": ts, "equity": equity, "cash": portfolio.cash})

        for symbol, pos in list(portfolio.positions.items()):
            price = last_prices.get(symbol, pos.entry_price)
            bars_held = len(merged_index) - 1 - opened_at_bar.get(symbol, len(merged_index) - 1)
            self._close(
                portfolio, pos, price, merged_index[-1], "end_of_test",
                result.trades, bars_held,
            )

        result.equity_curve = pd.DataFrame(equity_rows)
        result.benchmark_return = _buy_and_hold_return(data)
        if write_csv and not result.equity_curve.empty:
            eq_log = CsvLogger(
                "reports/backtest_equity.csv", ["timestamp", "equity", "cash"]
            )
            for row in equity_rows:
                eq_log.write({**row, "timestamp": row["timestamp"].isoformat()})
            trades_log = CsvLogger(
                "reports/backtest_trades.csv",
                [
                    "timestamp", "symbol", "side", "price", "amount",
                    "pnl", "reason", "bars_held", "mfe", "mae",
                ],
            )
            for t in result.trades:
                trades_log.write({**t, "timestamp": t["timestamp"].isoformat()})
        return result

    def _close(
        self,
        portfolio: Portfolio,
        pos: Position,
        price: float,
        ts,
        reason: str,
        trades: list[dict],
        bars_held: int = 0,
    ) -> float:
        fill = sell_fill(price, pos.amount, self.cfg.risk)
        pnl = (fill.price - pos.entry_price) * fill.amount - fill.fee_cost
        portfolio.cash += fill.cash_in
        portfolio.realised_pnl += pnl
        self.risk.record_trade_result(pnl)
        del portfolio.positions[pos.symbol]
        trades.append(
            {
                "timestamp": ts,
                "symbol": pos.symbol,
                "side": pos.side,
                "price": fill.price,
                "amount": fill.amount,
                "pnl": pnl,
                "reason": reason,
                "bars_held": bars_held,
                "mfe": round(pos.mfe, 6),
                "mae": round(pos.mae, 6),
            }
        )
        return pnl

    def _scale_out(
        self,
        portfolio: Portfolio,
        pos: Position,
        price: float,
        ts,
        trades: list[dict],
        bars_held: int,
    ) -> None:
        partial = pos.amount * self.cfg.risk.scale_out_fraction
        if partial <= 0 or partial >= pos.amount:
            return
        fill = sell_fill(price, partial, self.cfg.risk)
        pnl = (fill.price - pos.entry_price) * fill.amount - fill.fee_cost
        portfolio.cash += fill.cash_in
        portfolio.realised_pnl += pnl
        pos.amount -= fill.amount
        pos.scaled_out = True
        if pos.stop_loss < pos.entry_price:
            pos.stop_loss = pos.entry_price
        trades.append(
            {
                "timestamp": ts,
                "symbol": pos.symbol,
                "side": pos.side,
                "price": fill.price,
                "amount": fill.amount,
                "pnl": pnl,
                "reason": "scale_out",
                "bars_held": bars_held,
                "mfe": round(pos.mfe, 6),
                "mae": round(pos.mae, 6),
            }
        )


def _last_atr(df: pd.DataFrame, risk_cfg) -> float | None:
    period = risk_cfg.atr_period
    if len(df) < period + 1:
        return None
    try:
        series = _atr(df, period).dropna()
    except Exception:
        return None
    if series.empty:
        return None
    value = float(series.iloc[-1])
    return value if np.isfinite(value) and value > 0 else None


def _aligned_timestamps(data: dict[str, pd.DataFrame]) -> list:
    index = None
    for df in data.values():
        ts = pd.Index(df["timestamp"])
        index = ts if index is None else index.union(ts)
    return sorted(index) if index is not None else []


def build_backtester(cfg: Config) -> Backtester:
    exchange = build_exchange(cfg)
    filter_cfg = dict(cfg.strategy.filter or {})
    if int(filter_cfg.get("earnings_blackout_days", 0)) > 0:
        filter_cfg["earnings_symbols"] = list(cfg.trading.symbols)
    strategy = build_strategy_from_config(
        cfg.strategy.name,
        cfg.strategy.params,
        cfg.strategy.ensemble,
        filter_cfg,
    )
    risk = RiskManager(cfg.risk)
    return Backtester(cfg, exchange, strategy, risk)
