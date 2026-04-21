from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import Config
from .exchange import Exchange
from .logger import CsvLogger
from .portfolio import Portfolio, Position
from .risk import RiskManager
from .strategies import Strategy, StrategyContext, build_strategy_from_config

log = logging.getLogger("bot.backtest")


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list[dict] = field(default_factory=list)

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
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = -sum(t["pnl"] for t in losses)
        return {
            "final_equity": round(float(eq.iloc[-1]), 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 2),
            "trades": len(self.trades),
            "win_rate_pct": round(100 * len(wins) / len(self.trades), 2)
            if self.trades
            else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0,
            "avg_trade_pnl": round(
                sum(t["pnl"] for t in self.trades) / len(self.trades), 2
            )
            if self.trades
            else 0.0,
        }


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
        exchange: Exchange,
        strategy: Strategy,
        risk: RiskManager,
    ) -> None:
        self.cfg = cfg
        self.exchange = exchange
        self.strategy = strategy
        self.risk = risk

    def load_data(self, days: int) -> dict[str, pd.DataFrame]:
        until = datetime.now(timezone.utc)
        since = until - timedelta(days=days)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        out: dict[str, pd.DataFrame] = {}
        for symbol in self.cfg.trading.symbols:
            log.info("Fetching %s history for %d days", symbol, days)
            df = self.exchange.fetch_ohlcv_range(
                symbol, self.cfg.trading.timeframe, since_ms, until_ms
            )
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

        fee = self.cfg.risk.taker_fee_pct
        slip = self.cfg.risk.slippage_pct
        equity_rows: list[dict] = []
        last_prices: dict[str, float] = {}

        for ts in merged_index:
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
                pos.update_trailing(price)
                reason = self.risk.should_exit(pos, price)
                if reason:
                    self._close(portfolio, pos, price, ts, reason, fee, slip, result.trades)

            equity = portfolio.equity(prices)
            portfolio.mark_day(equity)
            halt = self.risk.trading_halted(portfolio, equity)

            if not halt:
                for symbol, sub in window.items():
                    ctx = StrategyContext(symbol=symbol, timeframe=self.cfg.trading.timeframe)
                    sig = self.strategy.generate(sub, ctx)
                    pos = portfolio.positions.get(symbol)

                    if pos is not None and sig < 0:
                        self._close(
                            portfolio, pos, prices[symbol], ts, "exit_signal", fee, slip, result.trades
                        )
                        continue

                    if pos is None and sig > 0 and self.risk.can_open(portfolio):
                        price = prices[symbol]
                        fill_price = price * (1 + slip)
                        sizing = self.risk.size("long", fill_price, equity, portfolio.cash)
                        if sizing.amount <= 0:
                            continue
                        cost = sizing.amount * fill_price
                        fee_cost = cost * fee
                        if cost + fee_cost > portfolio.cash:
                            continue
                        portfolio.cash -= cost + fee_cost
                        portfolio.positions[symbol] = Position(
                            symbol=symbol,
                            side="long",
                            amount=sizing.amount,
                            entry_price=fill_price,
                            stop_loss=sizing.stop_loss,
                            take_profit=sizing.take_profit,
                            peak_price=fill_price,
                            trailing_stop_pct=self.cfg.risk.trailing_stop_pct,
                            opened_at=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        )

            equity_rows.append({"timestamp": ts, "equity": equity, "cash": portfolio.cash})

        for symbol, pos in list(portfolio.positions.items()):
            price = last_prices.get(symbol, pos.entry_price)
            self._close(
                portfolio, pos, price, merged_index[-1], "end_of_test", fee, slip, result.trades
            )

        result.equity_curve = pd.DataFrame(equity_rows)
        if write_csv and not result.equity_curve.empty:
            eq_log = CsvLogger(
                "reports/backtest_equity.csv", ["timestamp", "equity", "cash"]
            )
            for row in equity_rows:
                eq_log.write({**row, "timestamp": row["timestamp"].isoformat()})
            trades_log = CsvLogger(
                "reports/backtest_trades.csv",
                ["timestamp", "symbol", "side", "price", "amount", "pnl", "reason"],
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
        fee: float,
        slip: float,
        trades: list[dict],
    ) -> None:
        fill_price = price * (1 - slip)
        proceeds = pos.amount * fill_price
        fee_cost = proceeds * fee
        pnl = (fill_price - pos.entry_price) * pos.amount - fee_cost
        portfolio.cash += proceeds - fee_cost
        portfolio.realised_pnl += pnl
        del portfolio.positions[pos.symbol]
        trades.append(
            {
                "timestamp": ts,
                "symbol": pos.symbol,
                "side": pos.side,
                "price": fill_price,
                "amount": pos.amount,
                "pnl": pnl,
                "reason": reason,
            }
        )


def _aligned_timestamps(data: dict[str, pd.DataFrame]) -> list:
    index = None
    for df in data.values():
        ts = pd.Index(df["timestamp"])
        index = ts if index is None else index.union(ts)
    return sorted(index) if index is not None else []


def build_backtester(cfg: Config) -> Backtester:
    exchange = Exchange(cfg)
    strategy = build_strategy_from_config(
        cfg.strategy.name,
        cfg.strategy.params,
        cfg.strategy.ensemble,
    )
    risk = RiskManager(cfg.risk)
    return Backtester(cfg, exchange, strategy, risk)
