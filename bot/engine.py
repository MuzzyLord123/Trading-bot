from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

from rich.table import Table

from .config import Config
from .exchange import Exchange
from .factory import build_exchange, is_stock_platform
from .logger import CsvLogger, console
from .news import NewsMonitor, NewsSignal
from .notifications import NullNotifier, TelegramNotifier
from .portfolio import Portfolio, Position
from .risk import RiskManager
from .stocks import is_stock_market_open
from .strategies import Strategy, StrategyContext, build_strategy_from_config

log = logging.getLogger("bot.engine")


class TradingEngine:
    """Event loop for paper and live trading."""

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
        self.portfolio = Portfolio.new(cfg.trading.starting_capital)
        self.trade_log = CsvLogger(
            cfg.logging.trade_log,
            ["timestamp", "mode", "symbol", "side", "price", "amount", "pnl", "reason"],
        )
        self.equity_log = CsvLogger(
            cfg.logging.equity_log, ["timestamp", "equity", "cash", "open_positions"]
        )
        self.notifier = (
            TelegramNotifier(
                cfg.secrets["telegram_bot_token"], cfg.secrets["telegram_chat_id"]
            )
            if cfg.notifications.telegram
            else NullNotifier()
        )
        self._halted = False
        self._cooldown_ticks_left: dict[str, int] = {}
        self._bars_held: dict[str, int] = {}
        self._news_blackout_until: dict[str, float] = {}
        self._news_monitor = (
            NewsMonitor(
                symbols=cfg.trading.symbols,
                major_threshold=cfg.news.major_threshold,
                max_age_minutes=cfg.news.max_age_minutes,
            )
            if cfg.news.enabled
            else None
        )

    def run_forever(self) -> None:
        self.exchange.load_markets()
        log.info(
            "Engine starting in [bold]%s[/bold] mode on %s",
            self.cfg.trading.mode,
            self.cfg.exchange.name,
            extra={"markup": True},
        )
        self.notifier.send(
            f"Trading bot started in {self.cfg.trading.mode} mode on {self.cfg.exchange.name}"
        )
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                log.info("Interrupted – shutting down")
                self.notifier.send("Trading bot stopped (interrupt)")
                return
            except Exception as exc:
                log.exception("tick error: %s", exc)
            time.sleep(self.cfg.trading.poll_interval_seconds)

    def tick(self) -> None:
        if is_stock_platform(self.cfg) and not is_stock_market_open():
            log.debug("Market closed – skipping tick")
            return
        prices: dict[str, float] = {}
        candles: dict[str, object] = {}
        symbols = self.cfg.trading.symbols
        tf = self.cfg.trading.timeframe
        limit = self.cfg.trading.history_candles
        batch_data: dict[str, object] = {}
        if hasattr(self.exchange, "fetch_ohlcv_batch") and len(symbols) > 10:
            try:
                batch_data = self.exchange.fetch_ohlcv_batch(symbols, tf, limit=limit)
            except Exception as exc:
                log.warning("batch fetch failed, falling back to per-symbol: %s", exc)
        for symbol in symbols:
            df = batch_data.get(symbol)
            if df is None:
                df = self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if df is None or getattr(df, "empty", True):
                continue
            last_close = float(df["close"].iloc[-1])
            if not math.isfinite(last_close) or last_close <= 0:
                log.warning("invalid price for %s: %s – skipping", symbol, last_close)
                continue
            candles[symbol] = df
            prices[symbol] = last_close

        if not prices:
            log.warning("No price data this tick")
            return

        news_signals: list[NewsSignal] = []
        if self._news_monitor is not None:
            try:
                news_signals = self._news_monitor.scan()
            except Exception as exc:
                log.warning("news scan failed: %s", exc)
            for signal in news_signals:
                self._handle_news(signal, prices)

        for symbol in list(self._cooldown_ticks_left.keys()):
            self._cooldown_ticks_left[symbol] -= 1
            if self._cooldown_ticks_left[symbol] <= 0:
                del self._cooldown_ticks_left[symbol]

        for symbol, pos in list(self.portfolio.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            pos.update_trailing(price)
            self.risk.maybe_move_to_breakeven(pos, price)
            self._bars_held[symbol] = self._bars_held.get(symbol, 0) + 1
            reason = self.risk.should_exit(
                pos, price, bars_held=self._bars_held.get(symbol, 0)
            )
            if reason:
                self._close(pos, price, reason)

        equity = self.portfolio.equity(prices)
        self.portfolio.mark_day(equity)

        halt_reason = self.risk.trading_halted(self.portfolio, equity)
        if halt_reason and not self._halted:
            self._halted = True
            log.warning("Trading halted: %s", halt_reason)
            self.notifier.send(f"Trading halted: {halt_reason}")
        if halt_reason is None:
            self._halted = False
            for symbol, df in candles.items():
                ctx = StrategyContext(symbol=symbol, timeframe=self.cfg.trading.timeframe)
                sig = self.strategy.generate(df, ctx)
                pos = self.portfolio.positions.get(symbol)
                if pos is not None and sig < 0:
                    self._close(pos, prices[symbol], "exit_signal")
                elif pos is None and sig > 0 and self.risk.can_open(self.portfolio):
                    if symbol in self._cooldown_ticks_left:
                        continue
                    if self._in_news_blackout(symbol):
                        continue
                    self._open(symbol, prices[symbol], equity)

        self.equity_log.write(
            {
                "timestamp": _now_iso(),
                "equity": round(equity, 4),
                "cash": round(self.portfolio.cash, 4),
                "open_positions": len(self.portfolio.positions),
            }
        )
        self._render_status(equity, prices)

    def _open(self, symbol: str, price: float, equity: float) -> None:
        sizing = self.risk.size(
            "long", price, equity, self.portfolio.cash, portfolio=self.portfolio
        )
        if sizing.amount <= 0:
            return
        amount = self.exchange.amount_to_precision(symbol, sizing.amount)
        if amount <= 0:
            return

        if self.cfg.trading.mode == "live":
            try:
                order = self.exchange.create_market_order(symbol, "buy", amount)
            except Exception as exc:
                log.error("live order failed for %s: %s", symbol, exc)
                return
            fill_price = order.price or price
            filled_amount = order.amount or amount
            # Reject a severely under-filled order rather than opening a mis-sized
            # position whose stop/TP were computed for the full amount.
            if filled_amount < amount * 0.95:
                log.warning(
                    "partial fill on %s: requested %.8f, filled %.8f – aborting",
                    symbol, amount, filled_amount,
                )
                return
            amount = filled_amount
        else:
            fill_price = price * (1 + self.cfg.risk.slippage_pct)

        fee_cost = amount * fill_price * self.cfg.risk.taker_fee_pct
        total_cost = amount * fill_price + fee_cost
        if total_cost > self.portfolio.cash:
            return
        self.portfolio.cash -= total_cost
        self.portfolio.positions[symbol] = Position(
            symbol=symbol,
            side="long",
            amount=amount,
            entry_price=fill_price,
            stop_loss=sizing.stop_loss,
            take_profit=sizing.take_profit,
            peak_price=fill_price,
            trailing_stop_pct=self.cfg.risk.trailing_stop_pct,
            opened_at=datetime.now(timezone.utc),
        )
        self.trade_log.write(
            {
                "timestamp": _now_iso(),
                "mode": self.cfg.trading.mode,
                "symbol": symbol,
                "side": "buy",
                "price": round(fill_price, 8),
                "amount": round(amount, 8),
                "pnl": "",
                "reason": "entry",
            }
        )
        log.info("OPEN %s %.6f @ %.4f (stop %.4f, tp %.4f)",
                 symbol, amount, fill_price, sizing.stop_loss, sizing.take_profit)
        self.notifier.send(
            f"OPEN {symbol} {amount:.6f} @ {fill_price:.4f}"
        )

    def _close(self, pos: Position, price: float, reason: str) -> None:
        amount = pos.amount
        if self.cfg.trading.mode == "live":
            try:
                order = self.exchange.create_market_order(pos.symbol, "sell", amount)
            except Exception as exc:
                log.error("live close failed for %s: %s", pos.symbol, exc)
                return
            fill_price = order.price or price
        else:
            fill_price = price * (1 - self.cfg.risk.slippage_pct)

        proceeds = amount * fill_price
        fee_cost = proceeds * self.cfg.risk.taker_fee_pct
        pnl = (fill_price - pos.entry_price) * amount - fee_cost
        self.portfolio.cash += proceeds - fee_cost
        self.portfolio.realised_pnl += pnl
        symbol = pos.symbol
        del self.portfolio.positions[symbol]
        self._bars_held.pop(symbol, None)
        if pnl < 0 and self.cfg.risk.cooldown_bars_after_loss > 0:
            self._cooldown_ticks_left[symbol] = self.cfg.risk.cooldown_bars_after_loss

        self.trade_log.write(
            {
                "timestamp": _now_iso(),
                "mode": self.cfg.trading.mode,
                "symbol": pos.symbol,
                "side": "sell",
                "price": round(fill_price, 8),
                "amount": round(amount, 8),
                "pnl": round(pnl, 4),
                "reason": reason,
            }
        )
        log.info("CLOSE %s %.6f @ %.4f pnl=%.2f (%s)",
                 pos.symbol, amount, fill_price, pnl, reason)
        self.notifier.send(
            f"CLOSE {pos.symbol} @ {fill_price:.4f} pnl={pnl:.2f} ({reason})"
        )

    def _in_news_blackout(self, symbol: str) -> bool:
        until = self._news_blackout_until.get(symbol)
        return until is not None and time.time() < until

    def _handle_news(self, signal: NewsSignal, prices: dict[str, float]) -> None:
        if signal.symbol not in self.cfg.trading.symbols:
            log.debug("news symbol %s not in trading universe, ignoring", signal.symbol)
            return
        log.info(
            "news %s %+d imp=%.1f: %s",
            signal.symbol, signal.direction, signal.importance, signal.title,
        )
        self.notifier.send(
            f"NEWS {signal.symbol} {'+' if signal.direction > 0 else '-'} "
            f"{signal.title[:140]}"
        )
        price = prices.get(signal.symbol)
        pos = self.portfolio.positions.get(signal.symbol)

        if signal.direction < 0:
            # Bad news: close any open position and enter blackout.
            if pos is not None and price is not None:
                self._close(pos, price, reason="news_bearish")
            self._news_blackout_until[signal.symbol] = (
                time.time() + self.cfg.news.blackout_seconds
            )
            return

        # Bullish news.
        if self.cfg.news.mode != "aggressive":
            return
        if pos is not None or price is None:
            return
        if self._halted or not self.risk.can_open(self.portfolio):
            return
        if signal.symbol in self._cooldown_ticks_left:
            return
        equity = self.portfolio.equity(prices)
        self._open(signal.symbol, price, equity)

    def _render_status(self, equity: float, prices: dict[str, float]) -> None:
        table = Table(title=f"{self.cfg.trading.mode.upper()} snapshot")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("equity", f"{equity:,.2f} {self.cfg.trading.quote_currency}")
        table.add_row("cash", f"{self.portfolio.cash:,.2f}")
        table.add_row("realised pnl", f"{self.portfolio.realised_pnl:,.2f}")
        table.add_row("positions", str(len(self.portfolio.positions)))
        table.add_row("peak equity", f"{self.portfolio.peak_equity:,.2f}")
        for symbol, pos in self.portfolio.positions.items():
            price = prices.get(symbol, pos.entry_price)
            table.add_row(
                f"  {symbol}",
                f"{pos.amount:.6f} @ {pos.entry_price:.4f} (mkt {price:.4f})",
            )
        console().print(table)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_engine(cfg: Config) -> TradingEngine:
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
    return TradingEngine(cfg, exchange, strategy, risk)
