from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

from rich.table import Table

from .config import Config
from .execution import buy_fill, sell_fill
from .factory import build_exchange
from .logger import CsvLogger, console
from .notifications import DesktopNotifier, MultiNotifier, NullNotifier, TelegramNotifier
from .portfolio import Portfolio, Position
from .risk import RiskManager
from .state import DEFAULT_PATH as STATE_PATH, load_portfolio, save_portfolio
from .stocks import StocksExchange, is_stock_market_open
from .strategies import Strategy, StrategyContext, build_strategy_from_config

log = logging.getLogger("bot.engine")


class TradingEngine:
    """Event loop for paper and live trading."""

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
        # Resume from the last saved portfolio state if one exists; otherwise
        # start fresh from the configured starting_capital. This keeps open
        # positions, cooldown-relevant realised P&L and peak equity across
        # restarts so stops and daily loss limits remain meaningful.
        restored = load_portfolio(STATE_PATH)
        if restored is not None:
            self.portfolio = restored
        else:
            self.portfolio = Portfolio.new(cfg.trading.starting_capital)
        self.trade_log = CsvLogger(
            cfg.logging.trade_log,
            [
                "timestamp", "mode", "symbol", "side", "price", "amount",
                "pnl", "reason", "bars_held", "mfe", "mae",
            ],
        )
        self.equity_log = CsvLogger(
            cfg.logging.equity_log, ["timestamp", "equity", "cash", "open_positions"]
        )
        # Notification channels. Telegram is always-on when configured
        # (you wouldn't enable it just to filter most messages out).
        # Desktop pop-ups are gated by severity because the same trade
        # log that's fine in a Telegram chat is too noisy as banners on
        # a busy day.
        remote_channels: list = []
        if cfg.notifications.telegram:
            remote_channels.append(TelegramNotifier(
                cfg.secrets.get("telegram_bot_token", ""),
                cfg.secrets.get("telegram_chat_id", ""),
            ))
        self.notifier = MultiNotifier(remote_channels) if remote_channels else NullNotifier()
        self.desktop = (
            DesktopNotifier() if cfg.notifications.desktop else None
        )
        self._desktop_min_severity = cfg.notifications.desktop_min_severity
        self._halted = False
        self._cooldown_ticks_left: dict[str, int] = {}
        self._bars_held: dict[str, int] = {}
        # Last-seen candle timestamp per symbol. Used to advance bars_held
        # and cooldown on a per-bar basis, not per tick — otherwise a
        # 5-minute poll on an hourly timeframe would decay these 12x faster
        # than the config intends.
        self._last_candle_ts: dict[str, object] = {}

    # Severity ranking for desktop pop-up gating. Higher index = louder.
    _SEVERITY_ORDER = {"trades": 0, "warnings": 1, "errors": 2}

    def _notify(self, message: str, severity: str = "trades", title: str | None = None) -> None:
        """Fan a short notification out to remote channels (Telegram if
        configured) and, when severity is at or above the configured
        threshold, to a desktop pop-up.

        Keep messages SHORT - desktop banners truncate around 100 chars
        on macOS and look cluttered with multi-line bodies.
        """
        # Remote channels (Telegram) get every message - the user
        # opted in to that channel knowing it would be chatty.
        try:
            self.notifier.send(message)
        except Exception as exc:
            log.debug("remote notifier failed: %s", exc)

        # Desktop is gated by severity so trade-by-trade pop-ups don't
        # bury the genuinely important "trading halted" banner.
        if self.desktop is None:
            return
        threshold = self._SEVERITY_ORDER.get(self._desktop_min_severity, 0)
        level = self._SEVERITY_ORDER.get(severity, 0)
        if level >= threshold:
            try:
                self.desktop.send(message, title=title or "Trading Bot")
            except Exception as exc:
                log.debug("desktop notifier failed: %s", exc)

    def _persist(self) -> None:
        """Snapshot the portfolio to disk. Called after every open/close
        so a crash between ticks never loses more than the current trade."""
        try:
            save_portfolio(self.portfolio, STATE_PATH)
        except Exception as exc:
            log.warning("portfolio persistence failed: %s", exc)

    def _reconcile_with_broker(self) -> None:
        """Bring the in-memory portfolio in line with what Trading 212 says
        is actually held. Only meaningful in live mode where a real broker
        sits behind the adapter."""
        broker = getattr(self.exchange, "broker", None)
        if broker is None:
            log.debug("exchange has no broker attribute; skipping reconciliation")
            return
        try:
            from .reconcile import reconcile
            report = reconcile(
                self.portfolio,
                broker,
                trailing_stop_pct=self.cfg.risk.trailing_stop_pct,
                stop_loss_pct_fallback=self.cfg.risk.stop_loss_pct,
            )
        except Exception as exc:
            log.error("reconciliation failed: %s", exc)
            return
        log.info("reconciliation: %s", report.summary())
        if not report.clean:
            # Reconciliation drift means the broker disagreed with our
            # local view - treat as a warning so it pops up on the desktop
            # even when severity is set to filter routine trade traffic.
            self._notify(
                f"Reconciled with broker: {report.summary()}",
                severity="warnings",
                title="Reconciliation",
            )
            self._persist()

    def run_forever(self) -> None:
        self.exchange.load_markets()
        log.info(
            "Engine starting in [bold]%s[/bold] mode on Trading 212",
            self.cfg.trading.mode,
            extra={"markup": True},
        )
        self._notify(
            f"Bot started in {self.cfg.trading.mode} mode",
            severity="warnings", title="Trading Bot",
        )
        # Live mode: reconcile against the broker before the first tick
        # so we don't act on stale local state. Paper mode has no broker
        # to query, so we skip.
        if self.cfg.trading.mode == "live":
            self._reconcile_with_broker()
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                log.info("Interrupted – shutting down")
                self._notify(
                    "Bot stopped (interrupt)",
                    severity="warnings", title="Trading Bot",
                )
                return
            except Exception as exc:
                log.exception("tick error: %s", exc)
                self._notify(
                    f"Tick error: {exc}"[:140],
                    severity="errors", title="Trading Bot ERROR",
                )
            time.sleep(self.cfg.trading.poll_interval_seconds)

    def tick(self) -> None:
        if not is_stock_market_open():
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

        # Detect which symbols printed a new candle this tick. bars_held
        # and cooldown decay are driven by bar count, not tick count.
        new_bar_for: set[str] = set()
        for symbol, df in candles.items():
            last_ts = df["timestamp"].iloc[-1]
            previous = self._last_candle_ts.get(symbol)
            if previous is None or last_ts != previous:
                self._last_candle_ts[symbol] = last_ts
                if previous is not None:
                    new_bar_for.add(symbol)

        for symbol in list(self._cooldown_ticks_left.keys()):
            if symbol not in new_bar_for:
                continue
            self._cooldown_ticks_left[symbol] -= 1
            if self._cooldown_ticks_left[symbol] <= 0:
                del self._cooldown_ticks_left[symbol]

        for symbol, pos in list(self.portfolio.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            pos.update_excursion(price)
            pos.update_trailing(price)
            self.risk.maybe_move_to_breakeven(pos, price)
            if symbol in new_bar_for:
                self._bars_held[symbol] = self._bars_held.get(symbol, 0) + 1
            if self.risk.should_scale_out(pos, price):
                self._scale_out(pos, price)
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
            self._notify(
                f"Trading halted: {halt_reason}",
                severity="errors", title="Trading HALTED",
            )
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
                    self._open(symbol, prices[symbol], equity, df)

        self.equity_log.write(
            {
                "timestamp": _now_iso(),
                "equity": round(equity, 4),
                "cash": round(self.portfolio.cash, 4),
                "open_positions": len(self.portfolio.positions),
            }
        )
        self._render_status(equity, prices)

    def _open(self, symbol: str, price: float, equity: float, candles=None) -> None:
        atr_value = self._atr_value(candles) if self.cfg.risk.use_atr_stop else None
        sizing = self.risk.size(
            "long", price, equity, self.portfolio.cash,
            portfolio=self.portfolio, atr=atr_value,
        )
        if sizing.amount <= 0:
            return
        amount = self.exchange.amount_to_precision(symbol, sizing.amount)
        if amount <= 0:
            return
        min_notional = self.cfg.risk.min_notional_value
        if min_notional > 0 and amount * price < min_notional:
            log.debug(
                "skipping %s: notional %.2f below min %.2f",
                symbol, amount * price, min_notional,
            )
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
            fee_cost = fill_price * amount * self.cfg.risk.taker_fee_pct
        else:
            fill = buy_fill(price, amount, self.cfg.risk)
            fill_price, fee_cost = fill.price, fill.fee_cost

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
                "bars_held": "",
                "mfe": "",
                "mae": "",
            }
        )
        log.info("OPEN %s %.6f @ %.4f (stop %.4f, tp %.4f)",
                 symbol, amount, fill_price, sizing.stop_loss, sizing.take_profit)
        self._notify(
            f"OPEN {symbol} {amount:.4f} @ {fill_price:.2f}",
            severity="trades", title="Trading Bot - OPEN",
        )
        self._persist()

    def _scale_out(self, pos: Position, price: float) -> None:
        """Close ``scale_out_fraction`` of the position and move stop to entry."""
        fraction = self.cfg.risk.scale_out_fraction
        partial = self.exchange.amount_to_precision(pos.symbol, pos.amount * fraction)
        if partial <= 0 or partial >= pos.amount:
            return
        if self.cfg.trading.mode == "live":
            try:
                order = self.exchange.create_market_order(pos.symbol, "sell", partial)
            except Exception as exc:
                log.error("live scale-out failed for %s: %s", pos.symbol, exc)
                return
            fill_price = order.price or price
            partial = order.amount or partial
            fee_cost = fill_price * partial * self.cfg.risk.taker_fee_pct
        else:
            fill = sell_fill(price, partial, self.cfg.risk)
            fill_price, fee_cost = fill.price, fill.fee_cost

        pnl = (fill_price - pos.entry_price) * partial - fee_cost
        self.portfolio.cash += partial * fill_price - fee_cost
        self.portfolio.realised_pnl += pnl
        pos.amount -= partial
        pos.scaled_out = True
        # Lock in break-even on the remainder.
        if pos.stop_loss < pos.entry_price:
            pos.stop_loss = pos.entry_price

        self.trade_log.write(
            {
                "timestamp": _now_iso(),
                "mode": self.cfg.trading.mode,
                "symbol": pos.symbol,
                "side": "sell",
                "price": round(fill_price, 8),
                "amount": round(partial, 8),
                "pnl": round(pnl, 4),
                "reason": "scale_out",
                "bars_held": self._bars_held.get(pos.symbol, 0),
                "mfe": round(pos.mfe, 4),
                "mae": round(pos.mae, 4),
            }
        )
        log.info(
            "SCALE-OUT %s %.6f @ %.4f pnl=%.2f (remaining %.6f, stop->%.4f)",
            pos.symbol, partial, fill_price, pnl, pos.amount, pos.stop_loss,
        )
        self._persist()

    def _close(self, pos: Position, price: float, reason: str) -> None:
        amount = pos.amount
        if self.cfg.trading.mode == "live":
            try:
                order = self.exchange.create_market_order(pos.symbol, "sell", amount)
            except Exception as exc:
                log.error("live close failed for %s: %s", pos.symbol, exc)
                return
            fill_price = order.price or price
            fee_cost = fill_price * amount * self.cfg.risk.taker_fee_pct
        else:
            fill = sell_fill(price, amount, self.cfg.risk)
            fill_price, fee_cost = fill.price, fill.fee_cost

        pnl = (fill_price - pos.entry_price) * amount - fee_cost
        self.portfolio.cash += amount * fill_price - fee_cost
        self.portfolio.realised_pnl += pnl
        self.risk.record_trade_result(pnl)
        symbol = pos.symbol
        bars_held = self._bars_held.get(symbol, 0)
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
                "bars_held": bars_held,
                "mfe": round(pos.mfe, 4),
                "mae": round(pos.mae, 4),
            }
        )
        log.info("CLOSE %s %.6f @ %.4f pnl=%.2f (%s)",
                 pos.symbol, amount, fill_price, pnl, reason)
        # A losing close on a stop-loss is more important than a normal
        # take-profit; bump severity so it pops on the desktop even when
        # routine trade traffic is filtered.
        sev = "warnings" if (pnl < 0 and reason == "stop_loss") else "trades"
        title = "Trading Bot - STOP HIT" if sev == "warnings" else "Trading Bot - CLOSE"
        self._notify(
            f"CLOSE {pos.symbol} @ {fill_price:.2f} pnl={pnl:+.2f} ({reason})",
            severity=sev, title=title,
        )
        self._persist()

    def _atr_value(self, candles) -> float | None:
        """Last finite ATR reading, or None if unavailable."""
        if candles is None or getattr(candles, "empty", True):
            return None
        try:
            from .indicators import atr as _atr
            series = _atr(candles, self.cfg.risk.atr_period).dropna()
        except Exception as exc:
            log.debug("atr calculation failed: %s", exc)
            return None
        if series.empty:
            return None
        value = float(series.iloc[-1])
        return value if math.isfinite(value) and value > 0 else None

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
    mtf_cfg = dict(getattr(cfg.strategy, "multi_timeframe", {}) or {})
    strategy = build_strategy_from_config(
        cfg.strategy.name,
        cfg.strategy.params,
        cfg.strategy.ensemble,
        filter_cfg,
        multi_timeframe_cfg=mtf_cfg,
    )
    risk = RiskManager(cfg.risk)
    return TradingEngine(cfg, exchange, strategy, risk)
