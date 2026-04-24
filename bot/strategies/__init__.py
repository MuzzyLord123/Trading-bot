from __future__ import annotations

from typing import Any

from .base import Signal, Strategy, StrategyContext
from .bollinger import BollingerStrategy
from .cci import CciStrategy
from .donchian import DonchianStrategy
from .ensemble import EnsembleStrategy
from .filtered import FilteredStrategy
from .keltner import KeltnerStrategy
from .ma_crossover import MaCrossoverStrategy
from .macd import MacdStrategy
from .multi_timeframe import MultiTimeframeStrategy
from .obv_trend import ObvTrendStrategy
from .rsi_reversion import RsiReversionStrategy
from .stochastic import StochasticStrategy

_REGISTRY: dict[str, type[Strategy]] = {
    "ma_crossover": MaCrossoverStrategy,
    "rsi_reversion": RsiReversionStrategy,
    "macd": MacdStrategy,
    "bollinger": BollingerStrategy,
    "stochastic": StochasticStrategy,
    "donchian": DonchianStrategy,
    "keltner": KeltnerStrategy,
    "cci": CciStrategy,
    "obv_trend": ObvTrendStrategy,
    "ensemble": EnsembleStrategy,
}


def _build_leaf(name: str, params: dict[str, Any]) -> Strategy:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown strategy '{name}'. Known: {sorted(_REGISTRY)}"
        ) from None
    if cls is EnsembleStrategy:
        raise ValueError("ensemble cannot be nested inside another ensemble here")
    return cls(**params)


def _maybe_filter(
    strategy: Strategy, filter_cfg: dict[str, Any] | None
) -> Strategy:
    if not filter_cfg or not filter_cfg.get("enabled", False):
        return strategy
    kwargs = {k: v for k, v in filter_cfg.items() if k != "enabled"}

    # Silently drop legacy Fear & Greed keys so old configs still load.
    for legacy in ("fng_enabled", "fng_days", "fng_min", "fng_max"):
        kwargs.pop(legacy, None)

    earnings_blackout = int(kwargs.pop("earnings_blackout_days", 0))
    earnings_symbols = kwargs.pop("earnings_symbols", None)
    if earnings_blackout > 0 and earnings_symbols:
        from ..news_stocks import load_earnings_calendar
        kwargs["earnings_calendar"] = load_earnings_calendar(list(earnings_symbols))
        kwargs["earnings_blackout_days"] = earnings_blackout
    else:
        kwargs.pop("earnings_calendar", None)
        kwargs["earnings_blackout_days"] = earnings_blackout

    return FilteredStrategy(inner=strategy, **kwargs)


def _maybe_multi_timeframe(
    strategy: Strategy, mtf_cfg: dict[str, Any] | None,
) -> Strategy:
    """Optionally wrap ``strategy`` in :class:`MultiTimeframeStrategy`.

    ``mtf_cfg`` is a dict-like object with optional ``enabled`` (default
    False), ``rule`` (e.g. "1D", auto-derived from the trading timeframe
    when None) and ``require_long_on_htf`` (default True).
    """
    if not mtf_cfg or not mtf_cfg.get("enabled", False):
        return strategy
    return MultiTimeframeStrategy(
        inner=strategy,
        htf_rule=mtf_cfg.get("rule"),
        require_long_on_htf=bool(mtf_cfg.get("require_long_on_htf", True)),
    )


def build_strategy_from_config(
    name: str,
    params: dict[str, dict[str, Any]],
    ensemble_cfg: dict[str, Any] | None = None,
    filter_cfg: dict[str, Any] | None = None,
    multi_timeframe_cfg: dict[str, Any] | None = None,
) -> Strategy:
    if name != "ensemble":
        core = _maybe_filter(_build_leaf(name, params.get(name, {})), filter_cfg)
        return _maybe_multi_timeframe(core, multi_timeframe_cfg)
    ensemble_cfg = ensemble_cfg or {}
    member_names: list[str] = ensemble_cfg.get("members") or list(
        k for k in _REGISTRY if k != "ensemble"
    )
    members = [_build_leaf(n, params.get(n, {})) for n in member_names]
    weights_cfg = ensemble_cfg.get("weights") or {}
    weights = [float(weights_cfg.get(n, 1.0)) for n in member_names]
    ens = EnsembleStrategy(
        members=members,
        weights=weights,
        min_agreement=int(ensemble_cfg.get("min_agreement", 2)),
        min_score=float(ensemble_cfg.get("min_score", 1.5)),
    )
    core = _maybe_filter(ens, filter_cfg)
    return _maybe_multi_timeframe(core, multi_timeframe_cfg)


__all__ = [
    "Signal",
    "Strategy",
    "StrategyContext",
    "build_strategy_from_config",
    "MaCrossoverStrategy",
    "RsiReversionStrategy",
    "MacdStrategy",
    "BollingerStrategy",
    "StochasticStrategy",
    "DonchianStrategy",
    "KeltnerStrategy",
    "CciStrategy",
    "ObvTrendStrategy",
    "EnsembleStrategy",
    "FilteredStrategy",
    "MultiTimeframeStrategy",
]
