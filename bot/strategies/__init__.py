from __future__ import annotations

from typing import Any

from .base import Signal, Strategy, StrategyContext
from .bollinger import BollingerStrategy
from .ensemble import EnsembleStrategy
from .filtered import FilteredStrategy
from .ma_crossover import MaCrossoverStrategy
from .macd import MacdStrategy
from .rsi_reversion import RsiReversionStrategy

_REGISTRY: dict[str, type[Strategy]] = {
    "ma_crossover": MaCrossoverStrategy,
    "rsi_reversion": RsiReversionStrategy,
    "macd": MacdStrategy,
    "bollinger": BollingerStrategy,
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

    fng_enabled = bool(kwargs.pop("fng_enabled", False))
    kwargs.pop("fng_days", None)
    if fng_enabled:
        from ..sentiment import load_fear_greed
        days = int(filter_cfg.get("fng_days", 400))
        kwargs["fng_df"] = load_fear_greed(days=days)
    else:
        kwargs.pop("fng_min", None)
        kwargs.pop("fng_max", None)

    return FilteredStrategy(inner=strategy, **kwargs)


def build_strategy_from_config(
    name: str,
    params: dict[str, dict[str, Any]],
    ensemble_cfg: dict[str, Any] | None = None,
    filter_cfg: dict[str, Any] | None = None,
) -> Strategy:
    if name != "ensemble":
        return _maybe_filter(_build_leaf(name, params.get(name, {})), filter_cfg)
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
    return _maybe_filter(ens, filter_cfg)


__all__ = [
    "Signal",
    "Strategy",
    "StrategyContext",
    "build_strategy_from_config",
    "MaCrossoverStrategy",
    "RsiReversionStrategy",
    "MacdStrategy",
    "BollingerStrategy",
    "EnsembleStrategy",
    "FilteredStrategy",
]
