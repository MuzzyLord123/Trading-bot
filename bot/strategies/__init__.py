from __future__ import annotations

from typing import Any

from .base import Signal, Strategy, StrategyContext
from .bollinger import BollingerStrategy
from .ensemble import EnsembleStrategy
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


def build_strategy_from_config(
    name: str,
    params: dict[str, dict[str, Any]],
    ensemble_cfg: dict[str, Any] | None = None,
) -> Strategy:
    if name != "ensemble":
        return _build_leaf(name, params.get(name, {}))
    ensemble_cfg = ensemble_cfg or {}
    member_names: list[str] = ensemble_cfg.get("members") or list(
        k for k in _REGISTRY if k != "ensemble"
    )
    members = [_build_leaf(n, params.get(n, {})) for n in member_names]
    return EnsembleStrategy(
        members=members,
        min_agreement=int(ensemble_cfg.get("min_agreement", 2)),
    )


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
]
