"""Sport registry: maps a config name to its ``SportRules`` implementation."""

from __future__ import annotations

from typing import Dict, List, Type

from ..core.types import Sport
from .base import SportRules
from .badminton import BadmintonMatch, BadmintonRules
from .tennis import TennisMatch, TennisRules


_REGISTRY: Dict[str, Type[SportRules]] = {
    Sport.BADMINTON.value: BadmintonRules,
    Sport.TENNIS.value: TennisRules,
}

_ALIASES = {
    "badminton": "badminton",
    "shuttle": "badminton",
    "badmin": "badminton",
    "tennis": "tennis",
    "lawn_tennis": "tennis",
}


def available_sports() -> List[str]:
    return sorted(_REGISTRY)


def get_rules(sport: str) -> SportRules:
    """Return a (stateless) rules instance for ``sport``."""
    key = _ALIASES.get(str(sport).strip().lower(), str(sport).strip().lower())
    if key not in _REGISTRY:
        raise KeyError(
            f"unsupported sport {sport!r}; available: {available_sports()} "
            "(volleyball/basketball ship a camera-only detector scaffold, see docs/ROADMAP.md)"
        )
    return _REGISTRY[key]()


__all__ = ["get_rules", "available_sports", "BadmintonMatch", "TennisMatch",
           "BadmintonRules", "TennisRules"]
