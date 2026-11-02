"""SmartGym: IoT / edge-AI match analysis for racket sports.

Public entry points::

    from smartgym import MatchAnalyzer, load_config
    result = MatchAnalyzer.from_config("configs/badminton.yaml").run("clip.mp4")

The package is split so that the *rules, scoring and evidence* layers are pure
NumPy + stdlib (unit-testable, edge friendly), while the perception layer
(ball / player / court detection) is pluggable between a classical CV
tracker and a trained deep model.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.config import load_config  # noqa: E402
from .core.types import Event, EventType, ScoreState, Sport  # noqa: E402
from .court.model import CourtModel, CourtSpec, BADMINTON, TENNIS  # noqa: E402
from .sports.registry import available_sports, get_rules  # noqa: E402

_LAZY = {
    "MatchAnalyzer": ("smartgym.pipeline.analyzer", "MatchAnalyzer"),
    "MatchResult": ("smartgym.pipeline.analyzer", "MatchResult"),
    "EventDetector": ("smartgym.events.detector", "EventDetector"),
    "RallyEngine": ("smartgym.analysis.rally", "RallyEngine"),
    "EvidenceRecorder": ("smartgym.evidence.recorder", "EvidenceRecorder"),
}


def __getattr__(name: str):  # PEP 562 lazy loading keeps `import smartgym` cheap
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'smartgym' has no attribute {name!r}")


__all__ = [
    "__version__",
    "load_config",
    "MatchAnalyzer",
    "MatchResult",
    "EventDetector",
    "RallyEngine",
    "EvidenceRecorder",
    "CourtModel",
    "CourtSpec",
    "BADMINTON",
    "TENNIS",
    "Event",
    "EventType",
    "ScoreState",
    "Sport",
    "get_rules",
    "available_sports",
]
