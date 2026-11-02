"""Shared data types for the SmartGym pipeline.

Everything here is plain Python / NumPy so that it can be serialised to JSON on
an edge device without pulling in heavy dependencies.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Sport(str, Enum):
    BADMINTON = "badminton"
    TENNIS = "tennis"
    VOLLEYBALL = "volleyball"
    BASKETBALL = "basketball"


class EventType(str, Enum):
    """Atomic observations produced by perception, consumed by the rule engine."""

    SERVE = "serve"                 # serve struck (side known)
    SHOT = "shot"                   # a mid-rally hit (racket contact)
    LANDING = "landing"             # physical floor contact, not yet adjudicated
    NET_CONTACT = "net_contact"     # ball touched the net / failed to clear
    BOUNCE_IN = "bounce_in"         # ball/shuttle landed inside the relevant court
    BOUNCE_OUT = "bounce_out"       # landed outside the relevant court
    NET_FAULT = "net_fault"         # ball failed to clear / hit the net
    DOUBLE_BOUNCE = "double_bounce"  # two bounces on one side without a return
    OUT_OF_PLAY = "out_of_play"     # ball left the tracking volume / rally died
    LINE_CALL = "line_call"         # ball landed within LINE_MARGIN of a line
    RALLY_END = "rally_end"         # synthetic: emitted by the rule engine
    POINT = "point"                 # synthetic: point awarded
    GAME = "game"                   # synthetic: game/set completed
    MATCH = "match"                 # synthetic: match completed


class Side(str, Enum):
    """Physical halves of the court in canonical court coordinates.

    ``A`` owns the region x < length/2, ``B`` owns x > length/2.  Which
    *player* stands there changes on change-of-ends, so the analizer maps
    physical side -> logical player with an explicit ``SideMap``.
    """

    A = "A"
    B = "B"

    @property
    def other(self) -> "Side":
        return Side.B if self is Side.A else Side.A


class Outcome(str, Enum):
    """Who won the rally, from the perspective of the *server*."""

    SERVER = "server"
    RECEIVER = "receiver"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Perception outputs
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One object detection in image space (pixels)."""

    xy: Point
    conf: float = 1.0
    label: str = "ball"
    box: Optional[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2


@dataclass
class TrackPoint:
    """A ball observation at a given frame, in image and (optional) court space."""

    frame: int
    t: float
    xy: Point                       # image pixels
    court_xy: Optional[Point] = None  # court metres, None if outside homography
    conf: float = 1.0
    interpolated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "t": round(self.t, 4),
            "xy": [round(float(v), 2) for v in self.xy],
            "court_xy": None if self.court_xy is None else [round(float(v), 3) for v in self.court_xy],
            "conf": round(float(self.conf), 3),
            "interpolated": self.interpolated,
        }


@dataclass
class PlayerBox:
    frame: int
    t: float
    side: Side
    box: Tuple[float, float, float, float]
    foot_img: Point
    foot_court: Optional[Point] = None
    conf: float = 1.0


# --------------------------------------------------------------------------- #
# Events & scoring
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """A semantic event with enough provenance to be used as evidence."""

    type: EventType
    frame: int
    t: float
    confidence: float = 1.0
    side: Optional[Side] = None            # side the event happened *in*
    court_xy: Optional[Point] = None       # landing / contact point in metres
    img_xy: Optional[Point] = None         # same point in pixels
    in_bounds: Optional[bool] = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type.value,
            "frame": self.frame,
            "t": round(float(self.t), 4),
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }
        if self.side is not None:
            d["side"] = self.side.value
        if self.court_xy is not None:
            d["court_xy"] = [round(float(v), 3) for v in self.court_xy]
        if self.img_xy is not None:
            d["img_xy"] = [round(float(v), 1) for v in self.img_xy]
        if self.in_bounds is not None:
            d["in_bounds"] = bool(self.in_bounds)
        if self.meta:
            d["meta"] = _jsonable(self.meta)
        return d


@dataclass
class ScoreState:
    """Immutable snapshot of the scoreboard at a point in time."""

    sport: Sport
    t: float
    frame: int
    serving_player: str
    server_side: Side
    score: Dict[str, Any]              # sport specific (see sports/*.py)
    display: str                       # e.g. "11-9  |  G2 5-3"
    leader: Optional[str] = None
    winner: Optional[str] = None
    match_over: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sport": self.sport.value,
            "t": round(float(self.t), 4),
            "frame": self.frame,
            "serving_player": self.serving_player,
            "server_side": self.server_side.value,
            "score": _jsonable(self.score),
            "display": self.display,
            "leader": self.leader,
            "winner": self.winner,
            "match_over": self.match_over,
        }


@dataclass
class PointRecord:
    """One completed rally and the score it produced (the unit of evidence)."""

    index: int
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int
    winner: str
    loser: str
    outcome: Outcome
    cause: str                      # e.g. "out_of_bounds", "double_bounce"
    score_before: Optional[ScoreState] = None
    score_after: Optional[ScoreState] = None
    confidence: float = 1.0
    needs_review: bool = False
    events: List[Event] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # relative proof-image paths

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "t_start": round(self.t_start, 4),
            "t_end": round(self.t_end, 4),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "winner": self.winner,
            "loser": self.loser,
            "outcome": self.outcome.value,
            "cause": self.cause,
            "confidence": round(float(self.confidence), 3),
            "needs_review": bool(self.needs_review),
            "score_before": None if self.score_before is None else self.score_before.as_dict(),
            "score_after": None if self.score_after is None else self.score_after.as_dict(),
            "events": [e.as_dict() for e in self.events],
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays / enums into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if dataclasses.is_dataclass(obj) and hasattr(obj, "as_dict"):
        return obj.as_dict()
    return obj


def events_to_json(events: Sequence[Event]) -> List[Dict[str, Any]]:
    return [e.as_dict() for e in events]
