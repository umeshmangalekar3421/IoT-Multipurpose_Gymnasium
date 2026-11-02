"""Sport rule interfaces: landing judgement + scoring state machines.

Two responsibilities are separated on purpose:

``SportRules``
    *Geometry-only* adjudication: given a landing point in court metres and the
    context (serve or rally, singles or doubles, which box), decide IN/OUT and
    whether it was a line call.  Pure function of the point -> easy to unit test
    and to replay on an edge device.

``MatchScorer``
    *Bookkeeping*: official scoring for the sport (badminton rally points,
    tennis 15/30/40/tiebreak/sets), serve rotation, change of ends, winner.

Neither class knows anything about video, which keeps them portable and makes
the rules testable against the official rule books.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.types import Point, ScoreState, Side, Sport
from ..court.model import CourtModel, CourtSpec


# --------------------------------------------------------------------------- #
# Landing adjudication
# --------------------------------------------------------------------------- #
@dataclass
class LandingContext:
    """Everything the adjudicator needs to know about a landing."""

    is_serve: bool = False
    mode: str = "singles"                      # singles | doubles
    receiving_side: Optional[Side] = None
    # polygon (court metres) the ball had to land inside for a legal serve
    expected_service_box: Optional[List[Point]] = None
    line_margin_m: float = 0.03                # ball-radius tolerance
    # side the ball was struck from (used for "landed in own court" faults)
    striking_side: Optional[Side] = None


@dataclass
class LandingVerdict:
    in_bounds: bool
    zone: str                     # "service_box" | "play_area" | "own_court" | "out"
    line_call: bool = False
    distance_to_line_m: float = float("inf")
    target_polygon: List[Point] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "in_bounds": self.in_bounds,
            "zone": self.zone,
            "line_call": self.line_call,
            "distance_to_line_m": round(self.distance_to_line_m, 3),
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Scoring interface
# --------------------------------------------------------------------------- #
class MatchScorer(ABC):
    """Official scoring bookkeeping for one sport."""

    sport: Sport

    def __init__(self, players: Sequence[str], mode: str = "singles",
                 best_of: int = 3, first_server: Optional[str] = None):
        if len(players) < 2:
            raise ValueError("need at least 2 players/teams")
        self.players: List[str] = list(players)
        self.mode = mode
        self.best_of = best_of
        self.sets_to_win = best_of // 2 + 1
        self.server: str = first_server or self.players[0]
        if self.server not in self.players:
            raise ValueError(f"first_server {self.server!r} not in players")
        self.side_of: Dict[str, Side] = {self.players[0]: Side.A,
                                         self.players[-1]: Side.B}
        self.match_over = False
        self.winner: Optional[str] = None
        self.points_played = 0

    # -- helpers ------------------------------------------------------------
    def opponent(self, name: str) -> str:
        return self.players[1] if name == self.players[0] else self.players[0]

    def player_at(self, side: Side) -> str:
        """Logical player occupying a physical side of the court."""
        for name, s in self.side_of.items():
            if s is side:
                return name
        return self.players[0]

    @property
    def server_side(self) -> Side:
        return self.side_of[self.server]

    def flip_sides(self) -> None:
        for name in list(self.side_of):
            self.side_of[name] = self.side_of[name].other

    @property
    def leader(self) -> Optional[str]:
        raise NotImplementedError

    # -- API ----------------------------------------------------------------
    @abstractmethod
    def award_point(self, winner: str, t: float = 0.0, frame: int = 0,
                    cause: str = "") -> ScoreState:
        """Credit one rally to ``winner`` and return the new scoreboard."""

    @abstractmethod
    def display(self) -> str:
        """Compact human scoreboard, e.g. ``21-19 | G2 11-9``."""

    @abstractmethod
    def score_dict(self) -> Dict[str, Any]:
        """Machine readable score (goes into the JSON timeline)."""

    @abstractmethod
    def serve_target(self) -> Tuple[Side, bool]:
        """(physical side to serve into, whether the target box is the y<mid band)."""

    def snapshot(self, t: float = 0.0, frame: int = 0) -> ScoreState:
        side, _ = self.serve_target()
        return ScoreState(
            sport=self.sport,
            t=t,
            frame=frame,
            serving_player=self.server,
            server_side=side,
            score=self.score_dict(),
            display=self.display(),
            leader=self.leader,
            winner=self.winner,
            match_over=self.match_over,
        )


# --------------------------------------------------------------------------- #
# Rules interface
# --------------------------------------------------------------------------- #
class SportRules(ABC):
    """Geometry + rule interpretation for one sport."""

    sport: Sport
    court_spec: CourtSpec
    doubles_serve_uses_long_line: bool = True

    @abstractmethod
    def new_match(self, players: Sequence[str], mode: str = "singles",
                  best_of: int = 3, first_server: Optional[str] = None) -> MatchScorer:
        ...

    @abstractmethod
    def judge_landing(self, court: CourtModel, pt: Point,
                      ctx: LandingContext) -> LandingVerdict:
        """Adjudicate one landing point (court metres)."""

    @abstractmethod
    def build_serve_box(self, court: CourtModel, scorer: MatchScorer,
                        mode: str) -> List[Point]:
        """Service box implied by the current scoreboard (used for serving faults)."""

    def make_landing_context(self, court: CourtModel, scorer: MatchScorer,
                             mode: str, is_serve: bool,
                             striking_side: Optional[Side] = None,
                             line_margin_m: float = 0.03) -> LandingContext:
        side, low_band = scorer.serve_target()
        return LandingContext(
            is_serve=is_serve,
            mode=mode,
            receiving_side=side,
            expected_service_box=self.build_serve_box(court, scorer, mode) if is_serve else None,
            line_margin_m=line_margin_m,
            striking_side=striking_side,
        )


def zone_of(spec: CourtSpec, pt: Point) -> str:
    """Human label for a court location (for evidence overlays)."""
    x, y = pt
    half = "A" if x < spec.net_x else "B"
    depth = "front" if abs(x - spec.net_x) <= spec.service_line_from_net else "back"
    band = "left" if y < spec.mid_y else "right"
    return f"{half}_{depth}_{band}"
