"""Badminton: BWF Laws of Badminton (rally point scoring, 21 x best of 3).

Rule references implemented here
--------------------------------
* Law 7.2  – rally point scoring, a point is won on every rally.
* Law 7.4  – game to 21 points; at 20-all, 2-point lead; at 29-all, the next
             point wins (30-point cap).
* Law 7.3  – match is best of 3 games by default.
* Law 9.2  – service courts are diagonal; even score => right service court.
* Law 9.3  – in doubles the serve must land before the doubles long service
             line (5.94 m from the net), in singles before the back boundary.
* Law 11   – change of ends after each game and, in the deciding game, when a
             side reaches 11 points.
* Law 13   – shuttle landing outside the boundary = fault by the striker.

The engine decides *who* won a rally from a :class:`LandingVerdict` sequence
produced by the perception layer; the rules below only encode the bookkeeping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.types import Point, ScoreState, Side, Sport
from ..court.model import BADMINTON, CourtModel, CourtSpec
from .base import (
    LandingContext,
    LandingVerdict,
    MatchScorer,
    SportRules,
    zone_of,
)


class BadmintonMatch(MatchScorer):
    """BWF rally-point scoring with serve-court parity and change of ends."""

    sport = Sport.BADMINTON

    def __init__(self, players: Sequence[str], mode: str = "singles", best_of: int = 3,
                 first_server: Optional[str] = None, points_to: int = 21,
                 points_cap: int = 30, win_by: int = 2,
                 change_ends: bool = True, deciding_game_interval: int = 11):
        super().__init__(players, mode, best_of, first_server)
        self.points_to = points_to
        self.points_cap = points_cap
        self.win_by = win_by
        self.change_ends = change_ends
        self.deciding_game_interval = deciding_game_interval

        self.points: Dict[str, int] = {p: 0 for p in self.players}
        self.games: Dict[str, int] = {p: 0 for p in self.players}
        self.game_scores: List[Tuple[int, int]] = []   # finished games, (p1, p2)
        self.current_game = 1
        self.ends_swapped_this_game = False

    # -- serve geometry -----------------------------------------------------
    def service_court_low_band(self) -> bool:
        """True if the server must stand in the ``y < width/2`` service court.

        Convention: each player's *right* service court is on their own
        right-hand side when facing the net.  For side A (x < net, facing +x)
        that is ``y > width/2``; for side B it is ``y < width/2``.
        """
        even_score = self.points[self.server] % 2 == 0
        right_low_band = self.server_side is Side.B
        return right_low_band if even_score else (not right_low_band)

    def serve_target(self) -> Tuple[Side, bool]:
        """(side to serve into, is the target box in the low y band?)"""
        server_low = self.service_court_low_band()
        # cross-court diagonal: target band is the opposite band
        return self.server_side.other, (not server_low)

    def serve_box(self, court: CourtModel) -> List[Point]:
        side, target_low = self.serve_target()
        return court.service_box_for_band(side, target_low, self.mode)

    # -- scoring ------------------------------------------------------------
    def _game_won_by(self, name: str) -> bool:
        mine = self.points[name]
        other = self.points[self.opponent(name)]
        if mine >= self.points_cap:
            return True
        return mine >= self.points_to and (mine - other) >= self.win_by

    def award_point(self, winner: str, t: float = 0.0, frame: int = 0,
                    cause: str = "") -> ScoreState:
        if self.match_over:
            return self.snapshot(t, frame)
        if winner not in self.points:
            raise ValueError(f"unknown player {winner!r}")

        self.points[winner] += 1
        self.points_played += 1

        # winner of the rally serves next
        self.server = winner

        if self._game_won_by(winner):
            self._close_game(winner)

        # deciding-game interval change of ends
        if (self.change_ends and not self.match_over
                and self.current_game == self.best_of
                and not self.ends_swapped_this_game
                and max(self.points.values()) >= self.deciding_game_interval):
            self.flip_sides()
            self.ends_swapped_this_game = True

        return self.snapshot(t, frame)

    def _close_game(self, winner: str) -> None:
        self.games[winner] += 1
        self.game_scores.append((self.points[self.players[0]],
                                 self.points[self.players[1]]))
        self.points = {p: 0 for p in self.players}
        self.current_game += 1
        self.ends_swapped_this_game = False
        if self.change_ends:
            self.flip_sides()
        # winner of the game serves first in the next game
        self.server = winner
        if self.games[winner] >= self.sets_to_win:
            self.match_over = True
            self.winner = winner

    # -- display ------------------------------------------------------------
    def display(self) -> str:
        parts = [f"{a}-{b}" for a, b in self.game_scores]
        cur = f"{self.points[self.players[0]]}-{self.points[self.players[1]]}"
        if self.match_over:
            return " ".join(parts) + f"  (match won by {self.winner})"
        return " ".join(parts + [cur]) if parts else cur

    def score_dict(self) -> Dict[str, Any]:
        return {
            "points": dict(self.points),
            "games": dict(self.games),
            "game_scores": [list(g) for g in self.game_scores],
            "game_number": self.current_game,
            "server": self.server,
            "server_side": self.server_side.value,
            "points_to": self.points_to,
            "mode": self.mode,
        }

    @property
    def leader(self) -> Optional[str]:
        if self.match_over:
            return self.winner
        a, b = self.players[0], self.players[1]
        if self.games[a] != self.games[b]:
            return a if self.games[a] > self.games[b] else b
        if self.points[a] != self.points[b]:
            return a if self.points[a] > self.points[b] else b
        return None


class BadmintonRules(SportRules):
    """In/out adjudication for badminton (shuttle landing point)."""

    sport = Sport.BADMINTON
    court_spec: CourtSpec = BADMINTON

    def new_match(self, players: Sequence[str], mode: str = "singles",
                  best_of: int = 3, first_server: Optional[str] = None) -> BadmintonMatch:
        return BadmintonMatch(players, mode=mode, best_of=best_of,
                              first_server=first_server)

    def build_serve_box(self, court: CourtModel, scorer: MatchScorer,
                        mode: str) -> List[Point]:
        if isinstance(scorer, BadmintonMatch):
            return scorer.serve_box(court)
        side, low = scorer.serve_target()
        return court.service_box_for_band(side, low, mode)

    def judge_landing(self, court: CourtModel, pt: Point,
                      ctx: LandingContext) -> LandingVerdict:
        margin = max(0.0, ctx.line_margin_m)
        spec = self.court_spec

        # landing back in the striker's own half is always a fault
        # A ball that comes down on the striker's own half has not crossed the
        # net (a shank).  A ball beyond the lines on that half is simply OUT, so
        # the own-court fault only applies inside the playing area.
        if (ctx.striking_side is not None and court.half_of(pt) is ctx.striking_side
                and court.in_play(pt, ctx.mode, 0.0)):
            return LandingVerdict(
                in_bounds=False, zone="own_court", line_call=False,
                target_polygon=court.play_polygon(ctx.mode),
                reason="shuttle landed in the striker's own court",
            )

        if ctx.is_serve and ctx.expected_service_box is not None:
            box = ctx.expected_service_box
            inside = court.in_service_box(pt, box, margin)
            dist = court.distance_to_nearest_line(pt, ctx.mode)
            return LandingVerdict(
                in_bounds=inside,
                zone="service_box" if inside else "out",
                line_call=inside and dist <= (margin + 0.02),
                distance_to_line_m=dist,
                target_polygon=box,
                reason=("serve landed in the correct service court" if inside
                        else "serve did not land in the correct service court"),
            )

        poly = court.play_polygon(ctx.mode)
        inside = court.in_play(pt, ctx.mode, margin)
        dist = court.distance_to_nearest_line(pt, ctx.mode)
        return LandingVerdict(
            in_bounds=inside,
            zone="play_area" if inside else "out",
            line_call=inside and dist <= (margin + 0.02),
            distance_to_line_m=dist,
            target_polygon=poly,
            reason=("shuttle landed in the court" if inside else "shuttle landed out of bounds"),
        )

    def describe_zone(self, pt: Point) -> str:
        return zone_of(self.court_spec, pt)
