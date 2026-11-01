"""Tennis: ITF Rules of Tennis (15/30/40, deuce/advantage, sets, tie-break).

Rule references implemented here
--------------------------------
* Rule 5  – points are 0 ("love"), 15, 30, 40; game needs 4 points and a
             2-point lead; at 40-40 ("deuce") a 2-point lead is needed
             ("advantage"), unless ``no_ad`` (college / FAST4) is enabled.
* Rule 5  – a set is won with 6 games and a 2-game lead.
* Rule 5  – at 6-6 a tie-break is played (7 points, 2-point lead).
* Rule 14 – the serve must land in the diagonally opposite service box.
* Rule 17/18 – the ball may not land outside the court lines; the singles
             sidelines apply in singles and the alley is out on a serve.
* Rule 10 – sides change after the 1st, 3rd and every subsequent odd game
             of each set, and at the end of each set.
* Serving side: the server stands behind the baseline, right of the centre
             mark when the total number of points in the game is even.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.types import Point, ScoreState, Side, Sport
from ..court.model import CourtModel, CourtSpec, TENNIS
from .base import LandingContext, LandingVerdict, MatchScorer, SportRules, zone_of

POINT_NAMES = ("0", "15", "30", "40")


class TennisMatch(MatchScorer):
    """ITF scoring with sets, tie-breaks and optional no-ad scoring."""

    sport = Sport.TENNIS
    court_spec = TENNIS

    def __init__(self, players: Sequence[str], mode: str = "singles", best_of: int = 3,
                 first_server: Optional[str] = None, no_ad: bool = False,
                 tiebreak_to: int = 7, final_set_tiebreak_to: Optional[int] = 7,
                 tiebreak_at: int = 6, change_ends: bool = True):
        super().__init__(players, mode, best_of, first_server)
        self.no_ad = no_ad
        self.tiebreak_to = tiebreak_to
        self.final_set_tiebreak_to = final_set_tiebreak_to
        self.tiebreak_at = tiebreak_at
        self.change_ends = change_ends

        self.points: Dict[str, int] = {p: 0 for p in self.players}
        self.games: Dict[str, int] = {p: 0 for p in self.players}
        self.sets_won: Dict[str, int] = {p: 0 for p in self.players}
        self.set_scores: List[Tuple[int, int]] = []
        self.in_tiebreak = False
        self.tiebreak_points_played = 0
        self.tiebreak_first_server: Optional[str] = None
        self.total_games = 0

    # -- serve geometry -----------------------------------------------------
    def serve_target(self) -> Tuple[Side, bool]:
        """(side to serve into, is that box in the low-y band?)."""
        even_points = (self.points[self.players[0]] + self.points[self.players[1]]) % 2 == 0
        right_low_band = self.server_side is Side.B
        server_low = right_low_band if even_points else (not right_low_band)
        return self.server_side.other, (not server_low)

    def serve_box(self, court: CourtModel, mode: Optional[str] = None) -> List[Point]:
        side, target_low = self.serve_target()
        return court.service_box_for_band(side, target_low, mode or self.mode)

    # -- point arithmetic ---------------------------------------------------
    def _leader_points(self) -> Optional[str]:
        a, b = self.players[0], self.players[1]
        if self.points[a] == self.points[b]:
            return None
        return a if self.points[a] > self.points[b] else b

    def _game_won_by(self, name: str) -> bool:
        mine = self.points[name]
        other = self.points[self.opponent(name)]
        if self.in_tiebreak:
            target = self._current_tiebreak_target()
            return mine >= target and (mine - other) >= 2
        if self.no_ad and mine >= 4 and other == 3:
            return True                     # deciding point at deuce
        return mine >= 4 and (mine - other) >= 2

    def _current_tiebreak_target(self) -> int:
        deciding = (self.sets_won[self.players[0]] + self.sets_won[self.players[1]]
                    == self.best_of - 1)
        if deciding and self.final_set_tiebreak_to:
            return self.final_set_tiebreak_to
        return self.tiebreak_to

    # -- serving rotation ---------------------------------------------------
    def _server_for_point(self, point_number: int) -> str:
        """Who serves point ``point_number`` (1-indexed) of the tie-break.

        Rule 5: the player due to serve serves the first point; the opponent
        serves the next two; thereafter serve alternates every two points.
        """
        first = self.tiebreak_first_server or self.server
        if point_number <= 1:
            return first
        block = (point_number - 2) // 2
        return self.opponent(first) if block % 2 == 0 else first

    # -- scoring ------------------------------------------------------------
    def award_point(self, winner: str, t: float = 0.0, frame: int = 0,
                    cause: str = "") -> ScoreState:
        if self.match_over:
            return self.snapshot(t, frame)
        if winner not in self.points:
            raise ValueError(f"unknown player {winner!r}")

        self.points[winner] += 1
        self.points_played += 1

        if self.in_tiebreak:
            self.tiebreak_points_played += 1
            self.server = self._server_for_point(self.tiebreak_points_played + 1)
            # change ends every 6 points in a tie-break
            if self.change_ends and self.tiebreak_points_played % 6 == 0:
                self.flip_sides()
        if self._game_won_by(winner):
            self._close_game(winner)
        return self.snapshot(t, frame)

    def _close_game(self, winner: str) -> None:
        was_tiebreak = self.in_tiebreak
        self.points = {p: 0 for p in self.players}
        self.games[winner] += 1
        self.total_games += 1

        set_over = False
        if was_tiebreak:
            set_over = True                    # tie-break decided the set
        else:
            mine, other = self.games[winner], self.games[self.opponent(winner)]
            if mine >= self.tiebreak_at and (mine - other) >= 2:
                set_over = True
            elif mine == self.tiebreak_at and other == self.tiebreak_at:
                self.in_tiebreak = True
                self.tiebreak_points_played = 0
                self.tiebreak_first_server = self.opponent(self.server)
                self.server = self.tiebreak_first_server
                if self.change_ends:
                    self.flip_sides()
                return

        # serve alternates every game
        self.server = self.opponent(self.server)
        if self.change_ends and self.total_games % 2 == 1:
            self.flip_sides()

        if set_over:
            self._close_set(winner)

    def _close_set(self, winner: str) -> None:
        self.set_scores.append((self.games[self.players[0]], self.games[self.players[1]]))
        self.sets_won[winner] += 1
        self.games = {p: 0 for p in self.players}
        self.in_tiebreak = False
        self.tiebreak_points_played = 0
        self.tiebreak_first_server = None
        if self.change_ends:
            self.flip_sides()
        if self.sets_won[winner] >= self.sets_to_win:
            self.match_over = True
            self.winner = winner

    # -- display ------------------------------------------------------------
    def points_display(self) -> str:
        a, b = self.players[0], self.players[1]
        pa, pb = self.points[a], self.points[b]
        if self.in_tiebreak:
            return f"{pa}-{pb} (tie-break)"
        if pa >= 3 and pb >= 3:
            if pa == pb:
                return "40-40 (deuce)" if not self.no_ad else "40-40 (deciding point)"
            return "AD-40" if pa > pb else "40-AD"
        return f"{POINT_NAMES[min(pa, 3)]}-{POINT_NAMES[min(pb, 3)]}"

    def display(self) -> str:
        sets = " ".join(f"{x}-{y}" for x, y in self.set_scores)
        cur = f"{self.games[self.players[0]]}-{self.games[self.players[1]]}"
        pts = self.points_display()
        body = " ".join(part for part in (sets, cur, pts) if part)
        if self.match_over:
            return f"{body}  (match won by {self.winner})"
        return body

    def score_dict(self) -> Dict[str, Any]:
        return {
            "points": dict(self.points),
            "games": dict(self.games),
            "sets_won": dict(self.sets_won),
            "set_scores": [list(s) for s in self.set_scores],
            "in_tiebreak": self.in_tiebreak,
            "points_display": self.points_display(),
            "server": self.server,
            "server_side": self.server_side.value,
            "mode": self.mode,
            "no_ad": self.no_ad,
        }

    @property
    def leader(self) -> Optional[str]:
        if self.match_over:
            return self.winner
        a, b = self.players[0], self.players[1]
        if self.sets_won[a] != self.sets_won[b]:
            return a if self.sets_won[a] > self.sets_won[b] else b
        if self.games[a] != self.games[b]:
            return a if self.games[a] > self.games[b] else b
        return self._leader_points()


class TennisRules(SportRules):
    """In/out adjudication for tennis ball landings."""

    sport = Sport.TENNIS
    court_spec: CourtSpec = TENNIS

    def new_match(self, players: Sequence[str], mode: str = "singles",
                  best_of: int = 3, first_server: Optional[str] = None,
                  **kwargs: Any) -> TennisMatch:
        return TennisMatch(players, mode=mode, best_of=best_of,
                           first_server=first_server, **kwargs)

    def build_serve_box(self, court: CourtModel, scorer: MatchScorer,
                        mode: str) -> List[Point]:
        return court.service_box_for_band(*scorer.serve_target(), mode=mode)

    def judge_landing(self, court: CourtModel, pt: Point,
                      ctx: LandingContext) -> LandingVerdict:
        margin = max(0.0, ctx.line_margin_m)

        # A ball that comes down on the striker's own half has not crossed the
        # net (a shank).  A ball beyond the lines on that half is simply OUT, so
        # the own-court fault only applies inside the playing area.
        if (ctx.striking_side is not None and court.half_of(pt) is ctx.striking_side
                and court.in_play(pt, ctx.mode, 0.0)):
            return LandingVerdict(in_bounds=False, zone="own_court",
                                  target_polygon=court.play_polygon(ctx.mode),
                                  reason="ball landed on the striker's own side of the net")

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
                reason="serve in" if inside else "serve out (wrong service box)",
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
            reason="ball landed in" if inside else "ball landed out",
        )

    def describe_zone(self, pt: Point) -> str:
        return zone_of(self.court_spec, pt)
