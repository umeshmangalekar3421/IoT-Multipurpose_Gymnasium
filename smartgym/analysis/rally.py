"""Rally interpretation: physical events -> points, per the official rule book.

The engine walks the event stream chronologically and keeps a small state
machine per rally.  Sport-specific knowledge lives in two small subclasses so
the shared machinery (serve detection, pending decisions, score bookkeeping,
evidence assembly) is written once.

Design notes worth defending in a viva
--------------------------------------
*Pending decisions* (badminton).  A shuttle that is returned low over the net
can look like a floor contact.  Rather than awarding a point immediately, an
in-bounds landing is held for ``confirm_frames``; if a racket contact appears
inside that window the landing is discarded as a false positive.  Out-of-bounds
landings are never re-opened — out is out.

*Two-serve model* (tennis).  A first-serve fault is a let, not a point.  The
engine counts faults, forces a second serve and only then awards the point.

*Provenance*.  Every point carries the events that produced it plus the
scoreboard before and after, which is exactly what the evidence recorder draws
onto the proof image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.types import Event, EventType, Outcome, PointRecord, ScoreState, Side
from ..court.model import CourtModel
from ..sports.base import LandingVerdict, MatchScorer, SportRules


@dataclass
class PendingDecision:
    """A point that is logically decided but not yet committed."""

    winner: str
    cause: str
    frame: int
    t: float
    confidence: float
    needs_review: bool
    reopen_if_hit: bool
    events: List[Event] = field(default_factory=list)


class RallyEngine:
    """Base interpreter shared by all sports."""

    sport_name = "generic"

    def __init__(self, rules: SportRules, scorer: MatchScorer, court: CourtModel,
                 mode: str = "singles", cfg: Optional[dict] = None):
        self.rules = rules
        self.scorer = scorer
        self.court = court
        self.mode = mode
        cfg = cfg or {}
        ev = cfg.get("events", {})
        self.line_margin_m = float(ev.get("line_margin_m", 0.03))
        self.rally_gap_frames = int(ev.get("rally_gap_frames", 120))
        self.confirm_frames = int(ev.get("confirm_landing_frames", 10))
        self.on_track_loss = str(ev.get("on_track_loss", "opponent"))

        self.points: List[PointRecord] = []
        self.timeline: List[ScoreState] = []
        self.warnings: List[str] = []

        self._reset_rally()

    # -- rally state --------------------------------------------------------
    def _reset_rally(self) -> None:
        self.in_rally = False
        self.rally_events: List[Event] = []
        self.striker_side: Optional[Side] = None
        self.serve_pending = True
        self.last_frame = -10 ** 9
        self.pending: Optional[PendingDecision] = None
        self._reset_sport_state()

    def _reset_sport_state(self) -> None:
        pass

    # -- public API ---------------------------------------------------------
    def process(self, events: Sequence[Event]) -> List[PointRecord]:
        for ev in sorted(events, key=lambda e: (e.frame, e.t)):
            self._feed(ev)
        self.finalize_rally()
        return self.points

    def finalize_rally(self) -> Optional[PointRecord]:
        """Commit a dangling decision / close the rally at end of stream."""
        if self.pending is not None:
            rec = self._commit_pending()
            self._reset_rally()
            return rec
        if self.in_rally:
            self._close_rally_without_point()
        return None

    # -- event dispatch -----------------------------------------------------
    def _feed(self, ev: Event) -> None:
        """Register one event, then let the sport interpret it.

        Order matters: the event is appended to ``rally_events`` *before* the
        handler runs, so any decision snapshotted by the handler already
        contains the evidence that caused it.
        """
        # A decision whose confirmation window has elapsed is committed *before*
        # the new event is interpreted, so that the event that arrives after the
        # window belongs to the next rally instead of the finished one.
        if self.pending is not None and ev.frame > self.pending.frame + self.confirm_frames:
            self._commit_pending()
            self._reset_rally()

        gap = ev.frame - self.last_frame
        new_rally = (not self.in_rally) or (gap > self.rally_gap_frames)

        if new_rally and ev.type in (EventType.SERVE, EventType.SHOT, EventType.LANDING):
            self._start_rally(ev)
            if ev.type is EventType.LANDING:
                # a landing with no visible stroke: attribute it to the server
                self.striker_side = self.scorer.server_side
                ev.meta["synthetic_rally_start"] = True

        self.rally_events.append(ev)
        self.last_frame = ev.frame

        if ev.type in (EventType.SERVE, EventType.SHOT):
            if self.expecting_serve:
                # the first stroke of a rally — and, in tennis, the stroke after
                # a fault — is always a serve
                ev.type = EventType.SERVE
                ev.reason = "serve"
                self.expecting_serve = False
                self.serve_pending = True
                self.striker_side = ev.side or self.scorer.server_side
                self._on_serve_struck(ev)
            else:
                self.serve_pending = False
                if ev.side is not None:
                    self.striker_side = ev.side
                self._on_hit(ev)
        elif ev.type is EventType.LANDING:
            self._on_landing(ev)
        elif ev.type is EventType.NET_CONTACT:
            self._on_net(ev)
        elif ev.type is EventType.OUT_OF_PLAY:
            self._on_out_of_play(ev)


    # -- rally lifecycle ----------------------------------------------------
    def _start_rally(self, ev: Event) -> None:
        self.in_rally = True
        self.rally_events = []
        self.pending = None
        self.serve_pending = True
        self.expecting_serve = True
        self.striker_side = None
        self._reset_sport_state()
        self._rally_start_frame = ev.frame
        self._rally_start_t = ev.t

    def _close_rally_without_point(self) -> None:
        if self.rally_events:
            self.warnings.append(
                f"rally at frame {getattr(self, '_rally_start_frame', -1)} "
                "ended without a decision")
        self._reset_rally()

    def _on_serve_struck(self, ev: Event) -> None:
        """Hook: a serve has just been struck."""

    def _on_hit(self, ev: Event) -> None:
        # a racket contact inside the confirmation window disproves the
        # landing that was about to be scored -> treat it as a return
        if self.pending is not None and self.pending.reopen_if_hit:
            if ev.frame - self.pending.frame <= self.confirm_frames:
                ev.meta["reopened_false_bounce"] = True
                self.pending = None
        if ev.side is not None:
            self.striker_side = ev.side
        self._after_hit(ev)

    def _after_hit(self, ev: Event) -> None:
        """Hook: common post-hit bookkeeping."""

    def _on_landing(self, ev: Event) -> None:
        self._last_landing_frame = ev.frame
        # A floor contact that the ball demonstrably survives (it keeps flying)
        # is not the end of the rally — it is just a bounce in the middle of it.
        # Keeping the rally alive here is what stops one rally being paid out as
        # two points.
        if ev.meta.get("terminal") is False:
            if ev.court_xy is not None and self.court.registered:
                landing_side = self.court.half_of(ev.court_xy)
                # the player on that side must now return the ball
                self.striker_side = landing_side
                self.serve_pending = False
            self._after_nonterminal_landing(ev)
            return
        self._handle_landing(ev)

    def _after_nonterminal_landing(self, ev: Event) -> None:
        """Hook: a bounce that did not end the rally (used by tennis)."""

    def _handle_landing(self, ev: Event, projected: bool = False) -> None:
        raise NotImplementedError

    def _on_net(self, ev: Event) -> None:
        if not self.in_rally:
            return
        striker = self._player_side()
        if striker is None:
            return
        self.rally_events.append(Event(
            type=EventType.NET_FAULT, frame=ev.frame, t=ev.t, confidence=ev.confidence,
            side=self.striker_side, img_xy=ev.img_xy, court_xy=ev.court_xy,
            reason="ball failed to clear the net"))
        self._decide(self.scorer.opponent(striker), cause="net_fault",
                     frame=ev.frame, t=ev.t, confidence=ev.confidence,
                     reopen_if_hit=False)

    def _on_out_of_play(self, ev: Event) -> None:
        if not self.in_rally:
            return
        if ev.court_xy is not None and self.court.registered:
            self.rally_events.append(ev)
            self._handle_landing(ev, projected=True)
            return
        striker = self._player_side()
        if striker is None:
            return
        policy = self.on_track_loss
        winner = (self.scorer.opponent(striker) if policy == "opponent"
                  else striker if policy == "striker" else None)
        if winner is None:
            self._close_rally_without_point()
            return
        self._decide(winner, cause="ball_out_of_view", frame=ev.frame, t=ev.t,
                     confidence=0.4, needs_review=True)

    # -- decisions ----------------------------------------------------------
    def _player_side(self) -> Optional[str]:
        if self.striker_side is None:
            return None
        return self.scorer.player_at(self.striker_side)

    def _attribute_striker(self, ev: Event, landing_side: Optional[Side]) -> Optional[str]:
        """Who struck the ball that produced this landing?

        The physics is unambiguous: a ball that lands on side ``Y`` was last
        struck from side ``X = opposite(Y)`` (a lift or a shot crosses the net).
        Deriving the striker from the landing point is therefore far more
        reliable than relying on the racket-contact event, which needs the ball
        to be visible exactly at the moment of contact — often it is not.

        Limitation: a ball that is shanked into the striker's *own* court is
        therefore attributed to the opponent. Such faults are rare, and the
        detector's strike evidence is not trustworthy enough to override the
        geometry — a wrong override costs far more points than a missed shank.
        """
        if landing_side is None:
            return self._player_side()
        return self.scorer.player_at(landing_side.other)

    def _verdict(self, ev: Event, is_serve: bool,
                 striking_side: Optional[Side] = None) -> Optional[LandingVerdict]:
        if ev.court_xy is None or not self.court.registered:
            return None
        ctx = self.rules.make_landing_context(
            self.court, self.scorer, self.mode, is_serve=is_serve,
            striking_side=striking_side or self.striker_side,
            line_margin_m=self.line_margin_m,
        )
        return self.rules.judge_landing(self.court, ev.court_xy, ctx)

    def _adjudicated_events(self, ev: Event, verdict: LandingVerdict) -> List[Event]:
        out = [Event(
            type=EventType.BOUNCE_IN if verdict.in_bounds else EventType.BOUNCE_OUT,
            frame=ev.frame, t=ev.t, confidence=ev.confidence,
            side=self.court.half_of(ev.court_xy) if ev.court_xy else ev.side,
            court_xy=ev.court_xy, img_xy=ev.img_xy, in_bounds=verdict.in_bounds,
            reason=verdict.reason,
            meta={"zone": verdict.zone,
                  "distance_to_line_m": round(verdict.distance_to_line_m, 3)},
        )]
        if verdict.line_call:
            out.append(Event(
                type=EventType.LINE_CALL, frame=ev.frame, t=ev.t,
                confidence=ev.confidence,
                side=self.court.half_of(ev.court_xy) if ev.court_xy else ev.side,
                court_xy=ev.court_xy, img_xy=ev.img_xy,
                reason=f"landed {verdict.distance_to_line_m * 100:.1f} cm from a line",
            ))
        return out

    def _decide(self, winner: str, cause: str, frame: int, t: float,
                confidence: float, event: Optional[Event] = None,
                needs_review: bool = False, reopen_if_hit: bool = False) -> None:
        evs = list(self.rally_events)
        if event is not None:
            evs.append(event)
        self.pending = PendingDecision(
            winner=winner, cause=cause, frame=frame, t=t,
            confidence=confidence, needs_review=needs_review,
            reopen_if_hit=reopen_if_hit, events=evs,
        )

    def _commit_pending(self) -> Optional[PointRecord]:
        p = self.pending
        if p is None:
            return None
        self.pending = None
        loser = self.scorer.opponent(p.winner)
        before = self.scorer.snapshot(p.t, p.frame)
        after = self.scorer.award_point(p.winner, t=p.t, frame=p.frame, cause=p.cause)

        events = list(p.events)
        events.append(Event(type=EventType.RALLY_END, frame=p.frame, t=p.t,
                            confidence=p.confidence, reason=p.cause))
        events.append(Event(type=EventType.POINT, frame=p.frame, t=p.t,
                            confidence=p.confidence,
                            reason=f"point to {p.winner} ({p.cause})",
                            meta={"winner": p.winner, "cause": p.cause}))

        start_frame = getattr(self, "_rally_start_frame", p.frame)
        start_t = getattr(self, "_rally_start_t", p.t)
        outcome = (Outcome.SERVER if p.winner == before.serving_player
                   else Outcome.RECEIVER)

        record = PointRecord(
            index=len(self.points) + 1,
            t_start=float(start_t), t_end=float(p.t),
            frame_start=int(start_frame), frame_end=int(p.frame),
            winner=p.winner, loser=loser, outcome=outcome, cause=p.cause,
            score_before=before, score_after=after,
            confidence=p.confidence, needs_review=p.needs_review, events=events,
        )

        if after.match_over:
            events.append(Event(type=EventType.MATCH, frame=p.frame, t=p.t,
                                reason=f"match won by {after.winner}"))
        elif _game_index(before) != _game_index(after):
            events.append(Event(type=EventType.GAME, frame=p.frame, t=p.t,
                                reason="game/set completed"))

        self.points.append(record)
        self.timeline.append(after)
        return record

    # -- helpers ------------------------------------------------------------
    def commit_if_over(self) -> None:  # pragma: no cover - convenience
        if self.pending is not None:
            self._commit_pending()

    @property
    def score(self) -> ScoreState:
        state = self.timeline[-1] if self.timeline else self.scorer.snapshot()
        return state


# --------------------------------------------------------------------------- #
# Badminton
# --------------------------------------------------------------------------- #
class BadmintonRallyEngine(RallyEngine):
    """A shuttle landing on the floor ends the rally immediately."""

    sport_name = "badminton"

    def _handle_landing(self, ev: Event, projected: bool = False) -> None:
        landing_side = (self.court.half_of(ev.court_xy)
                        if ev.court_xy is not None and self.court.registered else None)
        striker = self._attribute_striker(ev, landing_side)
        strike_side = None
        if striker is not None:
            strike_side = self.scorer.side_of.get(striker)
            self.striker_side = strike_side

        verdict = self._verdict(ev, is_serve=self.serve_pending,
                                striking_side=strike_side)

        if striker is None:
            reason = "landing with unknown striker"
            self.warnings.append(reason)
            return

        if verdict is None:
            # cannot judge geometry -> fall back to the loss-of-track policy
            self._decide(self.scorer.opponent(striker), cause="unregistered_court",
                         frame=ev.frame, t=ev.t, confidence=0.3, needs_review=True)
            return

        extra = self._adjudicated_events(ev, verdict)
        for e in extra:
            self.rally_events.append(e)

        conf = float(min(ev.confidence, 1.0)) * (0.85 if projected else 1.0)
        if verdict.zone == "own_court":
            self._decide(self.scorer.opponent(striker), cause="landed_own_court",
                         frame=ev.frame, t=ev.t, confidence=conf)
        elif verdict.in_bounds:
            # shuttle hit the floor in the opponent's court: striker wins
            self._decide(striker, cause="landing_in",
                         frame=ev.frame, t=ev.t, confidence=conf,
                         reopen_if_hit=True)
        elif self.serve_pending:
            self._decide(self.scorer.opponent(striker), cause="serve_fault_out",
                         frame=ev.frame, t=ev.t, confidence=conf)
        else:
            self._decide(self.scorer.opponent(striker), cause="landing_out",
                         frame=ev.frame, t=ev.t, confidence=conf)

    def _after_hit(self, ev: Event) -> None:
        # once the serve has been returned, later landings are rally landings
        self.serve_pending = False


# --------------------------------------------------------------------------- #
# Tennis
# --------------------------------------------------------------------------- #
class TennisRallyEngine(RallyEngine):
    """Bounces, double bounces and the two-serve fault model."""

    sport_name = "tennis"

    def _reset_sport_state(self) -> None:
        self.faults = 0
        self.last_bounce_side: Optional[Side] = None
        self.first_serve = True

    def _on_serve_struck(self, ev: Event) -> None:
        self.serve_pending = True

    def _after_hit(self, ev: Event) -> None:
        # once the serve has been returned, later landings are rally landings
        self.serve_pending = False
        self.last_bounce_side = None

    def _after_nonterminal_landing(self, ev: Event) -> None:
        # tennis: track which side the ball last bounced on so that a second
        # bounce on the same side can be recognised as a double bounce
        if ev.court_xy is None or not self.court.registered:
            return
        side = self.court.half_of(ev.court_xy)
        if self.last_bounce_side is side:
            striker = self.scorer.player_at(side.other)
            self.rally_events.append(Event(
                type=EventType.DOUBLE_BOUNCE, frame=ev.frame, t=ev.t,
                confidence=ev.confidence, side=side, court_xy=ev.court_xy,
                img_xy=ev.img_xy, reason="second bounce on the same side"))
            self._decide(striker, cause="double_bounce", frame=ev.frame, t=ev.t,
                         confidence=ev.confidence)
        self.last_bounce_side = side

    def _handle_landing(self, ev: Event, projected: bool = False) -> None:
        landing_side = (self.court.half_of(ev.court_xy)
                        if ev.court_xy is not None and self.court.registered else None)
        striker = self._attribute_striker(ev, landing_side)
        if striker is None:
            self.warnings.append("tennis landing with unknown striker")
            return

        is_serve = self.serve_pending
        verdict = self._verdict(ev, is_serve=is_serve)
        if verdict is None:
            if is_serve:
                self._register_fault(ev, striker, "unregistered_court")
            else:
                self._decide(self.scorer.opponent(striker), cause="unregistered_court",
                             frame=ev.frame, t=ev.t, confidence=0.3, needs_review=True)
            return

        extra = self._adjudicated_events(ev, verdict)
        for e in extra:
            self.rally_events.append(e)
        conf = float(min(ev.confidence, 1.0)) * (0.85 if projected else 1.0)

        if is_serve:
            if verdict.in_bounds:
                self.serve_pending = False
                self.expecting_serve = False
                self.last_bounce_side = (self.court.half_of(ev.court_xy)
                                         if ev.court_xy else None)
            else:
                self._register_fault(ev, striker, "serve_fault_out")
            return

        if verdict.zone == "own_court":
            self._decide(self.scorer.opponent(striker), cause="landed_own_court",
                         frame=ev.frame, t=ev.t, confidence=conf)
            return

        if not verdict.in_bounds:
            self._decide(self.scorer.opponent(striker), cause="landing_out",
                         frame=ev.frame, t=ev.t, confidence=conf)
            return

        landing_side = self.court.half_of(ev.court_xy) if ev.court_xy else None
        if landing_side is not None and self.last_bounce_side == landing_side:
            # two bounces on the same side without an intervening hit
            self.rally_events.append(Event(
                type=EventType.DOUBLE_BOUNCE, frame=ev.frame, t=ev.t,
                confidence=conf, side=landing_side, court_xy=ev.court_xy,
                img_xy=ev.img_xy, reason="second bounce on the same side"))
            self._decide(striker, cause="double_bounce",
                         frame=ev.frame, t=ev.t, confidence=conf)
        else:
            self.last_bounce_side = landing_side

    def _register_fault(self, ev: Event, striker: str, cause: str) -> None:
        self.faults += 1
        self.serve_pending = True
        self.expecting_serve = True          # second serve must be served
        self.last_bounce_side = None
        self.rally_events.append(Event(
            type=EventType.BOUNCE_OUT, frame=ev.frame, t=ev.t,
            confidence=ev.confidence, court_xy=ev.court_xy, img_xy=ev.img_xy,
            in_bounds=False, reason=f"{cause} (fault {self.faults})",
            meta={"fault": self.faults, "striker": striker}))
        if self.faults >= 2:
            self._decide(self.scorer.opponent(striker), cause="double_fault",
                         frame=ev.frame, t=ev.t, confidence=ev.confidence)
        else:
            self.warnings.append(f"first serve fault at frame {ev.frame}: let, second serve")

    def _on_net(self, ev: Event) -> None:
        striker = self._player_side()
        if striker is None:
            return
        if self.serve_pending:
            self._register_fault(ev, striker, "serve_fault_net")
            return
        super()._on_net(ev)


def _game_index(state: ScoreState) -> int:
    """Monotonic index of the game/set segment, used to detect game changes."""
    score = state.score or {}
    if "game_number" in score:
        return int(score["game_number"])
    return len(score.get("set_scores", [])) * 100 + sum(score.get("games", {}).values())


ENGINES: Dict[str, type] = {
    "badminton": BadmintonRallyEngine,
    "tennis": TennisRallyEngine,
}


def make_rally_engine(sport: str, rules: SportRules, scorer: MatchScorer,
                      court: CourtModel, mode: str = "singles",
                      cfg: Optional[dict] = None) -> RallyEngine:
    cls = ENGINES.get(str(sport).lower(), RallyEngine)
    return cls(rules, scorer, court, mode=mode, cfg=cfg)
