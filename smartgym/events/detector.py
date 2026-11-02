"""Event detector: tracked ball -> semantic, adjudicable events.

This module is the bridge between perception (dots on frames) and the rules
(points).  It emits *raw* physical events — ``SHOT``, ``LANDING``,
``NET_CONTACT``, ``OUT_OF_PLAY`` — and never decides who won a point; that is
the job of :mod:`smartgym.analysis.rally`, which owns the rule book.

A pluggable ML classifier
-------------------------
Hand-written thresholds saturate quickly when lighting, camera angle or sport
change.  ``EventDetector(model_path=...)`` therefore accepts a trained
classifier (:mod:`smartgym.training.train_event_model`) that scores each
contact from a feature window (velocity, direction change, speed drop, height
above the court, proximity to the net).  When a model is provided it *blends*
its probability with the physics confidence instead of replacing the physics,
so a badly calibrated model degrades gracefully instead of destroying the match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.types import Event, EventType, Point, Side, TrackPoint
from ..court.model import CourtModel
from .trajectory import (
    Contact,
    Segment,
    annotate_segment,
    build_segments,
    segment_contacts,
)


@dataclass
class DetectionResult:
    """Everything the detector produced for one clip."""

    events: List[Event]
    segments: List[Segment]
    contacts: List[Contact]
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def track_coverage(self) -> float:
        return float(self.stats.get("track_coverage", 0.0))


class EventDetector:
    """Sport-agnostic physical event extraction from a ball trajectory."""

    def __init__(self, court: CourtModel, fps: float = 30.0,
                 cfg: Optional[dict] = None,
                 classifier: Any = None):
        self.court = court
        self.fps = float(fps)
        ev = (cfg or {}).get("events", {}) if cfg else {}
        self.min_prominence = float(ev.get("min_bounce_prominence_px", 8.0))
        self.min_spacing = int(ev.get("min_bounce_spacing_frames", 4))
        self.return_window = int(ev.get("return_window_frames", 40))
        # Law-level fact: in badminton (and volleyball) a floor contact kills the
        # rally outright; in tennis a bounce is part of normal play and the rally
        # only ends when nobody returns the ball.
        # Default: decide from whether the ball is returned, which works for
        # every sport and for video where rallies follow each other quickly.
        # Force it with ``events.floor_contact_ends_rally: true`` when a floor
        # contact is decisive by law and returns are unreliable (e.g. badminton
        # on footage where the next serve follows within a second).
        self.floor_ends_rally = bool(ev.get("floor_contact_ends_rally", False))
        self.max_gap = int((cfg or {}).get("tracking", {}).get("max_coast_frames", 12)
                           if cfg else 12)
        self.max_speed_mps = 130.0
        self.net_band_m = 0.35
        self.classifier = classifier
        self.n_frames = 0

    # -- main API -----------------------------------------------------------
    def detect(self, track: Sequence[TrackPoint],
               n_frames: Optional[int] = None) -> DetectionResult:
        self.n_frames = int(n_frames or (track[-1].frame + 1 if track else 0))
        segments = build_segments(track, max_gap=self.max_gap,
                                  max_speed_mps=self.max_speed_mps, fps=self.fps)
        net_x = self.court.spec.net_x
        segments = [annotate_segment(s, net_x, self.net_band_m) for s in segments]

        contacts = segment_contacts(
            segments, net_x, net_band_m=self.net_band_m,
            bounce_kwargs=dict(min_prominence_px=self.min_prominence,
                               min_spacing_frames=self.min_spacing),
        )
        # A ball that vanishes mid-flight is still airborne: replace the raw
        # loss position with a predicted floor contact when the parabola says it
        # will come down shortly.  This is what makes in/out calls possible at
        # the moment the ball leaves the frame.
        contacts = self._predict_landings(contacts, segments)
        contacts = self._dedupe(contacts)
        contacts = self._mark_terminal(contacts)
        contacts = [self._apply_classifier(c, segments) for c in contacts]
        events = [self._to_event(c) for c in contacts]
        events = self._purge_noise(events)

        coverage = (len(track) / self.n_frames) if self.n_frames else 0.0
        stats = {
            "n_segments": len(segments),
            "n_contacts": len(contacts),
            "n_events": len(events),
            "track_coverage": round(float(coverage), 4),
            "n_bounces": sum(1 for c in contacts if c.kind == "bounce"),
            "n_hits": sum(1 for c in contacts if c.kind == "hit"),
            "n_net": sum(1 for c in contacts if c.kind == "net"),
            "max_ball_speed_mps": round(
                max([s.max_speed_mps or 0.0 for s in segments], default=0.0), 2),
        }
        return DetectionResult(events=events, segments=segments,
                               contacts=contacts, stats=stats)

    # -- internals ----------------------------------------------------------
    def _predict_landings(self, contacts: List[Contact],
                          segments: Sequence[Segment]) -> List[Contact]:
        from .trajectory import extrapolate_landing

        out: List[Contact] = []
        by_first = {s.first_frame: s for s in segments}
        for c in contacts:
            if c.kind != "lost":
                out.append(c)
                continue
            seg = next((s for s in segments if s.last_frame == c.frame), None)
            pred = extrapolate_landing(seg) if seg is not None else None
            if pred is None:
                out.append(c)
                continue
            court_xy = self.court.to_court_point(pred) if self.court.registered else None
            out.append(Contact(
                kind="bounce",
                frame=c.frame,
                t=c.t,
                img_xy=pred,
                court_xy=court_xy,
                side=(self.court.half_of(court_xy).value
                      if court_xy is not None else None),
                confidence=0.55,
                meta={"reason": "predicted_landing", "predicted": True,
                      "from": "ballistic_extrapolation"},
            ))
        return out

    def _mark_terminal(self, contacts: List[Contact]) -> List[Contact]:
        """Decide whether a floor contact ended the rally.

        The physics cannot answer this: a tennis ball bounces *and* keeps
        flying, while a shuttle that touches the floor is dead.  What actually
        decides it is whether anybody returned the ball — i.e. whether another
        racket contact follows.  A bounce with a return inside
        ``return_window_frames`` was a rally bounce; without one, the rally is
        over and the contact becomes decisive for scoring.
        """
        window = self.return_window
        hits = [c.frame for c in contacts if c.kind in ("hit", "net")]
        hits.sort()
        for c in contacts:
            if c.kind not in ("bounce",) or not isinstance(c.meta, dict):
                continue
            # An *observed* floor contact obeys the sport's law (a shuttle on the
            # floor is dead).  A *predicted* landing is only a hypothesis about a
            # ball that vanished, so it must be corroborated: if anybody returns
            # the ball, the flight was simply intercepted and the rally lives on.
            if c.meta.get("rebounded"):
                c.meta["terminal"] = False
                continue
            if self.floor_ends_rally and not c.meta.get("predicted"):
                c.meta["terminal"] = True
                continue
            nxt = next((f for f in hits if 0 < f - c.frame <= window), None)
            c.meta["terminal"] = nxt is None
            if nxt is not None:
                c.meta["returned_after_frames"] = int(nxt - c.frame)
        return contacts

    def _dedupe(self, contacts: List[Contact]) -> List[Contact]:
        """Collapse contacts that describe the same physical instant."""
        out: List[Contact] = []
        for c in sorted(contacts, key=lambda c: (c.frame, -c.confidence)):
            if out and abs(c.frame - out[-1].frame) <= 2 and c.kind == out[-1].kind:
                if c.confidence > out[-1].confidence:
                    out[-1] = c
                continue
            out.append(c)
        return out

    def _apply_classifier(self, contact: Contact, segments: List[Segment]) -> Contact:
        """Blend the physics confidence with a learned bounce/hit probability."""
        if self.classifier is None or contact.kind not in ("bounce", "hit"):
            return contact
        try:
            from ..training.features import contact_features  # local import: optional torch

            seg = next((s for s in segments
                        if s.first_frame <= contact.frame <= s.last_frame), None)
            if seg is None:
                return contact
            feat = contact_features(seg, contact, self.court, self.fps)
            prob = float(self.classifier.predict_proba(feat.reshape(1, -1))[0])
            contact.meta["classifier_p"] = round(prob, 3)
            # 60/40 blend keeps the physics prior dominant
            contact.confidence = float(np.clip(0.6 * contact.confidence + 0.4 * prob, 0, 1))
        except Exception as exc:  # never break the pipeline for an optional model
            contact.meta["classifier_error"] = str(exc)
        return contact

    def _to_event(self, c: Contact) -> Event:
        side: Optional[Side] = None
        if c.court_xy is not None and self.court.registered:
            side = self.court.half_of(c.court_xy)
        elif c.side in ("A", "B"):
            side = Side(c.side)

        if c.kind == "bounce":
            etype = EventType.LANDING
            reason = "ball contact with the floor"
        elif c.kind == "hit":
            etype = EventType.SHOT
            reason = "racket contact"
        elif c.kind == "net":
            etype = EventType.NET_CONTACT
            reason = "ball died at the net"
        else:
            etype = EventType.OUT_OF_PLAY
            reason = "ball left the tracking volume"

        meta = {**(c.meta or {}),
                "speed_mps": None if c.speed_mps is None else round(float(c.speed_mps), 2)}
        if etype is EventType.LANDING and "terminal" not in meta:
            meta["terminal"] = True      # unknown -> treat as decisive
        return Event(
            type=etype,
            frame=c.frame,
            t=c.t,
            confidence=c.confidence,
            side=side,
            court_xy=c.court_xy,
            img_xy=c.img_xy,
            reason=reason,
            meta=meta,
        )

    def _purge_noise(self, events: List[Event]) -> List[Event]:
        """Drop contacts that are physically redundant, keep chronology."""
        cleaned: List[Event] = []
        for e in events:
            if cleaned:
                prev = cleaned[-1]
                too_close = (e.frame - prev.frame) <= 1
                # consecutive 'hit' events with no ball travel are tracker jitter
                if too_close and e.type is prev.type and e.confidence <= prev.confidence:
                    continue
            cleaned.append(e)
        return cleaned
