"""Scripted demo matches with verified ground truth.

Rallies are generated *against the live rule engine*: the serve target is taken
from the service box implied by the current score, returns alternate between the
two sides, and a fraction of rallies end with an out landing, a serve fault or a
double fault.  Expected results are then produced by running the real rule engine
over the *perfect* event stream derived from the script — nothing is hard-coded.

That makes the benchmark meaningful: running the engine on perfect events and on
*detected* events differs only by perception.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Side
from ..court.model import BADMINTON, TENNIS, CourtModel, CourtSpec
from ..sports.registry import get_rules
from ..analysis.rally import make_rally_engine
from .scene import MatchScript, RallyScript, Shot, perfect_events

BUILDERS = {}


def _rand_point(rng, lo: float, hi: float, loy: float, hiy: float) -> Tuple[float, float]:
    return float(rng.uniform(lo, hi)), float(rng.uniform(loy, hiy))


def _play_area(spec: CourtSpec, side: Side, mode: str) -> Tuple[float, float, float, float]:
    y0, y1 = (0.0, spec.width) if mode == "doubles" else (spec.singles_y_min, spec.singles_y_max)
    if side is Side.A:
        return 0.4, spec.net_x - 0.5, y0 + 0.2, y1 - 0.2
    return spec.net_x + 0.5, spec.length - 0.4, y0 + 0.2, y1 - 0.2


def _outside_point(spec: CourtSpec, side: Side, mode: str, rng,
                   kind: str = "long") -> Tuple[float, float]:
    """A landing point just outside the court, on the given side."""
    y0, y1 = (0.0, spec.width) if mode == "doubles" else (spec.singles_y_min, spec.singles_y_max)
    ymid = (y0 + y1) / 2.0
    if kind == "wide":
        y = (y1 + 0.45) if rng.random() < 0.5 else (y0 - 0.45)
        x = rng.uniform(spec.net_x + 0.6, spec.length - 0.6) if side is Side.B else \
            rng.uniform(0.6, spec.net_x - 0.6)
        return float(x), float(y)
    x = (spec.length + 0.55) if side is Side.B else -0.55
    return float(x), float(rng.uniform(ymid - 1.0, ymid + 1.0))


def _generate(sport: str, spec: CourtSpec, fps: float, players: Tuple[str, str],
              n_rallies: int = 12, seed: int = 7, mode: str = "singles") -> MatchScript:
    rules = get_rules(sport)
    scorer = rules.new_match(list(players), mode=mode, best_of=3)
    court = CourtModel(spec, np.eye(3))
    rng = np.random.default_rng(seed)
    is_shuttle = sport == "badminton"

    rallies: List[RallyScript] = []
    t = 1.0
    while len(rallies) < n_rallies and not scorer.match_over:
        server_side = scorer.server_side
        tgt_side, target_low = scorer.serve_target()
        box = court.service_box_for_band(tgt_side, target_low, mode)
        x0 = min(p[0] for p in box); x1 = max(p[0] for p in box)
        y0 = min(p[1] for p in box); y1 = max(p[1] for p in box)

        serve_xy = (x0 + 0.62 * (x1 - x0), y0 + rng.uniform(0.3, 0.7) * (y1 - y0))
        sx = spec.net_x - 2.6 if server_side is Side.A else spec.net_x + 2.6
        sy = spec.mid_y * (0.62 if not target_low else 1.38)
        # amateur-pace flights: slow enough for a classical detector to follow
        # every frame, which keeps the demo focused on scoring correctness
        # Apex heights are chosen so the whole flight stays inside the camera
        # frustum (the renderer's vertical scale is ~36 px/m): a 5 m lob would
        # simply leave the top of the frame and could not be tracked at all.
        serve_apex = 2.4 if is_shuttle else 2.6
        z_serve = 1.5 if is_shuttle else 2.5

        shots: List[Shot] = [Shot(t=t, start=(sx, sy, z_serve),
                                  land=serve_xy, apex_m=serve_apex,
                                  struck_by=server_side, kind="serve")]
        prev_t = t + shots[-1].flight_time
        prev_land = serve_xy
        striker = server_side

        # ---- first serve fault? ------------------------------------------
        serve_fault = rng.random() < 0.16
        if serve_fault:
            bad = _outside_point(spec, tgt_side, mode, rng, "long")
            shots[0].land = bad
            prev_land, striker = bad, server_side
            if is_shuttle:
                rallies.append(RallyScript(len(rallies) + 1, shots, "", "", scorer.server))
                t = prev_t + 2.4
                scorer.award_point(scorer.opponent(scorer.server))
                continue
            # tennis: second serve from the same spot
            fault2 = rng.random() < 0.25
            second_land = (x0 + 0.5 * (x1 - x0),
                           y0 + rng.uniform(0.35, 0.65) * (y1 - y0))
            if fault2:
                second_land = _outside_point(spec, tgt_side, mode, rng, "wide")
            shots.append(Shot(t=prev_t + 0.9, start=(sx, sy, z_serve),
                              land=second_land, apex_m=serve_apex,
                              struck_by=server_side, kind="serve"))
            prev_t = prev_t + 0.9 + shots[-1].flight_time
            prev_land = second_land
            if fault2:
                rallies.append(RallyScript(len(rallies) + 1, shots, "", "", scorer.server))
                t = prev_t + 2.4
                scorer.award_point(scorer.opponent(scorer.server))
                continue

        # ---- rally returns -----------------------------------------------
        n_returns = int(rng.integers(0, 4 if is_shuttle else 5))
        final_out = rng.random() < 0.28
        side = tgt_side
        for k in range(n_returns):
            lo, hi, loy, hiy = _play_area(spec, side.other, mode)
            target = _rand_point(rng, lo, hi, loy, hiy)
            z_hit = 0.55 if is_shuttle else 0.9
            # The next shot is struck from where the ball came down, a moment
            # after it arrived: a poor player's return, but exactly what the
            # tracker sees, and it keeps every flight fully inside the frame.
            shots.append(Shot(t=prev_t + 0.07,
                              start=(prev_land[0], prev_land[1], z_hit),
                              land=target, apex_m=(2.9 if is_shuttle else 2.4),
                              struck_by=side, kind="shot"))
            prev_t = shots[-1].t_land
            prev_land, side = target, side.other

        # the last striker is whoever hit the ball last
        last_striker = shots[-1].struck_by
        if final_out:
            shots[-1].land = _outside_point(spec, last_striker.other, mode, rng,
                                           "wide" if rng.random() < 0.4 else "long")
            prev_land = shots[-1].land
            if not is_shuttle and shots[-1].kind == "serve":
                # A rally cannot end on a single out serve: the server gets a
                # second serve, which here is also out -> a legal double fault.
                shots.append(Shot(
                    t=shots[-1].t_land + 0.9,
                    start=shots[-1].start,
                    land=_outside_point(spec, last_striker.other, mode, rng,
                                        "wide" if rng.random() < 0.5 else "long"),
                    apex_m=serve_apex, struck_by=last_striker, kind="serve"))
                prev_t = shots[-1].t_land
        elif not is_shuttle:
            # a ball that is not returned bounces a second time on the same side
            lo, hi, loy, hiy = _play_area(spec, last_striker.other, mode)
            bounce2 = (float(np.clip(prev_land[0] + 1.2, lo, hi)),
                       float(np.clip(prev_land[1] + 0.4, loy, hiy)))
            shots.append(Shot(t=prev_t + 0.12, start=(prev_land[0], prev_land[1], 0.0),
                              land=bounce2, apex_m=0.55,
                              struck_by=last_striker, kind="dead_bounce"))
            prev_t = shots[-1].t_land

        rallies.append(RallyScript(len(rallies) + 1, shots, "", "", scorer.server))
        # provisional progression so the next rally's serve box is correct;
        # the reference engine pass below is what actually fills expected_winner
        striker_name = scorer.player_at(last_striker)
        provisional = (scorer.opponent(striker_name) if final_out
                                  else striker_name)
        scorer.award_point(provisional)
        t = prev_t + 2.4

    return MatchScript(sport=sport, spec=spec, players=players, fps=fps,
                       rallies=rallies, start_s=1.0)


def _annotate_with_reference(script: MatchScript, mode: str = "singles") -> MatchScript:
    """Fill expected results by running the rule engine on perfect events."""
    court = CourtModel(script.spec, np.eye(3))   # identity: metres == 'image'
    rules = get_rules(script.sport)
    scorer = rules.new_match(list(script.players), mode=mode, best_of=3)
    engine = make_rally_engine(script.sport, rules, scorer, court, mode, {})
    engine.process(perfect_events(script))
    engine.finalize_rally()
    for i, rally in enumerate(script.rallies):
        if i < len(engine.points):
            point = engine.points[i]
            rally.expected_winner = point.winner
            rally.cause = point.cause
            rally.expected_score = point.score_after.display if point.score_after else ""
        else:
            rally.expected_winner = ""
            rally.cause = "unresolved"
    script.reference = {
        "points": len(engine.points),
        "rallies": len(script.rallies),
        "final_score": scorer.display(),
        "winner": scorer.winner,
        "warnings": engine.warnings,
    }
    return script


def build_badminton_demo(fps: float = 30.0, players: Tuple[str, str] = ("P1", "P2"),
                         mode: str = "singles", n_rallies: int = 12,
                         seed: int = 7) -> MatchScript:
    script = _generate("badminton", BADMINTON, fps, players, n_rallies, seed, mode)
    return _annotate_with_reference(script, mode)


def build_tennis_demo(fps: float = 30.0, players: Tuple[str, str] = ("P1", "P2"),
                      mode: str = "singles", n_rallies: int = 10,
                      seed: int = 11) -> MatchScript:
    script = _generate("tennis", TENNIS, fps, players, n_rallies, seed, mode)
    return _annotate_with_reference(script, mode)


BUILDERS = {"badminton": build_badminton_demo, "tennis": build_tennis_demo}


def build_demo(sport: str, fps: float = 30.0, players: Tuple[str, str] = ("P1", "P2"),
               mode: str = "singles", n_rallies: int = 12, seed: int = 7) -> MatchScript:
    """Build a scripted match; ``n_rallies`` controls the length of the demo."""
    if sport not in BUILDERS:
        raise KeyError(f"no demo script for {sport!r}; choose from {sorted(BUILDERS)}")
    builder = build_badminton_demo if sport == "badminton" else build_tennis_demo
    return builder(fps=fps, players=players, mode=mode,
                   n_rallies=n_rallies, seed=seed)
