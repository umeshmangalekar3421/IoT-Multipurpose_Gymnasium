# Implemented rules

This file maps every implemented rule to its source and to the code that applies
it. If a decision is disputed, this is the table to check.

## Coordinate frame

All rules are written in **court metres**, never in pixels:

```
y
^
|  (0,0) ----------------- (0,W)      far baseline  (top of the image)
|    |                       |
|    |          NET          |        +x  runs towards the camera
|    |           |           |        +y  runs to the right of the image
|  (L,0) ----------------- (L,W)      near baseline (bottom of the image)
+----------------------------------> x
```

* `L` = 13.40 m (badminton) / 23.77 m (tennis)
* `W` = 6.10 m (badminton doubles) / 10.97 m (tennis doubles)
* Side **A** owns `x < L/2`, side **B** owns `x > L/2`.
* A player standing in side A *faces the camera*, so their right-hand service
  court is at **large y**. This single convention makes serve parity correct for
  both sports (see `MatchScorer.serve_target`).

The convention is fixed by `CourtModel.from_corners(..., order="tl_tr_br_bl")`:
corners are clicked clockwise from the far-left corner of the image and mapped to
`(0,0) → (0,W) → (L,W) → (L,0)`.

---

## Badminton — BWF Laws of Badminton

| Law | Rule | Implementation |
|---|---|---|
| 7.2 | Rally point scoring: a point is won on every rally | `BadmintonMatch.award_point` |
| 7.3 | Match is the best of 3 games | `best_of=3`, `sets_to_win` |
| 7.4.1 | Game to 21 points | `points_to=21` |
| 7.4.2 | At 20-all, a 2-point lead is required | `win_by=2` |
| 7.4.3 | At 29-all, the next point wins (30 cap) | `points_cap=30` |
| 9.2 | The serve is diagonal, into the opposite service court | `BadmintonMatch.serve_target`, `CourtModel.service_box_for_band` |
| 9.2 | Even score → right service court, odd → left | `service_court_low_band()` |
| 9.3 | In doubles the serve must land before the doubles long service line (5.94 m from the net) | `CourtSpec.doubles_long_service_inset`, `serve_depth("doubles")` |
| 9.3 | In singles the serve may reach the back boundary (13.40 m) | `serve_depth("singles")` |
| 11.3 | Sides change at the end of each game | `flip_sides()` in `_close_game` |
| 11.5 | In the deciding game, sides change when a side reaches 11 | `deciding_game_interval=11` |
| 13 | A shuttle landing outside the boundary is a fault by the striker | `BadmintonRules.judge_landing` |
| 13.3 | A serve not landing in the correct service court is a fault | `judge_landing(is_serve=True)` |
| 13.4 | A shuttle that fails to clear the net is a fault | `EventType.NET_CONTACT` → `net_fault` |

Serving side rule: BWF 9.2 says the *server* stands in the right service court
when their own score is even. `service_court_low_band()` implements exactly that
with the `right_low_band = (server_side is Side.B)` mapping derived from the
court frame above.

Singles vs doubles geometry:

| | Singles | Doubles |
|---|---|---|
| Sidelines | 5.18 m apart | 6.10 m apart (alleys are IN) |
| Rally back boundary | 13.40 m | 13.40 m |
| Serve back limit | 13.40 m from the server | 5.94 m from the net (doubles long service line) |

---

## Tennis — ITF Rules of Tennis

| Rule | Rule text | Implementation |
|---|---|---|
| 5 | Points: 0 (love), 15, 30, 40 | `POINT_NAMES`, `points_display` |
| 5 | A game needs 4 points and a 2-point lead | `_game_won_by` |
| 5 | At 40-40 (deuce) two consecutive points are needed (advantage) | `_game_won_by`, `no_ad=False` |
| 5 | Optional no-ad scoring (college / FAST4) | `no_ad=True` — one point at deuce |
| 5 | A set is won with 6 games and a 2-game lead | `_close_game` |
| 5 | At 6-6 a tie-break is played to 7 (2-point lead) | `in_tiebreak`, `tiebreak_to=7` |
| 5 | Tie-break serve rotation: 1 point, then 2 points each | `_server_for_point` |
| 5 | Sides change after the 1st, 3rd, 5th… game of a set | `total_games % 2 == 1` |
| 5 | Sides change every 6 points in a tie-break | `tiebreak_points_played % 6 == 0` |
| 10 | The receiver serves the next game | `_close_game` |
| 14 | The serve must land in the diagonally opposite service box | `serve_target`, `service_box_for_band` |
| 14 | The serve box is 6.40 m deep in singles **and** doubles | `doubles_serve_uses_long_service_line=False` |
| 14 | The doubles alley is OUT on a serve | `serve_uses_full_width_in_doubles=False` |
| 17/18 | A ball landing outside the lines is out | `TennisRules.judge_landing` |
| — | Two bounces on one side = point to the striker | `double_bounce` in `TennisRallyEngine` |
| — | First-serve fault is a let, not a point | `_register_fault` (two-serve model) |

Court: 23.77 m × 10.97 m, singles 8.23 m, service line 6.40 m from the net,
centre service line splitting each service box.

---

## Common adjudication rules

| Situation | Verdict | Code |
|---|---|---|
| Ball centre ≤ 3 cm outside a line | **IN** (ball radius tolerance) | `line_margin_m` |
| Landing within `margin + 2 cm` of a line | IN **and** flagged `LINE_CALL` | `LandingVerdict.line_call` |
| Ball lands on the striker's own half, inside the court | `own_court` fault | `judge_landing` |
| Ball lands beyond the lines on the striker's own half | `out` (not own-court) | same, gated on `in_play()` |
| Ball touches the net and dies | `net_fault` → point to the opponent | `RallyEngine._on_net` |
| Ball leaves the tracking volume mid-flight | ballistic extrapolation to a predicted landing; if impossible, `on_track_loss` policy | `extrapolate_landing`, `events.on_track_loss` |
| Floor contact after which the ball keeps flying | **not** the end of the rally | `terminal` flag in `detect_bounces` |

### Why the ball radius matters

A line judge (and Hawk-Eye) rule on whether the **ball** touches the line, not
whether its centre does. With a shuttle of radius ≈ 3 cm and a tennis ball of
radius ≈ 3.3 cm, using the centre alone produces systematically wrong calls of
exactly that magnitude — which is the width of the line paint. `line_margin_m`
implements the physical tolerance; setting it to `0` reproduces the naive
centre-based behaviour (used in the unit tests to prove the difference).

---

## Decision pipeline (who wins a rally)

```
event                      rule engine action
------------------------------------------------------------------------
SERVE / SHOT               rally starts (or continues); striker = event side
LANDING (terminal)         adjudicate point in court metres
  ├─ own court             point → opponent of the striker
  ├─ IN                    point → striker            (badminton: rally over)
  ├─ OUT                   point → opponent of striker
  └─ serve & outside box   point → opponent of the striker (serve fault)
LANDING (non-terminal)     rally continues; the player on that side returns
NET_CONTACT                point → opponent of the striker (net fault)
OUT_OF_PLAY                adjudicated as a landing when a court point exists,
                           otherwise the configured policy decides
------------------------------------------------------------------------
tennis extra:
  first serve fault        count a fault, expect a second serve (no point)
  second fault             double fault → point to the receiver
  second bounce, same side double bounce   → point to the striker
```

Striker attribution is **derived from the landing side** (the ball must have been
struck from the opposite half) rather than from the racket-contact event alone,
because the ball is frequently not visible exactly at contact. The tracked
striker is used only to recognise a shank into one's own court.
