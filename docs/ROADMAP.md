# Roadmap

## Done
- [x] Badminton: BWF laws, singles + doubles service geometry, change of ends
- [x] Tennis: ITF rules, deuce/ad, no-ad, sets, tie-break, serve rotation
- [x] Court calibration: corners, homography, auto line detection with
      rotation/mirror disambiguation and line-fit refinement
- [x] Ball detection (classical + YOLO adapter), offline linking, online Kalman
- [x] Event extraction: bounces, strokes, net contacts, ball at rest, ballistic
      landing extrapolation
- [x] Rally engine: serve faults, double faults, double bounces, own-court,
      non-terminal bounces, confirmation window
- [x] Evidence: per-point JPEGs, scoreboard, proof sheet, JSON + CSV timelines
- [x] Learned contact classifier (NumPy MLP) + dataset exporter
- [x] Simulation with verified ground truth + layered benchmark
- [x] Review dashboard (stdlib), CLI, 70 tests

## Next (highest value first)
1. **Train the contact classifier on real footage.** Export contacts from 5–10
   club matches, correct the contested ones in the dashboard, retrain. This is
   the single biggest accuracy lever.
2. **Train the ball detector.** `train-ball` → correct ~400 frames → YOLOv8n at
   960 px. Removes the "ball must be the only moving object" assumption.
3. **Hawk-eye style 3D correction.** Fit a ballistic model in 3D using the
   image-y signal as a height proxy, which would remove the residual landing bias
   near the baseline.
4. **Player-aware rallies.** Track both players, detect the strike from the
   racket-arm motion, use it to confirm striker attribution. Also enables
   distance-covered and heat-map analytics.

## Volleyball and basketball (second pair of sports)
Both share the perception and evidence layers. What is missing is sport logic:

* **Volleyball** — the ball must not touch the floor; a point ends when it does.
  Need: attack-line geometry (3 m), net-crossing rules, three-touch counting,
  and rotation tracking for the six players. `CourtSpec` already holds 18 × 9 m
  and the net; `EventType.NET_CONTACT` exists.
* **Basketball** — scoring is shot-based, not landing-based. Need: hoop
  localisation (two hoops), a shot arc classifier (make/miss, 2 vs 3 points by
  distance from the arc), 24-second shot clock, and period/clock handling. The
  `MatchScorer` interface fits; the geometry work is the hoop projection.

An honest estimate: volleyball ≈ 2–3 weeks (mostly player/rotation logic),
basketball ≈ 3–4 weeks (hoop detection + shot model).

## Not planned
- Handheld/panning camera support (needs stabilisation + moving homography).
- Ball spin estimation (needs high frame rate and sub-pixel tracking).
- Multi-camera fusion (out of scope for the current hardware budget).
