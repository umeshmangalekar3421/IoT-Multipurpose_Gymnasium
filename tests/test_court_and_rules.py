"""Court geometry, homography and in/out adjudication."""

from __future__ import annotations

import numpy as np
import pytest

from smartgym.core.geometry import (
    apply_homography,
    homography_dlt,
    local_maxima,
    parabolic_vertex,
    point_in_polygon,
    polygon_mask,
    savgol_smooth,
)
from smartgym.core.types import Side
from smartgym.court.model import BADMINTON, TENNIS, CourtModel
from smartgym.sports.base import LandingContext
from smartgym.sports.badminton import BadmintonRules
from smartgym.sports.tennis import TennisRules


class TestGeometry:
    def test_homography_roundtrip(self):
        court_pts = [(0, 0), (13.4, 0), (13.4, 6.1), (0, 6.1)]
        image_pts = [(100, 80), (1180, 90), (1240, 700), (40, 690)]
        H = homography_dlt(image_pts, court_pts)
        back = apply_homography(H, image_pts)
        assert np.allclose(back, np.asarray(court_pts), atol=1e-6)

    def test_homography_with_many_points(self):
        rng = np.random.default_rng(0)
        court_pts = rng.uniform(0, 10, size=(8, 2))
        image_pts = court_pts * 60 + np.array([320, 240])
        H = homography_dlt(image_pts, court_pts)
        assert np.allclose(apply_homography(H, image_pts), court_pts, atol=1e-6)

    def test_homography_requires_four_points(self):
        with pytest.raises(ValueError):
            homography_dlt([(0, 0), (1, 1)], [(0, 0), (1, 1)])

    def test_point_in_polygon(self):
        square = [(0, 0), (2, 0), (2, 2), (0, 2)]
        assert point_in_polygon(square, (1, 1))
        assert not point_in_polygon(square, (3, 1))
        assert polygon_mask(square, [(1, 1), (3, 3), (0.5, 1.5)]).tolist() == [
            True, False, True]

    def test_local_maxima(self):
        y = np.array([0, 1, 5, 1, 0, 1, 8, 1, 0], dtype=float)
        assert local_maxima(y, min_prominence=1.0) == [2, 6]
        assert local_maxima(y, min_prominence=6.0) == [6]

    def test_parabolic_vertex(self):
        off, val = parabolic_vertex(3.0, 4.0, 3.0)
        assert off == pytest.approx(0.0) and val == pytest.approx(4.0)
        off2, _ = parabolic_vertex(1.0, 4.0, 5.0)   # rising: peak shifted right
        assert off2 > 0
        off3, _ = parabolic_vertex(5.0, 4.0, 1.0)   # falling: peak shifted left
        assert off3 < 0

    def test_savgol_preserves_length_and_smooths(self):
        x = np.sin(np.linspace(0, 6, 40)) + np.random.default_rng(0).normal(0, 0.2, 40)
        y = savgol_smooth(x, 7)
        assert len(y) == len(x)
        assert np.std(np.diff(y)) < np.std(np.diff(x))


class TestCourtModel:
    def test_from_corners_maps_to_metres(self):
        # corners given clockwise from the far-left (image top-left)
        corners = [(420, 100), (860, 100), (1260, 684), (20, 684)]
        court = CourtModel.from_corners(BADMINTON, corners)
        mapped = court.to_court(corners)
        expected = [(0, 0), (0, 6.1), (13.4, 6.1), (13.4, 0)]
        assert np.allclose(mapped, expected, atol=1e-6)

    def test_half_of(self):
        court = CourtModel(BADMINTON, np.eye(3))
        assert court.half_of((2.0, 3.0)) is Side.A
        assert court.half_of((11.0, 3.0)) is Side.B

    def test_in_play_singles_vs_doubles(self):
        court = CourtModel(BADMINTON, np.eye(3))
        alley_point = (6.0, 0.2)          # inside doubles, outside singles
        assert court.in_play(alley_point, "doubles")
        assert not court.in_play(alley_point, "singles")

    def test_service_box_geometry(self):
        court = CourtModel(BADMINTON, np.eye(3))
        box = court.service_box_for_band(Side.B, target_low_band=True, mode="singles")
        xs = [p[0] for p in box]
        assert min(xs) == pytest.approx(6.7)
        assert max(xs) == pytest.approx(8.68)     # net + 1.98 m

    def test_badminton_doubles_serve_depth(self):
        court = CourtModel(BADMINTON, np.eye(3))
        singles = court.serve_depth("singles")
        doubles = court.serve_depth("doubles")
        assert singles == pytest.approx(1.98)
        assert doubles == pytest.approx(6.70 - 0.76)

    def test_tennis_serve_never_uses_the_alley(self):
        court = CourtModel(TENNIS, np.eye(3))
        lo, hi = court.serve_width_limits("doubles")
        assert lo == pytest.approx(TENNIS.singles_y_min)
        assert hi == pytest.approx(TENNIS.singles_y_max)
        assert court.serve_depth("doubles") == pytest.approx(TENNIS.service_line_from_net)


class TestAdjudication:
    def _ctx(self, is_serve=False, mode="singles", box=None, striking=None):
        return LandingContext(is_serve=is_serve, mode=mode,
                              expected_service_box=box, striking_side=striking,
                              line_margin_m=0.03)

    def test_badminton_aut_line(self):
        court = CourtModel(BADMINTON, np.eye(3))
        rules = BadmintonRules()
        v = rules.judge_landing(court, (6.0, 2.0), self._ctx())
        assert v.in_bounds and v.zone == "play_area"

    def test_badminton_back_boundary_is_shared(self):
        """The 13.40 m back boundary applies in both singles and doubles."""
        court = CourtModel(BADMINTON, np.eye(3))
        rules = BadmintonRules()
        assert rules.judge_landing(court, (13.35, 3.0), self._ctx(mode="singles")).in_bounds
        assert rules.judge_landing(court, (13.35, 3.0), self._ctx(mode="doubles")).in_bounds
        assert not rules.judge_landing(court, (13.45, 3.0), self._ctx(mode="doubles")).in_bounds

    def test_badminton_sidelines_differ(self):
        """Singles sidelines are 5.18 m apart, doubles 6.10 m: the alley is out."""
        court = CourtModel(BADMINTON, np.eye(3))
        rules = BadmintonRules()
        alley = (8.0, 0.3)
        assert not rules.judge_landing(court, alley, self._ctx(mode="singles")).in_bounds
        assert rules.judge_landing(court, alley, self._ctx(mode="doubles")).in_bounds

    def test_ball_radius_tolerance(self):
        """A ball whose centre is 2 cm outside a line is still IN."""
        court = CourtModel(BADMINTON, np.eye(3))
        rules = BadmintonRules()
        just_out = (13.42, 3.0)
        strict = LandingContext(mode="singles", line_margin_m=0.0)
        loose = LandingContext(mode="singles", line_margin_m=0.03)
        assert not rules.judge_landing(court, just_out, strict).in_bounds
        assert rules.judge_landing(court, just_out, loose).in_bounds

    def test_line_call_is_flagged(self):
        court = CourtModel(BADMINTON, np.eye(3))
        rules = BadmintonRules()
        v = rules.judge_landing(court, (13.35, 3.0), self._ctx())
        assert v.line_call, "a landing 5 cm from the baseline is a line call"

    def test_own_court_is_a_fault(self):
        court = CourtModel(BADMINTON, np.eye(3))
        rules = BadmintonRules()
        v = rules.judge_landing(court, (4.0, 2.0), self._ctx(striking=Side.A))
        assert not v.in_bounds and v.zone == "own_court"

    def test_serve_outside_service_box(self):
        court = CourtModel(BADMINTON, np.eye(3))
        rules = BadmintonRules()
        box = court.service_box_for_band(Side.B, True, "singles")
        # beyond the short service line
        v = rules.judge_landing(court, (10.5, 2.0), self._ctx(is_serve=True, box=box))
        assert not v.in_bounds and v.zone == "out"
        # inside the box
        v2 = rules.judge_landing(court, (7.5, 1.5), self._ctx(is_serve=True, box=box))
        assert v2.in_bounds

    def test_tennis_serve_box_and_alley(self):
        court = CourtModel(TENNIS, np.eye(3))
        rules = TennisRules()
        box = court.service_box_for_band(Side.B, True, "doubles")
        assert rules.judge_landing(court, (16.0, 2.0), self._ctx(
            is_serve=True, mode="doubles", box=box)).in_bounds
        # in the doubles alley: out on a serve even in doubles
        assert not rules.judge_landing(court, (16.0, 0.4), self._ctx(
            is_serve=True, mode="doubles", box=box)).in_bounds

    def test_tennis_baseline(self):
        court = CourtModel(TENNIS, np.eye(3))
        rules = TennisRules()
        assert not rules.judge_landing(court, (24.0, 5.0), self._ctx()).in_bounds
        assert rules.judge_landing(court, (23.0, 5.0), self._ctx()).in_bounds
