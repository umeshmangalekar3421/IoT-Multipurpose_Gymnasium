"""Official scoring rules, verified against the rule books.

Each test names the law/rule it checks so the suite doubles as documentation of
the implemented rule set.
"""

from __future__ import annotations

import pytest

from smartgym.sports.badminton import BadmintonMatch
from smartgym.sports.tennis import TennisMatch


def play(scorer, sequence):
    for winner in sequence:
        scorer.award_point(winner)
    return scorer


# --------------------------------------------------------------------------- #
# Badminton (BWF Laws 7, 9, 11)
# --------------------------------------------------------------------------- #
class TestBadminton:
    def test_game_to_21(self):
        """Law 7.4.1: a game is won by the first side to 21 points."""
        m = play(BadmintonMatch(["P1", "P2"]), ["P1"] * 20)
        assert not m.match_over
        assert m.games["P1"] == 0
        m.award_point("P1")
        assert m.games["P1"] == 1
        assert m.game_scores == [(21, 0)]

    def test_must_win_by_two(self):
        """Law 7.4.2: at 20-all a 2-point lead is required."""
        m = BadmintonMatch(["P1", "P2"])
        for _ in range(20):
            m.award_point("P1")
            m.award_point("P2")
        assert m.points == {"P1": 20, "P2": 20}
        m.award_point("P1")
        assert m.games["P1"] == 0, "21-20 must not win the game"
        m.award_point("P1")
        assert m.games["P1"] == 1, "22-20 wins the game"

    def test_30_point_cap(self):
        """Law 7.4.3: at 29-all the next point wins."""
        m = BadmintonMatch(["P1", "P2"])
        for _ in range(29):
            m.award_point("P1")
            m.award_point("P2")
        m.award_point("P1")
        assert m.games["P1"] == 1
        assert m.game_scores == [(30, 29)]

    def test_best_of_three_match(self):
        """Law 7.3: the match is the best of three games."""
        m = BadmintonMatch(["P1", "P2"], best_of=3)
        for _ in range(2):
            for _ in range(21):
                m.award_point("P1")
        assert m.match_over and m.winner == "P1"
        assert m.games == {"P1": 2, "P2": 0}

    def test_rally_point_scoring_server_follows_winner(self):
        """Law 7.2: the side that wins a rally serves next."""
        m = BadmintonMatch(["P1", "P2"], first_server="P1")
        assert m.server == "P1"
        m.award_point("P2")
        assert m.server == "P2"
        m.award_point("P2")
        assert m.server == "P2"
        m.award_point("P1")
        assert m.server == "P1"

    def test_service_court_parity(self):
        """Law 9.2: even score serves from the right court, odd from the left."""
        m = BadmintonMatch(["P1", "P2"], first_server="P1")
        side0, low0 = m.serve_target()
        assert m.points["P1"] == 0
        m.award_point("P1")
        assert m.points["P1"] == 1
        side1, low1 = m.serve_target()
        # same side of the court, opposite service court
        assert side0 is side1
        assert low0 is not low1

    def test_service_court_crosses_the_net(self):
        """Law 9.2: the serve is diagonal, into the opposite half."""
        m = BadmintonMatch(["P1", "P2"], first_server="P1")
        target_side, low_band = m.serve_target()
        assert target_side is not m.server_side

    def test_change_of_ends_between_games(self):
        """Law 11.3: sides change at the end of each game."""
        m = BadmintonMatch(["P1", "P2"])
        before = dict(m.side_of)
        for _ in range(21):
            m.award_point("P1")
        assert m.side_of["P1"] is not before["P1"]

    def test_deciding_game_interval(self):
        """Law 11.5: in the deciding game ends change at 11 points."""
        m = BadmintonMatch(["P1", "P2"], best_of=3)
        for _ in range(21):
            m.award_point("P1")
        for _ in range(21):
            m.award_point("P2")
        assert m.current_game == 3
        sides_before = dict(m.side_of)
        for _ in range(11):
            m.award_point("P1")
        assert m.side_of["P1"] is not sides_before["P1"]


# --------------------------------------------------------------------------- #
# Tennis (ITF Rules of Tennis, Rule 5 / 10 / 14)
# --------------------------------------------------------------------------- #
class TestTennis:
    def test_four_points_win_a_game(self):
        m = play(TennisMatch(["A", "B"]), ["A", "A", "A"])
        assert m.points_display() == "40-0"
        m.award_point("A")
        assert m.games["A"] == 1
        assert m.points == {"A": 0, "B": 0}

    def test_deuce_and_advantage(self):
        """Rule 5: at 40-40 two consecutive points are needed."""
        m = play(TennisMatch(["A", "B"]), ["A", "A", "A", "B", "B", "B"])
        assert m.points_display() == "40-40 (deuce)"
        m.award_point("A")
        assert m.points_display() == "AD-40"
        m.award_point("B")
        assert m.points_display() == "40-40 (deuce)"
        m.award_point("B")
        m.award_point("B")
        assert m.games["B"] == 1

    def test_no_ad_scoring(self):
        m = play(TennisMatch(["A", "B"], no_ad=True), ["A", "A", "A", "B", "B", "B"])
        assert m.points_display() == "40-40 (deciding point)"
        m.award_point("A")
        assert m.games["A"] == 1, "no-ad: one point decides at deuce"

    def test_set_at_six_with_two_game_lead(self):
        m = TennisMatch(["A", "B"])
        for _ in range(5):
            for _ in range(4):
                m.award_point("A")
            for _ in range(4):
                m.award_point("B")
        assert m.games == {"A": 5, "B": 5}
        for _ in range(4):
            m.award_point("A")
        assert m.games["A"] == 6 and m.set_scores == []

    def test_tiebreak_at_six_all(self):
        m = TennisMatch(["A", "B"])
        for _ in range(6):
            for _ in range(4):
                m.award_point("A")
            for _ in range(4):
                m.award_point("B")
        assert m.in_tiebreak
        for _ in range(7):
            m.award_point("A")
        assert m.set_scores == [(7, 6)]
        assert m.sets_won["A"] == 1

    def test_tiebreak_serve_rotation(self):
        """Rule 5: first point by the due server, then two points each."""
        m = TennisMatch(["A", "B"])
        for _ in range(6):
            for _ in range(4):
                m.award_point("A")
            for _ in range(4):
                m.award_point("B")
        assert m.in_tiebreak
        first_server = m.server          # serves point 1
        servers = [first_server]
        for _ in range(5):
            m.award_point("A")
            servers.append(m.server)     # server of the next point
        other = "B" if first_server == "A" else "A"
        # point 1: first; points 2-3: other; points 4-5: first; point 6: other
        assert servers[0] == first_server
        assert servers[1] == other and servers[2] == other
        assert servers[3] == first_server and servers[4] == first_server
        assert servers[5] == other

    def test_match_best_of_three(self):
        m = TennisMatch(["A", "B"], best_of=3)
        for _ in range(2):
            for _ in range(6):
                for _ in range(4):
                    m.award_point("A")
        assert m.match_over and m.winner == "A"

    def test_serve_alternates_each_game(self):
        m = TennisMatch(["A", "B"], first_server="A")
        assert m.server == "A"
        for _ in range(4):
            m.award_point("A")
        assert m.server == "B", "the receiver serves the next game"

    def test_serve_court_parity(self):
        """Rule 14: serve from the right court when the point total is even."""
        m = TennisMatch(["A", "B"])
        side_even, low_even = m.serve_target()
        m.award_point("A")
        side_odd, low_odd = m.serve_target()
        assert side_even is side_odd
        assert low_even is not low_odd


class TestScorerInvariants:
    def test_unknown_player_rejected(self):
        with pytest.raises(ValueError):
            BadmintonMatch(["P1", "P2"]).award_point("P3")
        with pytest.raises(ValueError):
            TennisMatch(["A", "B"]).award_point("C")

    def test_snapshot_is_json_serialisable(self):
        import json

        m = BadmintonMatch(["P1", "P2"])
        m.award_point("P1")
        json.dumps(m.snapshot(1.0, 30).as_dict())

    def test_match_over_ignores_further_points(self):
        m = BadmintonMatch(["P1", "P2"])
        for _ in range(21):
            m.award_point("P1")
        for _ in range(21):
            m.award_point("P1")
        assert m.match_over and m.winner == "P1"
        assert m.games == {"P1": 2, "P2": 0}
