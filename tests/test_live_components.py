"""Tests for the live view's markup.

The components return plain HTML, so the accessibility and layout rules the
view depends on can be asserted directly rather than eyeballed.
"""

from __future__ import annotations

import re
import unittest

from dashboard.components import (
    GameView,
    diamond_html,
    dots_html,
    last_pitch_html,
    prediction_html,
    probability_bars_html,
    scoreboard_html,
    tile_html,
)


def _view(**overrides) -> GameView:
    defaults = dict(
        away="STL",
        home="CIN",
        away_score=3,
        home_score=4,
        inning=6,
        topbot="Bot",
        balls=2,
        strikes=1,
        outs=2,
        status="In Progress",
        on_1b=True,
        on_2b=False,
        on_3b=True,
    )
    defaults.update(overrides)
    return GameView(**defaults)


class ProbabilityBarTests(unittest.TestCase):
    def test_only_the_predicted_pitch_carries_the_accent(self) -> None:
        # Emphasis form: one accent, the rest recede. A single series needs no
        # legend, and no categorical CVD pair is ever put on screen.
        html = probability_bars_html({"SI": 0.5, "SL": 0.3, "CU": 0.2}, "SI")
        self.assertEqual(html.count("is-predicted"), 1)
        accent_row = html.split('<div class="bar-row">')[1]
        self.assertIn("SI", accent_row)

    def test_bars_scale_to_the_leading_probability(self) -> None:
        html = probability_bars_html({"SI": 0.4, "SL": 0.2}, "SI")
        widths = [float(w) for w in re.findall(r"width:([\d.]+)%", html)]
        self.assertEqual(widths[0], 100.0)
        self.assertAlmostEqual(widths[1], 50.0, places=1)

    def test_bars_are_sorted_high_to_low(self) -> None:
        html = probability_bars_html({"CU": 0.1, "SI": 0.6, "SL": 0.3}, "SI")
        keys = re.findall(r'class="bar-key">(\w+)<', html)
        self.assertEqual(keys, ["SI", "SL", "CU"])

    def test_a_near_certain_bar_keeps_its_label_inside_the_track(self) -> None:
        # A label placed past the track's end would overflow the card, so a
        # bar that nearly fills the track carries its value inside instead.
        html = probability_bars_html({"FF": 0.94, "SL": 0.06}, "FF")
        leading = html.split('<div class="bar-row">')[1]
        self.assertIn("right:0", leading)
        self.assertNotIn("left:100", leading)

    def test_every_bar_is_labelled_with_its_value(self) -> None:
        html = probability_bars_html({"SI": 0.5, "SL": 0.3, "CU": 0.2}, "SI")
        self.assertEqual(html.count('class="bar-value"'), 3)
        for expected in ("50%", "30%", "20%"):
            self.assertIn(expected, html)

    def test_empty_probabilities_do_not_raise(self) -> None:
        self.assertEqual(probability_bars_html({}, "SI"), "")


class PredictionCardTests(unittest.TestCase):
    def test_hero_shows_the_predicted_pitch(self) -> None:
        html = prediction_html(
            predicted="SI",
            confidence=0.41,
            probabilities={"SI": 0.41, "SL": 0.3},
            batter="Dane Myers",
            balls=2,
            strikes=1,
            pitch_number=78,
        )
        self.assertIn('class="hero-value">SI<', html)
        self.assertIn("41% confidence", html)
        self.assertIn("Dane Myers", html)
        self.assertIn("2-1", html)

    def test_placeholder_when_no_pitch_is_pending(self) -> None:
        html = prediction_html(
            predicted=None,
            confidence=None,
            probabilities=None,
            batter=None,
            balls=None,
            strikes=None,
            pitch_number=None,
        )
        self.assertIn("Waiting for the pitcher", html)
        self.assertNotIn("bar-fill", html)


class StatusTests(unittest.TestCase):
    def test_result_never_relies_on_color_alone(self) -> None:
        # good and critical sit only dE 4.1 apart under deuteranopia, so the
        # dot must always be paired with a word.
        correct = last_pitch_html(
            batter="A", predicted="SI", actual="SI", correct=True,
            timing="before_pitch", lead_seconds=11.6,
        )
        missed = last_pitch_html(
            batter="A", predicted="SL", actual="CU", correct=False,
            timing="before_pitch", lead_seconds=9.0,
        )
        self.assertIn("Correct", correct)
        self.assertIn("pill-dot", correct)
        self.assertIn("Missed", missed)
        self.assertIn("pill-dot", missed)

    def test_lead_time_is_stated_in_plain_words(self) -> None:
        html = last_pitch_html(
            batter="A", predicted="SI", actual="SI", correct=True,
            timing="before_pitch", lead_seconds=11.6,
        )
        self.assertIn("Predicted 11.6s before the pitch", html)
        # The raw state name should not leak into the view.
        self.assertNotIn("before_pitch", html)

    def test_a_late_prediction_says_it_is_excluded(self) -> None:
        html = last_pitch_html(
            batter="A", predicted="SI", actual="SI", correct=True,
            timing="late", lead_seconds=-2.0,
        )
        self.assertIn("excluded from accuracy", html)

    def test_placeholder_before_any_pitch_is_scored(self) -> None:
        html = last_pitch_html(
            batter=None, predicted=None, actual=None, correct=None,
            timing=None, lead_seconds=None,
        )
        self.assertIn("No pitch scored yet", html)


class ScoreboardTests(unittest.TestCase):
    def test_count_dots_fill_to_the_current_count(self) -> None:
        self.assertEqual(dots_html(2, 3).count("dot on"), 2)
        self.assertEqual(dots_html(2, 3).count('class="dot"'), 1)
        self.assertEqual(dots_html(0, 2).count("dot on"), 0)

    def test_occupied_bases_are_filled_and_empty_ones_are_not(self) -> None:
        html = diamond_html(_view(on_1b=True, on_2b=False, on_3b=True))
        self.assertEqual(html.count('fill="var(--text-primary)"'), 2)
        self.assertEqual(html.count('fill="none"'), 1)

    def test_scoreboard_carries_both_teams_and_the_situation(self) -> None:
        html = scoreboard_html(_view())
        for expected in ("STL", "CIN", "Bot 6", "balls", "strikes", "outs",
                         "In Progress"):
            self.assertIn(expected, html)

    def test_scoreboard_renders_an_empty_game_state(self) -> None:
        html = scoreboard_html(
            _view(inning=0, balls=0, strikes=0, outs=0, status="Scheduled",
                  on_1b=False, on_2b=False, on_3b=False)
        )
        self.assertIn("Scheduled", html)
        self.assertNotIn("dot on", html)


class TileTests(unittest.TestCase):
    def test_tile_follows_the_label_value_sub_contract(self) -> None:
        html = tile_html("Pitches scored", "47", "19 correct")
        self.assertIn('class="tile-label">Pitches scored<', html)
        self.assertIn('class="tile-value">47<', html)
        self.assertIn('class="tile-sub">19 correct<', html)


if __name__ == "__main__":
    unittest.main()
