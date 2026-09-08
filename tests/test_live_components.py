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
    game_card_html,
    hero_metric_html,
    last_pitch_html,
    pitch_table_html,
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


class ScoresGridTests(unittest.TestCase):
    def test_card_emphasises_the_leading_team(self) -> None:
        html = game_card_html(
            away="STL", home="CIN", away_score=3, home_score=4,
            status="Bot 6", is_live=True, detail="2-1, 2 out",
            pitchers=["Sonny Gray", "Hunter Greene"],
        )
        # The trailing team recedes; the leader is not marked, so exactly one
        # row carries the trailing style.
        self.assertEqual(html.count("is-trailing"), 1)
        away_row = html.split('class="game-row')[1]
        self.assertIn("is-trailing", away_row)

    def test_a_live_game_is_marked_live(self) -> None:
        live = game_card_html(
            away="STL", home="CIN", away_score=1, home_score=0, status="Top 3",
            is_live=True, detail="", pitchers=[],
        )
        final = game_card_html(
            away="STL", home="CIN", away_score=1, home_score=0, status="Final",
            is_live=False, detail="", pitchers=[],
        )
        self.assertIn("is-live", live)
        self.assertNotIn("is-live", final)

    def test_a_scheduled_game_shows_dashes_not_zeros(self) -> None:
        # A game that has not started has no score; showing 0-0 would imply it
        # is underway and tied.
        html = game_card_html(
            away="STL", home="CIN", away_score=None, home_score=None,
            status="Scheduled", is_live=False, detail="7:05 PM ET",
            pitchers=["Sonny Gray"],
        )
        self.assertIn(">-<", html)
        self.assertNotIn("is-trailing", html)

    def test_missing_model_note_is_surfaced(self) -> None:
        html = game_card_html(
            away="STL", home="CIN", away_score=0, home_score=0, status="Final",
            is_live=False, detail="", pitchers=["A"],
            note="No pre-game model for this date",
        )
        self.assertIn("No pre-game model", html)


class PitchTableTests(unittest.TestCase):
    def _rows(self, count: int) -> list[dict]:
        return [
            {
                "inning": "Top 1",
                "count": "0-0",
                "predicted": "SI",
                "actual": "SI" if index % 2 == 0 else "SL",
                "correct": index % 2 == 0,
            }
            for index in range(count)
        ]

    def test_result_is_a_glyph_and_a_color_never_color_alone(self) -> None:
        html = pitch_table_html(self._rows(2))
        self.assertIn("&#10003;", html)  # check
        self.assertIn("&#10007;", html)  # cross
        self.assertIn("mark good", html)
        self.assertIn("mark bad", html)

    def test_newest_row_is_highlighted(self) -> None:
        html = pitch_table_html(self._rows(3))
        self.assertEqual(html.count("is-latest"), 1)
        self.assertLess(html.index("is-latest"), html.index("</tbody>"))

    def test_table_is_capped_so_it_cannot_outgrow_the_card(self) -> None:
        html = pitch_table_html(self._rows(40), limit=12)
        self.assertEqual(html.count("<tr"), 13)  # 12 rows plus the header

    def test_empty_state_explains_itself(self) -> None:
        html = pitch_table_html([])
        self.assertIn("No pitch has been thrown", html)
        self.assertNotIn("<tbody>", html)

    def test_a_pitch_with_no_actual_yet_renders_a_dash(self) -> None:
        html = pitch_table_html(
            [{"inning": "Top 1", "count": "0-0", "predicted": "SI",
              "actual": None, "correct": None}]
        )
        self.assertIn("&ndash;", html)


class HeadlineMetricTests(unittest.TestCase):
    def test_metric_follows_the_label_value_sub_contract(self) -> None:
        html = hero_metric_html("Pitch prediction accuracy", "40.1%", "5,750 pitches")
        self.assertIn('class="metric-label">Pitch prediction accuracy<', html)
        self.assertIn('class="metric-value">40.1%<', html)
        self.assertIn('class="metric-sub">5,750 pitches<', html)


if __name__ == "__main__":
    unittest.main()