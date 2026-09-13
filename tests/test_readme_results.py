"""Tests for the README showcase game.

The daily workflow regenerates the README results block unattended, so the
selection rules that decide which outing gets published are asserted here
rather than checked by eye after the fact.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_readme_results import (
    daily_frame,
    load_window,
    repertoire_breadth,
    select_showcase,
    showcase_markdown,
)


def _predictions(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a prediction log from (predicted, actual) pairs."""

    return pd.DataFrame(
        {
            "pitch_number_of_game": range(1, len(pairs) + 1),
            "inning": [1 + index // 15 for index in range(len(pairs))],
            "count": ["0-0"] * len(pairs),
            "model_prediction": [predicted for predicted, _ in pairs],
            "actual_pitch": [actual for _, actual in pairs],
            "model_correct": [
                predicted == actual for predicted, actual in pairs
            ],
        }
    )


class LoadWindowTests(unittest.TestCase):
    """The window is N evaluated game dates, not N calendar days."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def _history(self, dates: list[str]) -> Path:
        history = pd.DataFrame(
            {
                "game_date": dates,
                "game_pk": range(len(dates)),
                "pitcher_id": range(len(dates)),
                "pitch_count": [100] * len(dates),
                "model_correct": [60] * len(dates),
                "baseline_correct": [30] * len(dates),
            }
        )

        path = self.root / "history.csv"
        history.to_csv(path, index=False)
        return path

    def test_takes_the_most_recent_n_dates(self) -> None:
        dates = [f"2026-07-{day:02d}" for day in range(1, 21)]

        window, start, end = load_window(self._history(dates), 5)

        self.assertEqual(len(window), 5)
        self.assertEqual(f"{start:%Y-%m-%d}", "2026-07-16")
        self.assertEqual(f"{end:%Y-%m-%d}", "2026-07-20")

    def test_calendar_gaps_do_not_shrink_the_sample(self) -> None:
        # Ten dates either side of a month-long break: a 30-day window would
        # return half of them, a 20-date window must return all twenty.
        dates = [f"2026-06-{day:02d}" for day in range(1, 11)] + [
            f"2026-08-{day:02d}" for day in range(1, 11)
        ]

        window, start, end = load_window(self._history(dates), 20)

        self.assertEqual(len(window), 20)
        self.assertGreater((end - start).days, 30)

    def test_requesting_more_dates_than_exist_returns_all(self) -> None:
        dates = [f"2026-07-{day:02d}" for day in range(1, 6)]

        window, _, _ = load_window(self._history(dates), 30)

        self.assertEqual(len(window), 5)

    def test_all_pitcher_games_on_a_date_are_kept(self) -> None:
        # Three dates, two pitcher-games each; asking for two dates must
        # return four rows, not two.
        dates = ["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02",
                 "2026-07-03", "2026-07-03"]

        window, _, _ = load_window(self._history(dates), 2)

        self.assertEqual(len(window), 4)
        self.assertEqual(len(daily_frame(window)), 2)


class RepertoireBreadthTests(unittest.TestCase):
    def test_constant_prediction_counts_as_one_type(self) -> None:
        predictions = _predictions([("FF", "FF")] * 20)

        self.assertEqual(repertoire_breadth(predictions), 1)

    def test_types_hit_only_once_do_not_count(self) -> None:
        # FF clears the floor; SL is correct a single time and should not.
        predictions = _predictions(
            [("FF", "FF")] * 10 + [("SL", "SL")] + [("SL", "FF")] * 3
        )

        self.assertEqual(repertoire_breadth(predictions), 1)

    def test_repeatedly_correct_types_count(self) -> None:
        predictions = _predictions(
            [("FF", "FF")] * 5 + [("SL", "SL")] * 4 + [("CH", "CH")] * 3
        )

        self.assertEqual(repertoire_breadth(predictions), 3)


class SelectShowcaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def _write(self, name: str, predictions: pd.DataFrame) -> str:
        relative = f"predictions/{name}.csv"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(path, index=False)
        return relative

    def _window(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_varied_game_beats_a_more_accurate_monotone_game(self) -> None:
        monotone = self._write(
            "monotone",
            _predictions([("FF", "FF")] * 90),
        )

        varied = self._write(
            "varied",
            _predictions(
                [("FF", "FF")] * 40
                + [("SL", "SL")] * 20
                + [("CH", "CH")] * 20
                + [("FF", "SL")] * 10
            ),
        )

        window = self._window(
            [
                {
                    "pitcher_name": "Monotone",
                    "pitch_count": 90,
                    "model_accuracy": 0.99,
                    "predictions_path": monotone,
                },
                {
                    "pitcher_name": "Varied",
                    "pitch_count": 90,
                    "model_accuracy": 0.80,
                    "predictions_path": varied,
                },
            ]
        )

        game, predictions = select_showcase(window, self.root)

        self.assertIsNotNone(game)
        self.assertEqual(game["pitcher_name"], "Varied")
        self.assertEqual(len(predictions), 90)

    def test_short_outings_are_not_eligible(self) -> None:
        varied = self._write(
            "short",
            _predictions(
                [("FF", "FF")] * 10
                + [("SL", "SL")] * 10
                + [("CH", "CH")] * 10
            ),
        )

        window = self._window(
            [
                {
                    "pitcher_name": "Short",
                    "pitch_count": 30,
                    "model_accuracy": 0.99,
                    "predictions_path": varied,
                }
            ]
        )

        self.assertEqual(select_showcase(window, self.root), (None, None))

    def test_missing_prediction_file_is_skipped(self) -> None:
        varied = self._write(
            "present",
            _predictions(
                [("FF", "FF")] * 40
                + [("SL", "SL")] * 25
                + [("CH", "CH")] * 25
            ),
        )

        window = self._window(
            [
                {
                    "pitcher_name": "Pruned",
                    "pitch_count": 95,
                    "model_accuracy": 0.99,
                    "predictions_path": "predictions/gone.csv",
                },
                {
                    "pitcher_name": "Present",
                    "pitch_count": 90,
                    "model_accuracy": 0.70,
                    "predictions_path": varied,
                },
            ]
        )

        game, _ = select_showcase(window, self.root)

        self.assertIsNotNone(game)
        self.assertEqual(game["pitcher_name"], "Present")


class ShowcaseMarkdownTests(unittest.TestCase):
    def _game(self) -> pd.Series:
        return pd.Series(
            {
                "pitcher_name": "Test Pitcher",
                "game_date": pd.Timestamp("2026-08-24"),
                "opponent": "Seattle Mariners",
                "pitch_count": 100,
                "model_accuracy": 0.6,
                "baseline_accuracy": 0.3,
                "relative_improvement": 1.0,
            }
        )

    def test_table_has_one_row_per_pitch_of_the_outing(self) -> None:
        predictions = _predictions(
            [("FF", "FF")] * 40 + [("SL", "SL")] * 30 + [("CH", "CH")] * 30
        )

        lines = showcase_markdown(self._game(), predictions)

        rows = [
            line
            for line in lines
            if line.startswith("| ") and not line.startswith("| Pitch")
        ]

        self.assertEqual(len(rows), len(predictions))

    def test_check_marks_match_the_published_accuracy(self) -> None:
        # The whole outing is published, so a reader can count the table and
        # arrive at the headline number themselves.
        predictions = _predictions(
            [("FF", "FF")] * 60 + [("FF", "SL")] * 40
        )

        text = "\n".join(showcase_markdown(self._game(), predictions))

        self.assertEqual(text.count("&#10003;"), 60)
        self.assertEqual(text.count("&#10007;"), 40)

    def test_header_reports_the_whole_outing(self) -> None:
        predictions = _predictions(
            [("FF", "FF")] * 40 + [("SL", "SL")] * 30 + [("CH", "CH")] * 30
        )

        text = "\n".join(showcase_markdown(self._game(), predictions))

        self.assertIn("60.0% correct on 100 pitches", text)
        self.assertIn("30.0% baseline", text)

    def test_legend_names_every_code_in_the_table(self) -> None:
        predictions = _predictions(
            [("FF", "FF")] * 40 + [("SL", "SL")] * 30 + [("CH", "CH")] * 30
        )

        lines = showcase_markdown(self._game(), predictions)

        expected = set(predictions["model_prediction"]) | set(
            predictions["actual_pitch"]
        )

        legend = lines[-1]

        for code in expected:
            self.assertIn(f"`{code}`", legend)

        # and nothing the reader will not see in the table
        self.assertEqual(legend.count("`") // 2, len(expected))


if __name__ == "__main__":
    unittest.main()
