"""Tests for the live prediction engine's timing discipline.

The engine's value depends on two guarantees: a prediction is issued before
its pitch is thrown, and the engine can prove it. These tests pin both.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from pitch_prediction.live.context import PregameContext
from pitch_prediction.live.engine import LivePredictionEngine, parse_timecode
from tests.test_live_features import (
    BATTER_A,
    PITCHER,
    _pitch,
    _play,
    _snapshot,
)


class StubModel:
    """Always predicts a slider, so outcomes are deterministic."""

    classes_ = np.array(["CH", "FF", "SL"])

    def predict(self, features):
        return np.array(["SL"] * len(features))

    def predict_proba(self, features):
        return np.tile(np.array([0.2, 0.3, 0.5]), (len(features), 1))


def _context() -> PregameContext:
    return PregameContext(
        pitcher_id=PITCHER,
        season=2026,
        game_date=date(2026, 8, 20),
        pitcher_days_since_prev_game=4.0,
    )


def _engine(clock=None, log_path: Path | None = None) -> LivePredictionEngine:
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return LivePredictionEngine(
        model=StubModel(),
        context=_context(),
        pitcher_name="Test Pitcher",
        log_path=log_path,
        **kwargs,
    )


def _in_progress(events: list[dict]) -> list[dict]:
    return [
        _play(0, batter=BATTER_A, event_type=None, complete=False, events=events)
    ]


class PendingPredictionTests(unittest.TestCase):
    def test_predicts_the_pitch_that_has_not_been_thrown(self) -> None:
        engine = _engine()
        # No pitch thrown yet: the prediction is for the first pitch.
        engine.observe(_snapshot(_in_progress([])))
        self.assertIsNotNone(engine.pending)
        self.assertEqual(engine.pending.pitch_number_of_ab, 1)
        self.assertEqual(engine.pending.predicted_pitch_type, "SL")
        self.assertAlmostEqual(engine.pending.confidence, 0.5)
        self.assertEqual(engine.pending.balls, 0)
        self.assertEqual(engine.stats.predictions_made, 1)
        self.assertEqual(engine.stats.resolved, 0)

    def test_unchanged_state_does_not_reissue_a_prediction(self) -> None:
        engine = _engine()
        snapshot = _snapshot(_in_progress([]))
        engine.observe(snapshot)
        engine.observe(snapshot)
        engine.observe(snapshot)
        self.assertEqual(engine.stats.predictions_made, 1)
        self.assertEqual(engine.stats.superseded, 0)

    def test_probabilities_cover_the_model_classes(self) -> None:
        engine = _engine()
        engine.observe(_snapshot(_in_progress([])))
        self.assertEqual(
            set(engine.pending.probabilities), {"CH", "FF", "SL"}
        )


class ResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thrown = _pitch(
            1, call="C", pitch_type="FF", speed=95.0, zone=5, balls=0, strikes=1
        )

    def test_scores_the_standing_prediction_against_the_thrown_pitch(self) -> None:
        engine = _engine()
        engine.observe(_snapshot(_in_progress([])))
        resolved = engine.observe(_snapshot(_in_progress([self.thrown])))

        self.assertEqual(len(resolved), 1)
        prediction = resolved[0]
        self.assertEqual(prediction.predicted_pitch_type, "SL")
        self.assertEqual(prediction.actual_pitch_type, "FF")
        self.assertFalse(prediction.correct)
        self.assertEqual(engine.stats.resolved, 1)
        self.assertEqual(engine.stats.correct, 0)

    def test_a_correct_prediction_is_counted(self) -> None:
        engine = _engine()
        engine.observe(_snapshot(_in_progress([])))
        slider = _pitch(
            1, call="C", pitch_type="SL", speed=86.0, zone=5, balls=0, strikes=1
        )
        resolved = engine.observe(_snapshot(_in_progress([slider])))
        self.assertTrue(resolved[0].correct)
        self.assertEqual(engine.stats.correct, 1)

    def test_a_pitch_is_never_scored_twice(self) -> None:
        engine = _engine()
        engine.observe(_snapshot(_in_progress([])))
        after = _snapshot(_in_progress([self.thrown]))
        engine.observe(after)
        self.assertEqual(engine.observe(after), [])
        self.assertEqual(engine.stats.resolved, 1)

    def test_a_pitch_with_no_standing_prediction_is_not_scored(self) -> None:
        # The engine started mid-at-bat, so the first pitch it sees was thrown
        # before it could predict anything. It must not claim credit for it.
        engine = _engine()
        resolved = engine.observe(_snapshot(_in_progress([self.thrown])))
        self.assertEqual(resolved, [])
        self.assertEqual(engine.stats.resolved, 0)
        # It does predict the next pitch.
        self.assertIsNotNone(engine.pending)
        self.assertEqual(engine.pending.pitch_number_of_ab, 2)

    def test_new_count_supersedes_the_previous_prediction(self) -> None:
        engine = _engine()
        engine.observe(_snapshot(_in_progress([])))
        engine.observe(_snapshot(_in_progress([self.thrown])))
        # Pitch 2 is now pending at 0-1 rather than 0-0.
        self.assertEqual(engine.pending.pitch_number_of_ab, 2)
        self.assertEqual(engine.pending.strikes, 1)


class TimingAuditTests(unittest.TestCase):
    def test_prediction_before_the_pitch_is_marked_in_time(self) -> None:
        # The stub clock sits well before the fixture's pitch start times.
        clock = lambda: parse_timecode("20260820_175900")  # noqa: E731
        engine = _engine(clock=clock)
        engine.observe(_snapshot(_in_progress([])))
        thrown = _pitch(
            1, call="C", pitch_type="FF", speed=95.0, zone=5, balls=0, strikes=1
        )
        resolved = engine.observe(_snapshot(_in_progress([thrown])))

        prediction = resolved[0]
        self.assertEqual(prediction.timing, "before_pitch")
        self.assertGreater(prediction.prediction_lead_seconds, 0)
        self.assertEqual(engine.stats.before_pitch, 1)
        self.assertEqual(engine.stats.late, 0)

    def test_prediction_after_the_pitch_is_marked_late(self) -> None:
        clock = lambda: parse_timecode("20260820_190000")  # noqa: E731
        engine = _engine(clock=clock)
        engine.observe(_snapshot(_in_progress([])))
        thrown = _pitch(
            1, call="C", pitch_type="SL", speed=86.0, zone=5, balls=0, strikes=1
        )
        resolved = engine.observe(_snapshot(_in_progress([thrown])))

        prediction = resolved[0]
        self.assertEqual(prediction.timing, "late")
        self.assertLess(prediction.prediction_lead_seconds, 0)
        self.assertEqual(engine.stats.late, 1)
        # A late prediction is excluded from the honest accuracy figure even
        # though it happened to be correct.
        self.assertTrue(prediction.correct)
        self.assertIsNone(engine.stats.honest_accuracy(engine.history))

    def test_honest_accuracy_counts_only_in_time_predictions(self) -> None:
        clock = lambda: parse_timecode("20260820_175900")  # noqa: E731
        engine = _engine(clock=clock)
        engine.observe(_snapshot(_in_progress([])))
        slider = _pitch(
            1, call="C", pitch_type="SL", speed=86.0, zone=5, balls=0, strikes=1
        )
        engine.observe(_snapshot(_in_progress([slider])))
        self.assertEqual(engine.stats.honest_accuracy(engine.history), 1.0)


class PredictionLogTests(unittest.TestCase):
    def test_resolved_predictions_are_appended_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "nested" / "log.jsonl"
            engine = _engine(log_path=log_path)
            engine.observe(_snapshot(_in_progress([])))
            thrown = _pitch(
                1, call="C", pitch_type="FF", speed=95.0, zone=5, balls=0, strikes=1
            )
            engine.observe(_snapshot(_in_progress([thrown])))

            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["predicted_pitch_type"], "SL")
            self.assertEqual(record["actual_pitch_type"], "FF")
            self.assertIn("predicted_at_utc", record)
            self.assertIn("probabilities", record)


if __name__ == "__main__":
    unittest.main()
