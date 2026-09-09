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
from pitch_prediction.live.feature_sets import PREDICT_AHEAD_EXCLUSIONS
from pitch_prediction.model import PitchModelTrainer
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


def _context(prior: dict[str, float] | None = None) -> PregameContext:
    return PregameContext(
        pitcher_id=PITCHER,
        season=2026,
        game_date=date(2026, 8, 20),
        pitcher_days_since_prev_game=4.0,
        pitch_type_prior=prior or {},
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


def _ahead_engine(prior=None, clock=None) -> LivePredictionEngine:
    """An engine using a model that ignores the previous pitch's measurements."""

    kwargs = {"clock": clock} if clock else {}
    return LivePredictionEngine(
        model=StubModel(),
        context=_context(prior or {"SL": 0.5, "FF": 0.3, "CH": 0.2}),
        pitcher_name="Test Pitcher",
        trainer=PitchModelTrainer(exclude_features=PREDICT_AHEAD_EXCLUSIONS),
        **kwargs,
    )


class PredictAheadTests(unittest.TestCase):
    """Predicting the pitch after next, by enumerating the states it may face.

    The feed publishes a pitch about as long after it is thrown as the gap to
    the next one, so a predictor that waits for it is usually too late. With
    the previous pitch's measurements excluded from the model, the remaining
    unknowns are enumerable and can be scored in advance.
    """

    def test_speculation_is_off_without_a_compatible_model(self) -> None:
        # A model that uses the previous pitch's measurements cannot be run
        # ahead, because those cannot be enumerated.
        self.assertFalse(_engine().speculate)
        self.assertTrue(_ahead_engine().speculate)

    def test_candidate_counts_cover_ball_strike_and_two_strike_foul(self) -> None:
        engine = _ahead_engine()
        self.assertEqual(sorted(engine._candidate_counts(0, 0)), [(0, 1), (1, 0)])
        # With two strikes a foul leaves the count alone.
        self.assertEqual(sorted(engine._candidate_counts(1, 2)), [(1, 2), (2, 2)])
        # A fourth ball would end the plate appearance, so it is not enumerated.
        self.assertEqual(engine._candidate_counts(3, 1), [(3, 2)])
        self.assertEqual(engine._candidate_counts(3, 2), [(3, 2)])

    def test_candidate_pitch_types_come_from_the_repertoire(self) -> None:
        # The cutoff is deliberately generous: a missing candidate costs a
        # fallback prediction and its headroom, while a surplus one costs
        # about a millisecond. A pitch at 0.5% of the mix still gets one.
        engine = _ahead_engine(prior={"SL": 0.5, "FF": 0.45, "EP": 0.005})
        self.assertEqual(sorted(engine._candidate_pitch_types()), ["EP", "FF", "SL"])

    def test_a_vanishingly_rare_pitch_gets_no_candidate(self) -> None:
        engine = _ahead_engine(prior={"SL": 0.6, "FF": 0.399, "PO": 0.001})
        self.assertEqual(sorted(engine._candidate_pitch_types()), ["FF", "SL"])

    def test_the_repertoire_is_capped(self) -> None:
        prior = {name: 1 / 12 for name in
                 ("FF", "SI", "SL", "ST", "CU", "KC", "CH", "FS", "FC", "SV",
                  "EP", "KN")}
        engine = _ahead_engine(prior=prior)
        self.assertEqual(len(engine._candidate_pitch_types()), 8)

    def test_candidates_are_scored_before_the_current_pitch_is_thrown(self) -> None:
        engine = _ahead_engine()
        engine.observe(_snapshot(_in_progress([])))
        # Pitch 1 is pending; pitch 2's candidate states are already scored.
        self.assertTrue(engine._speculative)
        for key in engine._speculative:
            self.assertEqual(key[1], 2)  # all candidates are for pitch 2

    def test_the_realized_state_reuses_its_precomputed_prediction(self) -> None:
        clock_times = iter(["20260820_180000", "20260820_180030"])
        current = {"value": "20260820_180000"}
        engine = _ahead_engine(clock=lambda: parse_timecode(current["value"]))

        engine.observe(_snapshot(_in_progress([])))
        precomputed = dict(engine._speculative)
        self.assertTrue(precomputed)

        # Time moves on, then the feed reveals a called strike on a fastball.
        current["value"] = "20260820_180030"
        thrown = _pitch(
            1, call="C", pitch_type="FF", speed=95.0, zone=5, balls=0, strikes=1
        )
        engine.observe(_snapshot(_in_progress([thrown])))

        pending = engine.pending
        self.assertIsNotNone(pending)
        self.assertEqual((pending.balls, pending.strikes), (0, 1))
        self.assertTrue(pending.predicted_ahead)
        self.assertEqual(engine.stats.predicted_ahead, 1)
        # It keeps the earlier timestamp, which is the whole point.
        self.assertEqual(pending.predicted_at_utc[:19], "2026-08-20T18:00:00")

    def test_an_unenumerated_state_still_gets_a_fresh_prediction(self) -> None:
        # A plate appearance ending is deliberately not enumerated, so the new
        # batter's first pitch is predicted the ordinary way.
        engine = _ahead_engine()
        engine.observe(_snapshot(_in_progress([])))
        walk = [
            _play(0, batter=BATTER_A, event_type="walk", events=[
                _pitch(n, call="B", pitch_type="FF", speed=95.0, zone=13,
                       balls=n, strikes=0) for n in range(1, 5)
            ]),
            _play(1, batter=PITCHER, event_type=None, complete=False, events=[]),
        ]
        engine.observe(_snapshot(walk))
        self.assertIsNotNone(engine.pending)
        self.assertFalse(engine.pending.predicted_ahead)

    def test_candidates_are_discarded_once_a_pitch_lands(self) -> None:
        engine = _ahead_engine()
        engine.observe(_snapshot(_in_progress([])))
        thrown = _pitch(
            1, call="C", pitch_type="FF", speed=95.0, zone=5, balls=0, strikes=1
        )
        engine.observe(_snapshot(_in_progress([thrown])))
        # The surviving candidates belong to the new state, not the old one.
        for key in engine._speculative:
            self.assertEqual(key[1], 3)


    def test_the_candidate_key_covers_every_state_a_candidate_assumes(self) -> None:
        """Candidates hold bases, outs and score fixed, so the key must too.

        A candidate copies the current row and varies only the count and the
        previous pitch type. If a steal, pickoff or wild pitch moves anything
        else, no candidate should match, because serving one would mean
        serving a prediction computed from a state that never happened.
        """

        import pandas as pd

        engine = _ahead_engine()
        base = pd.Series(
            {
                "balls": 1,
                "strikes": 1,
                "pitch_type_of_prev_pitch": "FF",
                "outs_when_up": 1,
                "on_1b": None,
                "on_2b": None,
                "on_3b": None,
                "bat_score": 2,
                "fld_score": 3,
            }
        )
        original = engine._candidate_key((4, 3), base)

        for field, changed in (
            ("outs_when_up", 2),
            ("on_1b", 12345),
            ("on_2b", 12345),
            ("on_3b", 12345),
            ("bat_score", 3),
            ("fld_score", 4),
            ("balls", 2),
            ("strikes", 2),
            ("pitch_type_of_prev_pitch", "SL"),
        ):
            moved = base.copy()
            moved[field] = changed
            self.assertNotEqual(
                engine._candidate_key((4, 3), moved),
                original,
                f"a change to {field} must invalidate the candidate",
            )

        # Missing and NaN are the same absence, so they must key alike.
        with_nan = base.copy()
        with_nan["on_1b"] = float("nan")
        self.assertEqual(engine._candidate_key((4, 3), with_nan), original)


if __name__ == "__main__":
    unittest.main()