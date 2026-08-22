"""Architecture and modeling safety tests.

These tests cover the most important assumptions in the current project:

1. Continuous repertoire weighting behaves correctly.
2. Mild recency and repertoire weights combine correctly.
3. The target column never becomes a model feature.
4. Historical evaluation remains chronological and leakage-safe.
5. The trained sklearn Pipeline can be saved, reloaded, and used.
6. The saved model has the expected Random Forest architecture.
7. Post-game replay trains only on games BEFORE the target game.
8. Replay predicts exactly the target game's pitches.
9. Prediction logs and accuracy metrics are internally consistent.

All tests use synthetic data and small Random Forests so they remain
fast and do not require MLB/Baseball Savant network access.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from pitch_prediction.model import PitchModelTrainer
from pitch_prediction.postgame_replay import PostgameReplayer
from pitch_prediction.repertoire import (
    RepertoireSettings,
    continuous_repertoire_weights,
)


# ============================================================
# TEST HELPERS
# ============================================================


def fast_trainer() -> PitchModelTrainer:
    """
    Return the real PitchModelTrainer with a tiny Random Forest.

    Production uses 800 trees.

    Tests do not need 800 trees to verify architecture, so we use
    only 8 to keep pytest fast.
    """

    trainer = PitchModelTrainer(
        n_estimators=8,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
        random_state=42,
    )

    # Synthetic tests contain far fewer than 300 pitches.
    # Lowering these requirements is only for testing the logic.
    trainer.repertoire_settings = RepertoireSettings(
        min_recent_pitches=1,
        min_older_season_pitches=1,
    )

    return trainer


def small_repertoire_settings() -> RepertoireSettings:
    """Settings that allow tiny synthetic repertoire datasets."""

    return RepertoireSettings(
        min_recent_pitches=1,
        min_older_season_pitches=1,
    )


def repertoire_frame(
    prior_ff: int,
    prior_sl: int,
    recent_ff: int,
    recent_sl: int,
) -> pd.DataFrame:
    """Build a simple two-season repertoire dataset."""

    rows = []

    for pitch_type, count in [
        ("FF", prior_ff),
        ("SL", prior_sl),
    ]:
        for _ in range(count):
            rows.append(
                {
                    "season": 2024,
                    "pitch_type": pitch_type,
                }
            )

    for pitch_type, count in [
        ("FF", recent_ff),
        ("SL", recent_sl),
    ]:
        for _ in range(count):
            rows.append(
                {
                    "season": 2025,
                    "pitch_type": pitch_type,
                }
            )

    return pd.DataFrame(rows)


def make_game(
    *,
    game_pk: int,
    game_date: str,
    pitch_types: list[str],
    pitcher_id: int = 999,
) -> pd.DataFrame:
    """
    Create synthetic KG4-like rows for one game.

    The columns are intentionally simple, but include both numeric
    and categorical model features plus columns used by replay.
    """

    rows = []

    previous_pitch = None

    for index, pitch_type in enumerate(
        pitch_types,
        start=1,
    ):

        pitch_in_ab = (
            (index - 1) % 4
        ) + 1

        at_bat_number = (
            (index - 1) // 4
        ) + 1

        balls = min(
            (pitch_in_ab - 1) % 4,
            3,
        )

        strikes = min(
            (pitch_in_ab - 1) % 3,
            2,
        )

        rows.append(
            {
                # Target / metadata
                "pitch_type": pitch_type,
                "game_pk": game_pk,
                "game_date": game_date,
                "pitcher": pitcher_id,

                # Replay metadata
                "batter": 1000 + at_bat_number,
                "inning": 1 + ((at_bat_number - 1) // 3),
                "inning_topbot": "Top",
                "at_bat_number_of_game": at_bat_number,
                "pitch_number_of_ab": pitch_in_ab,
                "pitch_number_of_game": index,

                # Model inputs
                "balls": balls,
                "strikes": strikes,
                "count": f"{balls}-{strikes}",
                "outs_when_up": (at_bat_number - 1) % 3,
                "stand": "R" if at_bat_number % 2 else "L",
                "p_throws": "R",

                # Leakage-safe previous-pitch feature
                "pitch_type_of_prev_pitch": previous_pitch,

                # Generic numeric feature representing some
                # already-engineered KG4 information.
                "synthetic_numeric_feature":
                    float(index % 7),
            }
        )

        previous_pitch = pitch_type

    return pd.DataFrame(rows)


def make_training_dataset() -> pd.DataFrame:
    """
    Build five chronologically ordered games.

    The final 2026 game contains CU pitches that did not appear
    in earlier seasons. With a 20% test fraction, this final game
    becomes the held-out evaluation game.
    """

    frames = [
        make_game(
            game_pk=101,
            game_date="2024-05-01",
            pitch_types=[
                "FF",
                "SL",
            ] * 8,
        ),
        make_game(
            game_pk=102,
            game_date="2024-06-01",
            pitch_types=[
                "FF",
                "FF",
                "SL",
                "FF",
            ] * 4,
        ),
        make_game(
            game_pk=103,
            game_date="2025-05-01",
            pitch_types=[
                "FF",
                "SL",
                "SL",
                "FF",
            ] * 4,
        ),
        make_game(
            game_pk=104,
            game_date="2025-06-01",
            pitch_types=[
                "SL",
                "FF",
            ] * 8,
        ),

        # Held-out newest game.
        # CU intentionally appears only here.
        make_game(
            game_pk=105,
            game_date="2026-05-01",
            pitch_types=[
                "CU",
                "FF",
                "CU",
                "SL",
            ] * 4,
        ),
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# REPERTOIRE TESTS
# ============================================================


def test_stable_repertoire_keeps_unit_weights():
    """
    If pitch usage is unchanged between seasons, repertoire
    weighting should not modify observations.
    """

    data = repertoire_frame(
        prior_ff=50,
        prior_sl=50,
        recent_ff=50,
        recent_sl=50,
    )

    weights, diagnostics = (
        continuous_repertoire_weights(
            data,
            settings=(
                small_repertoire_settings()
            ),
        )
    )

    assert np.allclose(
        weights,
        1.0,
    )

    assert all(
        item.status == "stable"
        for item in diagnostics
    )


def test_declining_pitch_is_downweighted_and_increasing_pitch_is_boosted():
    """
    FF falls from 80% usage to 20%.
    SL rises from 20% usage to 80%.

    Therefore:
    - old FF rows should receive less than 1.0
    - recent SL rows should receive greater than 1.0
    """

    data = repertoire_frame(
        prior_ff=80,
        prior_sl=20,
        recent_ff=20,
        recent_sl=80,
    )

    weights, diagnostics = (
        continuous_repertoire_weights(
            data,
            settings=(
                small_repertoire_settings()
            ),
        )
    )

    old_ff_mask = (
        (data["season"] == 2024)
        &
        (data["pitch_type"] == "FF")
    )

    recent_sl_mask = (
        (data["season"] == 2025)
        &
        (data["pitch_type"] == "SL")
    )

    assert (
        weights[
            old_ff_mask.to_numpy()
        ].mean()
        < 1.0
    )

    assert (
        weights[
            recent_sl_mask.to_numpy()
        ].mean()
        > 1.0
    )

    statuses = {
        item.pitch_type:
            item.status
        for item in diagnostics
    }

    assert statuses["FF"] == "declining"

    assert statuses["SL"] in {
        "increasing",
        "emerging",
    }


def test_repertoire_weight_bounds_are_respected():
    """
    Repertoire weights must remain inside the frozen experimental
    bounds of 0.15 to 1.50.
    """

    data = repertoire_frame(
        prior_ff=99,
        prior_sl=1,
        recent_ff=1,
        recent_sl=99,
    )

    weights, _ = (
        continuous_repertoire_weights(
            data,
            settings=(
                small_repertoire_settings()
            ),
        )
    )

    assert weights.min() >= 0.15

    assert weights.max() <= 1.50


def test_one_season_repertoire_is_noop():
    """
    Repertoire change cannot be inferred from only one season.
    """

    data = pd.DataFrame(
        {
            "season":
                [2025] * 20,

            "pitch_type":
                ["FF"] * 10
                + ["SL"] * 10,
        }
    )

    weights, diagnostics = (
        continuous_repertoire_weights(
            data,
            settings=(
                small_repertoire_settings()
            ),
        )
    )

    assert np.allclose(
        weights,
        1.0,
    )

    assert diagnostics == []


def test_insufficient_recent_sample_is_noop():
    """
    Production defaults require at least 300 pitches in the
    latest season before changing repertoire weights.
    """

    data = repertoire_frame(
        prior_ff=200,
        prior_sl=200,
        recent_ff=20,
        recent_sl=20,
    )

    weights, diagnostics = (
        continuous_repertoire_weights(
            data,
            settings=RepertoireSettings(),
        )
    )

    assert np.allclose(
        weights,
        1.0,
    )

    assert diagnostics == []


# ============================================================
# COMBINED WEIGHTING TESTS
# ============================================================


def test_final_training_weight_equals_recency_times_repertoire():
    """
    The production rule must remain:

        final weight
            =
        mild recency
            ×
        continuous repertoire
    """

    trainer = fast_trainer()

    data = repertoire_frame(
        prior_ff=80,
        prior_sl=20,
        recent_ff=20,
        recent_sl=80,
    )

    data["game_date"] = pd.to_datetime(
        data["season"]
        .astype(str)
        + "-06-01"
    )

    recency_weights = (
        trainer._season_sample_weights(
            data,
            reference_season=2025,
        )
    )

    repertoire_weights, _ = (
        trainer._repertoire_sample_weights(
            data
        )
    )

    final_weights, _ = (
        trainer._training_sample_weights(
            data,
            reference_season=2025,
        )
    )

    assert np.allclose(
        final_weights,
        recency_weights
        * repertoire_weights,
    )


def test_recency_weights_favor_newer_seasons():
    """Newer seasons should receive greater training weight."""

    trainer = fast_trainer()

    data = pd.DataFrame(
        {
            "season": [
                2022,
                2023,
                2024,
                2025,
            ]
        }
    )

    weights = (
        trainer._season_sample_weights(
            data,
            reference_season=2025,
        )
    )

    assert (
        weights[0]
        < weights[1]
        < weights[2]
        < weights[3]
    )

    assert weights[-1] == pytest.approx(
        1.0
    )


# ============================================================
# MODEL FEATURE / ARCHITECTURE TESTS
# ============================================================


def test_target_and_metadata_are_not_model_features():
    """
    pitch_type is the answer we are predicting.

    It must never appear inside X.
    """

    trainer = fast_trainer()

    data = make_training_dataset()

    data["season"] = (
        pd.to_datetime(
            data["game_date"]
        ).dt.year
    )

    X = trainer._features(
        data
    )

    forbidden = {
        "pitch_type",
        "game_pk",
        "game_date",
        "pitcher",
        "season",
    }

    assert forbidden.isdisjoint(
        set(X.columns)
    )

    assert (
        "synthetic_numeric_feature"
        in X.columns
    )


def test_model_training_saves_reloadable_sklearn_pipeline(
    tmp_path: Path,
):
    """
    The production artifact must:

    KG4
      -> preprocessing
      -> Random Forest
      -> .joblib
      -> reload
      -> predict
    """

    trainer = fast_trainer()

    data = make_training_dataset()

    result = trainer.train(
        data,
        pitcher_id=999,
        pitcher_name="Test Pitcher",
        reference_season=2026,
        model_dir=tmp_path,
    )

    model_path = Path(
        result.model_path
    )

    metrics_path = Path(
        result.metrics_path
    )

    assert model_path.exists()

    assert metrics_path.exists()

    model = joblib.load(
        model_path
    )

    assert (
        "preprocessor"
        in model.named_steps
    )

    assert (
        "classifier"
        in model.named_steps
    )

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    assert isinstance(
        classifier,
        RandomForestClassifier,
    )

    # Verify test configuration was actually used.
    assert classifier.n_estimators == 8

    X = trainer._features(
        data.head(5)
    )

    predictions = model.predict(
        X
    )

    assert len(predictions) == 5

    assert set(predictions).issubset(
        set(
            data["pitch_type"]
        )
    )


# ============================================================
# LEAKAGE / CHRONOLOGICAL EVALUATION TEST
# ============================================================


def test_historical_evaluation_does_not_use_future_repertoire(
    tmp_path: Path,
):
    """
    The newest game contains CU, which never appeared previously.

    Because evaluation weighting must use TRAINING DATA ONLY:

    - evaluation repertoire diagnostics must NOT know about CU
    - production diagnostics MAY know about CU because production
      trains on all supplied history.
    """

    trainer = fast_trainer()

    data = make_training_dataset()

    result = trainer.train(
        data,
        pitcher_id=999,
        pitcher_name="Leakage Test",
        reference_season=2026,
        model_dir=tmp_path,
    )

    metrics = json.loads(
        Path(
            result.metrics_path
        ).read_text(
            encoding="utf-8"
        )
    )

    evaluation = metrics[
        "evaluation"
    ]

    weighting = metrics[
        "weighting"
    ]

    # With five games and a 20% test fraction,
    # the newest game should be the single test game.
    assert evaluation["train_games"] == 4
    assert evaluation["test_games"] == 1

    # Training data ends in 2025.
    assert (
        evaluation[
            "evaluation_reference_season"
        ]
        == 2025
    )

    evaluation_pitch_types = {
        item["pitch_type"]
        for item
        in weighting[
            "repertoire"
        ][
            "evaluation_diagnostics"
        ]
    }

    production_pitch_types = {
        item["pitch_type"]
        for item
        in weighting[
            "repertoire"
        ][
            "production_diagnostics"
        ]
    }

    # CU exists only in the held-out 2026 game.
    assert "CU" not in evaluation_pitch_types

    # Production uses all provided history,
    # so CU may now appear.
    assert "CU" in production_pitch_types


# ============================================================
# POST-GAME REPLAY TEST HELPERS
# ============================================================


class FakePostgamePipeline:
    """
    Minimal stand-in for DailyStarterPipeline.

    It does no downloading or cleaning.

    Every stage simply points the replayer to the synthetic KG4 CSV.

    This lets us test the entire replay architecture without
    internet access.
    """

    def __init__(
        self,
        kg4_path: Path,
    ) -> None:

        self.kg4_path = kg4_path

        self.requested_history_through = None

        self.download_called = False
        self.clean_called = False
        self.engineer_called = False

    def _download_pitcher(
        self,
        starter,
        history_through,
    ):

        self.download_called = True

        self.requested_history_through = (
            history_through
        )

        return SimpleNamespace(
            path=str(
                self.kg4_path
            )
        )

    def _clean_pitcher(
        self,
        starter,
        path,
    ):

        self.clean_called = True

        return SimpleNamespace(
            cleaned_path=str(
                self.kg4_path
            )
        )

    def _engineer_pitcher(
        self,
        starter,
        path,
    ):

        self.engineer_called = True

        return SimpleNamespace(
            kg4_path=str(
                self.kg4_path
            )
        )


def make_postgame_dataset(
    target_game_pk: int,
) -> pd.DataFrame:
    """
    Two pre-game games plus one completed target game.
    """

    pregame_1 = make_game(
        game_pk=501,
        game_date="2026-08-01",
        pitch_types=[
            "FF",
            "SL",
            "FF",
            "FF",
        ] * 4,
        pitcher_id=12345,
    )

    pregame_2 = make_game(
        game_pk=502,
        game_date="2026-08-10",
        pitch_types=[
            "SL",
            "FF",
            "SL",
            "FF",
        ] * 4,
        pitcher_id=12345,
    )

    target = make_game(
        game_pk=target_game_pk,
        game_date="2026-08-20",
        pitch_types=[
            "FF",
            "SL",
            "SL",
            "FF",
            "FF",
            "SL",
            "FF",
            "SL",
        ],
        pitcher_id=12345,
    )

    # Deliberately mix row order.
    # The replay code should isolate and sort the target game.
    combined = pd.concat(
        [
            target.iloc[4:],
            pregame_1,
            target.iloc[:4],
            pregame_2,
        ],
        ignore_index=True,
    )

    return combined


# ============================================================
# FULL POST-GAME ARCHITECTURE TEST
# ============================================================


def test_postgame_replay_end_to_end_without_network(
    tmp_path: Path,
):
    """
    This is the main architecture smoke test.

    It verifies:

    synthetic completed Statcast data
        ->
    fake download/clean/feature stages
        ->
    pre-game history isolation
        ->
    frozen Random Forest training
        ->
    target-game prediction
        ->
    CSV log
        ->
    JSON summary

    It also checks that the target game NEVER enters training.
    """

    target_game_pk = 900001

    game_date = date(
        2026,
        8,
        20,
    )

    data = make_postgame_dataset(
        target_game_pk
    )

    kg4_path = (
        tmp_path
        / "synthetic_kg4.csv"
    )

    data.to_csv(
        kg4_path,
        index=False,
    )

    fake_pipeline = (
        FakePostgamePipeline(
            kg4_path
        )
    )

    trainer = fast_trainer()

    output_root = (
        tmp_path
        / "output"
    )

    replayer = PostgameReplayer(
        pipeline=fake_pipeline,
        model_trainer=trainer,
        output_root=output_root,
    )

    starter = SimpleNamespace(
        pitcher_id=12345,
        pitcher_name="Architecture Test Pitcher",
        game_pk=target_game_pk,
    )

    result = replayer.replay(
        starter=starter,
        game_date=game_date,
    )

    # --------------------------------------------------------
    # PIPELINE STAGES WERE CALLED
    # --------------------------------------------------------

    assert fake_pipeline.download_called
    assert fake_pipeline.clean_called
    assert fake_pipeline.engineer_called

    # Post-game download should include the completed game.
    assert (
        fake_pipeline.requested_history_through
        == game_date
    )

    # --------------------------------------------------------
    # ARTIFACTS EXIST
    # --------------------------------------------------------

    assert Path(
        result.model_path
    ).exists()

    assert Path(
        result.predictions_path
    ).exists()

    assert Path(
        result.summary_path
    ).exists()

    # --------------------------------------------------------
    # ONLY TARGET GAME WAS PREDICTED
    # --------------------------------------------------------

    target_rows = data[
        data["game_pk"]
        == target_game_pk
    ]

    prediction_log = pd.read_csv(
        result.predictions_path
    )

    assert (
        len(prediction_log)
        == len(target_rows)
    )

    assert result.pitch_count == len(
        target_rows
    )

    assert set(
        prediction_log[
            "game_pk"
        ]
    ) == {
        target_game_pk
    }

    # --------------------------------------------------------
    # TARGET GAME DID NOT ENTER TRAINING
    # --------------------------------------------------------

    model_metrics_path = (
        output_root
        / "models"
        / game_date.isoformat()
        / "pitchers"
        / "12345.json"
    )

    assert model_metrics_path.exists()

    model_metrics = json.loads(
        model_metrics_path.read_text(
            encoding="utf-8"
        )
    )

    expected_pregame_rows = data[
        pd.to_datetime(
            data["game_date"]
        ).dt.date
        < game_date
    ]

    assert (
        model_metrics[
            "production"
        ][
            "training_pitches"
        ]
        == len(
            expected_pregame_rows
        )
    )

    # Only the two earlier games should train the model.
    assert (
        model_metrics[
            "production"
        ][
            "training_games"
        ]
        == 2
    )

    # --------------------------------------------------------
    # LOG CORRECTNESS
    # --------------------------------------------------------

    expected_model_correct = (
        prediction_log[
            "actual_pitch"
        ]
        ==
        prediction_log[
            "model_prediction"
        ]
    )

    assert (
        prediction_log[
            "model_correct"
        ]
        .astype(bool)
        .tolist()
        ==
        expected_model_correct
        .tolist()
    )

    expected_baseline_correct = (
        prediction_log[
            "actual_pitch"
        ]
        ==
        prediction_log[
            "baseline_prediction"
        ]
    )

    assert (
        prediction_log[
            "baseline_correct"
        ]
        .astype(bool)
        .tolist()
        ==
        expected_baseline_correct
        .tolist()
    )

    # --------------------------------------------------------
    # METRICS MATCH THE LOG
    # --------------------------------------------------------

    calculated_model_accuracy = float(
        expected_model_correct.mean()
    )

    calculated_baseline_accuracy = float(
        expected_baseline_correct.mean()
    )

    assert (
        result.model_accuracy
        ==
        pytest.approx(
            calculated_model_accuracy
        )
    )

    assert (
        result.baseline_accuracy
        ==
        pytest.approx(
            calculated_baseline_accuracy
        )
    )

    assert (
        result.accuracy_over_baseline
        ==
        pytest.approx(
            calculated_model_accuracy
            - calculated_baseline_accuracy
        )
    )

    # --------------------------------------------------------
    # SUMMARY JSON AGREES WITH RESULT
    # --------------------------------------------------------

    summary = json.loads(
        Path(
            result.summary_path
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        summary["pitch_count"]
        == result.pitch_count
    )

    assert (
        summary["model_accuracy"]
        ==
        pytest.approx(
            result.model_accuracy
        )
    )

    assert (
        summary["baseline_accuracy"]
        ==
        pytest.approx(
            result.baseline_accuracy
        )
    )


# ============================================================
# PROJECT ARCHITECTURE CONTRACT
# ============================================================


def test_core_project_architecture_contract():
    """
    Basic smoke test for the interfaces the current architecture
    depends on.

    If one of these methods disappears during a refactor, pytest
    tells us immediately that another component will break.
    """

    required_trainer_methods = [
        "train",
        "_features",
        "_build_pipeline",
        "_ordered_games",
        "_decay_factor",
        "_season_sample_weights",
        "_repertoire_sample_weights",
        "_training_sample_weights",
        "weighted_majority_pitch",
    ]

    for method_name in (
        required_trainer_methods
    ):

        assert hasattr(
            PitchModelTrainer,
            method_name,
        ), (
            f"PitchModelTrainer is missing "
            f"required method: "
            f"{method_name}"
        )

    assert hasattr(
        PostgameReplayer,
        "replay",
    )