"""Post-game prediction replay and accuracy logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from .clients import Starter
from .model import PitchModelTrainer
from .pipeline import DailyStarterPipeline


@dataclass(frozen=True)
class PostgameReplayResult:
    game_pk: int
    game_date: str

    pitcher_id: int
    pitcher_name: str

    pitch_count: int

    model_accuracy: float
    baseline_accuracy: float

    # Absolute improvement:
    # model accuracy - baseline accuracy
    accuracy_over_baseline: float

    # Relative improvement:
    # (model - baseline) / baseline
    relative_improvement: float | None

    baseline_strategy: str

    model_path: str
    predictions_path: str
    summary_path: str


class PostgameReplayer:
    """Replay a completed game using a frozen pre-game model."""

    def __init__(
        self,
        pipeline: DailyStarterPipeline,
        model_trainer: PitchModelTrainer,
        output_root: Path,
    ) -> None:

        self.pipeline = pipeline
        self.model_trainer = model_trainer
        self.output_root = output_root

    def replay(
        self,
        starter: Starter,
        game_date: date,
    ) -> PostgameReplayResult:

        # ====================================================
        # DOWNLOAD DATA INCLUDING COMPLETED GAME
        # ====================================================

        download = (
            self.pipeline._download_pitcher(
                starter,
                history_through=game_date,
            )
        )

        cleaning = (
            self.pipeline._clean_pitcher(
                starter,
                Path(
                    download.path
                ),
            )
        )

        features = (
            self.pipeline._engineer_pitcher(
                starter,
                Path(
                    cleaning.cleaned_path
                ),
            )
        )

        kg4 = pd.read_csv(
            features.kg4_path,
            low_memory=False,
        )

        kg4["game_date"] = (
            pd.to_datetime(
                kg4["game_date"],
                errors="raise",
            )
        )

        # ====================================================
        # PRE-GAME HISTORY
        # ====================================================

        pregame = kg4[
            kg4["game_date"]
            .dt.date
            < game_date
        ].copy()

        if pregame.empty:
            raise ValueError(
                f"No pre-game history exists for "
                f"{starter.pitcher_name}."
            )

        # ====================================================
        # TARGET GAME
        # ====================================================

        game_rows = kg4[
            pd.to_numeric(
                kg4["game_pk"],
                errors="coerce",
            )
            == starter.game_pk
        ].copy()

        if game_rows.empty:
            raise ValueError(
                "The completed game was not found in "
                "the Baseball Savant data. "
                "Savant may not have published it yet."
            )

        sort_columns = [
            column
            for column in [
                "at_bat_number_of_game",
                "pitch_number_of_ab",
                "pitch_number_of_game",
            ]
            if column
            in game_rows.columns
        ]

        if sort_columns:

            game_rows = (
                game_rows
                .sort_values(
                    sort_columns,
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        else:

            game_rows = (
                game_rows
                .reset_index(
                    drop=True
                )
            )

        # ====================================================
        # FROZEN PRE-GAME MODEL
        # ====================================================

        model_dir = (
            self.output_root
            / "models"
            / game_date.isoformat()
            / "pitchers"
        )

        model_path = (
            model_dir
            / f"{starter.pitcher_id}.joblib"
        )

        # Historical backfill:
        # if the pre-game daily pipeline was never run, create
        # the model now using ONLY rows before the target game.

        if not model_path.exists():

            print(
                "No frozen pre-game model found."
            )

            print(
                "Creating a historical pre-game model "
                "using only data before the game..."
            )

            self.model_trainer.train(
                pregame,
                pitcher_id=(
                    starter.pitcher_id
                ),
                pitcher_name=(
                    starter.pitcher_name
                ),
                reference_season=(
                    game_date.year
                ),
                model_dir=model_dir,
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model was not created at "
                f"{model_path}"
            )

        model = joblib.load(
            model_path
        )

        # ====================================================
        # RANDOM FOREST PREDICTIONS
        # ====================================================

        X_game = (
            self.model_trainer
            ._features(
                game_rows
            )
        )

        if hasattr(
            model,
            "feature_names_in_",
        ):

            X_game = X_game.reindex(
                columns=list(
                    model.feature_names_in_
                )
            )

        predictions = model.predict(
            X_game
        )

        actual = (
            game_rows["pitch_type"]
            .astype(str)
            .to_numpy()
        )

        model_correct = (
            predictions
            == actual
        )

        model_accuracy = float(
            accuracy_score(
                actual,
                predictions,
            )
        )

        # ====================================================
        # MODEL PROBABILITIES
        # ====================================================

        probability_frame = pd.DataFrame(
            index=game_rows.index
        )

        model_confidence = np.full(
            len(game_rows),
            np.nan,
        )

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = (
                model.predict_proba(
                    X_game
                )
            )

            classes = list(
                model.classes_
            )

            model_confidence = (
                probabilities.max(
                    axis=1
                )
            )

            probability_frame = (
                pd.DataFrame(
                    probabilities,
                    columns=[
                        f"prob_{pitch_type}"
                        for pitch_type
                        in classes
                    ],
                )
            )

        # ====================================================
        # STRATIFIED BASELINE
        # ====================================================
        #
        # This uses sklearn's DummyClassifier.
        #
        # It learns ONLY the pre-game pitch proportions.
        #
        # If history is:
        #
        # FF 50%
        # SL 30%
        # CH 20%
        #
        # it randomly guesses according to approximately those
        # proportions.
        #
        # It does not use game context.
        # ====================================================

        X_pregame = (
            self.model_trainer
            ._features(
                pregame
            )
        )

        y_pregame = (
            pregame["pitch_type"]
            .astype(str)
        )

        baseline_model = DummyClassifier(
            strategy="stratified",
            random_state=(
                self.model_trainer
                .random_state
            ),
        )

        baseline_model.fit(
            X_pregame,
            y_pregame,
        )

        baseline_X_game = (
            X_game.reindex(
                columns=(
                    X_pregame.columns
                )
            )
        )

        baseline_predictions = (
            baseline_model.predict(
                baseline_X_game
            )
        )

        baseline_correct = (
            baseline_predictions
            == actual
        )

        baseline_accuracy = float(
            accuracy_score(
                actual,
                baseline_predictions,
            )
        )

        accuracy_over_baseline = float(
            model_accuracy
            - baseline_accuracy
        )

        relative_improvement = (
            self.model_trainer
            .relative_improvement(
                model_accuracy,
                baseline_accuracy,
            )
        )

        baseline_distribution = {
            str(pitch_type):
                float(probability)
            for pitch_type, probability
            in zip(
                baseline_model.classes_,
                baseline_model.class_prior_,
            )
        }

        # ====================================================
        # PITCH-BY-PITCH LOG
        # ====================================================

        log_columns = [
            "game_pk",
            "game_date",
            "pitcher",
            "batter",
            "inning",
            "inning_topbot",
            "at_bat_number_of_game",
            "pitch_number_of_ab",
            "pitch_number_of_game",
            "balls",
            "strikes",
            "count",
            "outs_when_up",
            "pitch_type_of_prev_pitch",
        ]

        available_log_columns = [
            column
            for column in log_columns
            if column
            in game_rows.columns
        ]

        prediction_log = (
            game_rows[
                available_log_columns
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        prediction_log[
            "pitcher_name"
        ] = starter.pitcher_name

        prediction_log[
            "actual_pitch"
        ] = actual

        prediction_log[
            "model_prediction"
        ] = predictions

        prediction_log[
            "model_confidence"
        ] = model_confidence

        prediction_log[
            "model_correct"
        ] = model_correct

        prediction_log[
            "baseline_prediction"
        ] = baseline_predictions

        prediction_log[
            "baseline_correct"
        ] = baseline_correct

        probability_frame = (
            probability_frame
            .reset_index(
                drop=True
            )
        )

        prediction_log = pd.concat(
            [
                prediction_log,
                probability_frame,
            ],
            axis=1,
        )

        # ====================================================
        # SAVE
        # ====================================================

        replay_dir = (
            self.output_root
            / "predictions"
            / "postgame"
            / game_date.isoformat()
        )

        replay_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        base_name = (
            f"{starter.game_pk}_"
            f"{starter.pitcher_id}"
        )

        predictions_path = (
            replay_dir
            / f"{base_name}.csv"
        )

        summary_path = (
            replay_dir
            / f"{base_name}.json"
        )

        prediction_log.to_csv(
            predictions_path,
            index=False,
        )

        result = PostgameReplayResult(
            game_pk=starter.game_pk,
            game_date=(
                game_date.isoformat()
            ),

            pitcher_id=(
                starter.pitcher_id
            ),

            pitcher_name=(
                starter.pitcher_name
            ),

            pitch_count=len(
                game_rows
            ),

            model_accuracy=(
                model_accuracy
            ),

            baseline_accuracy=(
                baseline_accuracy
            ),

            accuracy_over_baseline=(
                accuracy_over_baseline
            ),

            relative_improvement=(
                relative_improvement
            ),

            baseline_strategy=(
                "stratified"
            ),

            model_path=str(
                model_path
            ),

            predictions_path=str(
                predictions_path
            ),

            summary_path=str(
                summary_path
            ),
        )

        summary = {
            **asdict(result),

            "model_accuracy_percent":
                model_accuracy
                * 100,

            "baseline_accuracy_percent":
                baseline_accuracy
                * 100,

            "accuracy_over_baseline_pp":
                accuracy_over_baseline
                * 100,

            "relative_improvement_percent":
                (
                    relative_improvement
                    * 100
                    if relative_improvement
                    is not None
                    else None
                ),

            "baseline_distribution":
                baseline_distribution,

            "generated_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "important_note":
                (
                    "The model was trained only on "
                    "data before this game. The "
                    "stratified baseline was trained "
                    "only on the pitcher's pre-game "
                    "pitch proportions."
                ),
        }

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return result