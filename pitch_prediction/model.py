"""Per-pitcher Random Forest model training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .repertoire import (
    RepertoireSettings,
    continuous_repertoire_weights,
    diagnostics_to_dicts,
)


NON_FEATURE_COLUMNS = (
    "pitch_type",
    "game_pk",
    "game_date",
    "pitcher",
    "season",
)


@dataclass(frozen=True)
class PitchModelResult:
    """Summary returned after training one pitcher's model."""

    pitcher_id: int
    pitcher_name: str

    career_seasons: int
    career_games: int
    career_pitches: int

    train_accuracy: float
    test_accuracy: float

    baseline_strategy: str
    baseline_accuracy: float

    # Absolute improvement:
    # 0.75 - 0.50 = 0.25 = +25 percentage points
    accuracy_over_baseline: float

    # Relative improvement:
    # (0.75 - 0.50) / 0.50 = 0.50 = +50%
    relative_improvement: float | None

    evaluation_decay_factor: float
    production_decay_factor: float

    model_path: str
    metrics_path: str


class PitchModelTrainer:
    """Train and evaluate one Random Forest per pitcher."""

    def __init__(
        self,
        *,
        test_fraction: float = 0.20,

        # Mild recency weighting
        starting_decay_factor: float = 0.98,
        decay_per_extra_season: float = 0.01,
        minimum_decay_factor: float = 0.80,
        minimum_sample_weight: float = 0.10,

        # Random Forest
        n_estimators: int = 800,
        max_depth: int | None = 15,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        max_features: str | int | float | None = "log2",
        bootstrap: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:

        if not 0 < test_fraction < 1:
            raise ValueError(
                "test_fraction must be between 0 and 1."
            )

        self.test_fraction = test_fraction

        self.starting_decay_factor = (
            starting_decay_factor
        )

        self.decay_per_extra_season = (
            decay_per_extra_season
        )

        self.minimum_decay_factor = (
            minimum_decay_factor
        )

        self.minimum_sample_weight = (
            minimum_sample_weight
        )

        self.repertoire_settings = (
            RepertoireSettings()
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ========================================================
    # FEATURES
    # ========================================================

    def _features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        columns_to_drop = [
            column
            for column in NON_FEATURE_COLUMNS
            if column in data.columns
        ]

        return (
            data
            .drop(
                columns=columns_to_drop,
                errors="ignore",
            )
            .copy()
        )

    # ========================================================
    # MODEL PIPELINE
    # ========================================================

    def _build_pipeline(
        self,
        X: pd.DataFrame,
    ) -> Pipeline:

        categorical_columns = (
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            )
            .columns
            .tolist()
        )

        numeric_columns = [
            column
            for column in X.columns
            if column not in categorical_columns
        ]

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=-999,
                    ),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Missing",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        transformers = []

        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=(
                self.min_samples_split
            ),
            min_samples_leaf=(
                self.min_samples_leaf
            ),
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

    # ========================================================
    # GAME ORDER
    # ========================================================

    def _ordered_games(
        self,
        data: pd.DataFrame,
    ) -> list:

        required_columns = {
            "game_pk",
            "game_date",
        }

        missing = (
            required_columns
            - set(data.columns)
        )

        if missing:
            raise ValueError(
                "Game ordering requires columns: "
                f"{sorted(missing)}"
            )

        working = data[
            [
                "game_pk",
                "game_date",
            ]
        ].copy()

        working["game_date"] = (
            pd.to_datetime(
                working["game_date"],
                errors="raise",
            )
        )

        games = (
            working
            .dropna(
                subset=[
                    "game_pk",
                    "game_date",
                ]
            )
            .groupby(
                "game_pk",
                as_index=False,
            )["game_date"]
            .min()
            .sort_values(
                [
                    "game_date",
                    "game_pk",
                ],
                kind="stable",
            )
        )

        return (
            games["game_pk"]
            .tolist()
        )

    # ========================================================
    # RECENCY WEIGHTING
    # ========================================================

    def _decay_factor(
        self,
        total_career_seasons: int,
    ) -> float:

        if total_career_seasons < 1:
            raise ValueError(
                "total_career_seasons must be >= 1."
            )

        decay = (
            self.starting_decay_factor
            - self.decay_per_extra_season
            * (
                total_career_seasons
                - 1
            )
        )

        return float(
            max(
                self.minimum_decay_factor,
                decay,
            )
        )

    def _season_sample_weights(
        self,
        data: pd.DataFrame,
        *,
        reference_season: int,
    ) -> np.ndarray:

        if "season" in data.columns:

            seasons = (
                pd.to_numeric(
                    data["season"],
                    errors="raise",
                )
                .astype(int)
            )

        elif "game_date" in data.columns:

            seasons = (
                pd.to_datetime(
                    data["game_date"],
                    errors="raise",
                )
                .dt.year
            )

        else:
            raise ValueError(
                "Season weighting requires "
                "'season' or 'game_date'."
            )

        career_seasons = int(
            seasons.nunique()
        )

        decay = self._decay_factor(
            career_seasons
        )

        seasons_ago = (
            reference_season
            - seasons
        ).clip(
            lower=0
        )

        weights = np.power(
            decay,
            seasons_ago.to_numpy(
                dtype=float
            ),
        )

        weights = np.maximum(
            weights,
            self.minimum_sample_weight,
        )

        return weights.astype(
            float
        )

    # ========================================================
    # REPERTOIRE WEIGHTING
    # ========================================================

    def _repertoire_sample_weights(
        self,
        data: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        list[dict],
    ]:

        working = data.copy()

        if "season" not in working.columns:

            if "game_date" not in working.columns:
                raise ValueError(
                    "Repertoire weighting requires "
                    "'season' or 'game_date'."
                )

            working["season"] = (
                pd.to_datetime(
                    working["game_date"],
                    errors="raise",
                )
                .dt.year
            )

        (
            weights,
            diagnostics,
        ) = (
            continuous_repertoire_weights(
                working,
                settings=(
                    self.repertoire_settings
                ),
            )
        )

        return (
            weights,
            diagnostics_to_dicts(
                diagnostics
            ),
        )

    def _training_sample_weights(
        self,
        data: pd.DataFrame,
        *,
        reference_season: int,
    ) -> tuple[
        np.ndarray,
        list[dict],
    ]:

        recency_weights = (
            self._season_sample_weights(
                data,
                reference_season=(
                    reference_season
                ),
            )
        )

        (
            repertoire_weights,
            repertoire_diagnostics,
        ) = (
            self._repertoire_sample_weights(
                data
            )
        )

        final_weights = (
            recency_weights
            * repertoire_weights
        )

        return (
            final_weights.astype(float),
            repertoire_diagnostics,
        )

    # ========================================================
    # RELATIVE IMPROVEMENT
    # ========================================================

    @staticmethod
    def relative_improvement(
        model_accuracy: float,
        baseline_accuracy: float,
    ) -> float | None:

        if baseline_accuracy <= 0:
            return None

        return (
            model_accuracy
            - baseline_accuracy
        ) / baseline_accuracy

    # ========================================================
    # LEGACY MOST-FREQUENT HELPER
    # ========================================================
    #
    # Kept for compatibility with older tests/scripts.
    # It is no longer the baseline used for evaluation.
    # ========================================================

    def weighted_majority_pitch(
        self,
        data: pd.DataFrame,
        *,
        reference_season: int,
    ) -> str:

        if data.empty:
            raise ValueError(
                "Cannot calculate baseline "
                "from empty data."
            )

        weights, _ = (
            self._training_sample_weights(
                data,
                reference_season=(
                    reference_season
                ),
            )
        )

        weighted = pd.DataFrame(
            {
                "pitch_type":
                    data["pitch_type"]
                    .astype(str)
                    .to_numpy(),

                "weight":
                    weights,
            }
        )

        totals = (
            weighted
            .groupby("pitch_type")[
                "weight"
            ]
            .sum()
        )

        return str(
            totals.idxmax()
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def _feature_importance(
        self,
        model: Pipeline,
        *,
        top_n: int = 25,
    ) -> list[dict]:

        try:

            preprocessor = (
                model.named_steps[
                    "preprocessor"
                ]
            )

            classifier = (
                model.named_steps[
                    "classifier"
                ]
            )

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            importances = (
                classifier
                .feature_importances_
            )

            table = pd.DataFrame(
                {
                    "feature":
                        feature_names,

                    "importance":
                        importances,
                }
            )

            table = (
                table
                .sort_values(
                    "importance",
                    ascending=False,
                )
                .head(top_n)
            )

            return [
                {
                    "feature":
                        str(row.feature),

                    "importance":
                        float(
                            row.importance
                        ),
                }
                for row
                in table.itertuples(
                    index=False
                )
            ]

        except Exception:

            return []

    # ========================================================
    # SEASON DIAGNOSTICS
    # ========================================================

    def _season_weight_table(
        self,
        data: pd.DataFrame,
        *,
        reference_season: int,
    ) -> list[dict]:

        if "season" in data.columns:

            seasons = (
                pd.to_numeric(
                    data["season"],
                    errors="raise",
                )
                .astype(int)
            )

        else:

            seasons = (
                pd.to_datetime(
                    data["game_date"],
                    errors="raise",
                )
                .dt.year
            )

        career_seasons = int(
            seasons.nunique()
        )

        decay = self._decay_factor(
            career_seasons
        )

        rows = []

        for season in sorted(
            seasons.unique(),
            reverse=True,
        ):

            seasons_ago = max(
                0,
                reference_season
                - int(season),
            )

            weight = max(
                self.minimum_sample_weight,
                decay ** seasons_ago,
            )

            rows.append(
                {
                    "season":
                        int(season),

                    "seasons_ago":
                        int(seasons_ago),

                    "recency_weight":
                        float(weight),
                }
            )

        return rows

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        kg4: pd.DataFrame,
        *,
        pitcher_id: int,
        pitcher_name: str,
        reference_season: int,
        model_dir: Path,
    ) -> PitchModelResult:

        required = {
            "pitch_type",
            "game_pk",
            "game_date",
        }

        missing = (
            required
            - set(kg4.columns)
        )

        if missing:
            raise ValueError(
                "KG4 is missing required columns: "
                f"{sorted(missing)}"
            )

        data = (
            kg4
            .dropna(
                subset=[
                    "pitch_type",
                    "game_pk",
                    "game_date",
                ]
            )
            .copy()
        )

        if data.empty:
            raise ValueError(
                "No usable pitches remain."
            )

        data["game_date"] = (
            pd.to_datetime(
                data["game_date"],
                errors="raise",
            )
        )

        data["season"] = (
            data["game_date"]
            .dt.year
        )

        data["pitch_type"] = (
            data["pitch_type"]
            .astype(str)
        )

        game_order = (
            self._ordered_games(
                data
            )
        )

        if len(game_order) < 2:
            raise ValueError(
                "At least two games are required."
            )

        career_seasons = int(
            data["season"].nunique()
        )

        career_games = int(
            len(game_order)
        )

        career_pitches = int(
            len(data)
        )

        # ====================================================
        # CHRONOLOGICAL SPLIT
        # ====================================================

        n_test_games = max(
            1,
            math.ceil(
                career_games
                * self.test_fraction
            ),
        )

        n_test_games = min(
            n_test_games,
            career_games - 1,
        )

        train_game_ids = set(
            game_order[
                :-n_test_games
            ]
        )

        test_game_ids = set(
            game_order[
                -n_test_games:
            ]
        )

        train = data[
            data["game_pk"].isin(
                train_game_ids
            )
        ].copy()

        test = data[
            data["game_pk"].isin(
                test_game_ids
            )
        ].copy()

        if train.empty or test.empty:
            raise ValueError(
                "Chronological split produced "
                "an empty train or test set."
            )

        X_train = self._features(
            train
        )

        y_train = (
            train["pitch_type"]
            .astype(str)
        )

        X_test = self._features(
            test
        )

        y_test = (
            test["pitch_type"]
            .astype(str)
        )

        # ====================================================
        # TRAINING WEIGHTS — TRAIN DATA ONLY
        # ====================================================

        evaluation_reference_season = int(
            train["season"].max()
        )

        (
            evaluation_weights,
            evaluation_repertoire,
        ) = (
            self._training_sample_weights(
                train,
                reference_season=(
                    evaluation_reference_season
                ),
            )
        )

        evaluation_decay_factor = (
            self._decay_factor(
                int(
                    train["season"]
                    .nunique()
                )
            )
        )

        # ====================================================
        # RANDOM FOREST EVALUATION MODEL
        # ====================================================

        evaluation_model = (
            self._build_pipeline(
                X_train
            )
        )

        evaluation_model.fit(
            X_train,
            y_train,
            classifier__sample_weight=(
                evaluation_weights
            ),
        )

        train_predictions = (
            evaluation_model.predict(
                X_train
            )
        )

        test_predictions = (
            evaluation_model.predict(
                X_test
            )
        )

        train_accuracy = float(
            accuracy_score(
                y_train,
                train_predictions,
            )
        )

        test_accuracy = float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        )

        # ====================================================
        # STRATIFIED BASELINE
        # ====================================================
        #
        # This is sklearn's standard stratified DummyClassifier.
        #
        # It learns only the class proportions in y_train and
        # randomly guesses according to those proportions.
        #
        # Example:
        #
        # FF = 50%
        # SL = 30%
        # CH = 20%
        #
        # Baseline guesses approximately:
        #
        # 50% FF
        # 30% SL
        # 20% CH
        #
        # random_state makes results reproducible.
        # ====================================================

        baseline_model = DummyClassifier(
            strategy="stratified",
            random_state=(
                self.random_state
            ),
        )

        baseline_model.fit(
            X_train,
            y_train,
        )

        baseline_predictions = (
            baseline_model.predict(
                X_test
            )
        )

        baseline_accuracy = float(
            accuracy_score(
                y_test,
                baseline_predictions,
            )
        )

        accuracy_over_baseline = float(
            test_accuracy
            - baseline_accuracy
        )

        relative_improvement = (
            self.relative_improvement(
                test_accuracy,
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
        # PRODUCTION MODEL
        # ====================================================

        X_all = self._features(
            data
        )

        y_all = (
            data["pitch_type"]
            .astype(str)
        )

        (
            production_weights,
            production_repertoire,
        ) = (
            self._training_sample_weights(
                data,
                reference_season=(
                    reference_season
                ),
            )
        )

        production_decay_factor = (
            self._decay_factor(
                career_seasons
            )
        )

        production_model = (
            self._build_pipeline(
                X_all
            )
        )

        production_model.fit(
            X_all,
            y_all,
            classifier__sample_weight=(
                production_weights
            ),
        )

        # ====================================================
        # SAVE
        # ====================================================

        model_dir = Path(
            model_dir
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            model_dir
            / f"{pitcher_id}.joblib"
        )

        metrics_path = (
            model_dir
            / f"{pitcher_id}.json"
        )

        joblib.dump(
            production_model,
            model_path,
        )

        season_weights = (
            self._season_weight_table(
                data,
                reference_season=(
                    reference_season
                ),
            )
        )

        feature_importance = (
            self._feature_importance(
                production_model,
                top_n=25,
            )
        )

        result = PitchModelResult(
            pitcher_id=int(
                pitcher_id
            ),

            pitcher_name=str(
                pitcher_name
            ),

            career_seasons=(
                career_seasons
            ),

            career_games=(
                career_games
            ),

            career_pitches=(
                career_pitches
            ),

            train_accuracy=(
                train_accuracy
            ),

            test_accuracy=(
                test_accuracy
            ),

            baseline_strategy=(
                "stratified"
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

            evaluation_decay_factor=(
                evaluation_decay_factor
            ),

            production_decay_factor=(
                production_decay_factor
            ),

            model_path=str(
                model_path
            ),

            metrics_path=str(
                metrics_path
            ),
        )

        metrics = {
            "pitcher": {
                "pitcher_id":
                    int(pitcher_id),

                "pitcher_name":
                    str(pitcher_name),
            },

            "dataset": {
                "career_seasons":
                    career_seasons,

                "career_games":
                    career_games,

                "career_pitches":
                    career_pitches,
            },

            "evaluation": {
                "method":
                    "chronological_game_split",

                "test_fraction":
                    float(
                        self.test_fraction
                    ),

                "train_games":
                    int(
                        len(
                            train_game_ids
                        )
                    ),

                "test_games":
                    int(
                        len(
                            test_game_ids
                        )
                    ),

                "train_pitches":
                    int(
                        len(train)
                    ),

                "test_pitches":
                    int(
                        len(test)
                    ),

                "evaluation_reference_season":
                    evaluation_reference_season,

                "train_accuracy":
                    train_accuracy,

                "test_accuracy":
                    test_accuracy,

                "baseline_strategy":
                    "stratified",

                "baseline_distribution":
                    baseline_distribution,

                "baseline_accuracy":
                    baseline_accuracy,

                "accuracy_over_baseline":
                    accuracy_over_baseline,

                "accuracy_over_baseline_pp":
                    accuracy_over_baseline
                    * 100,

                "relative_improvement":
                    relative_improvement,

                "relative_improvement_percent":
                    (
                        relative_improvement
                        * 100
                        if relative_improvement
                        is not None
                        else None
                    ),
            },

            "production": {
                "reference_season":
                    int(
                        reference_season
                    ),

                "training_games":
                    career_games,

                "training_pitches":
                    career_pitches,

                "model_path":
                    str(
                        model_path
                    ),
            },

            "random_forest": {
                "n_estimators":
                    self.n_estimators,

                "max_depth":
                    self.max_depth,

                "min_samples_split":
                    self.min_samples_split,

                "min_samples_leaf":
                    self.min_samples_leaf,

                "max_features":
                    self.max_features,

                "bootstrap":
                    self.bootstrap,

                "random_state":
                    self.random_state,

                "n_jobs":
                    self.n_jobs,
            },

            "weighting": {
                "method":
                    (
                        "mild_recency_x_"
                        "continuous_repertoire"
                    ),

                "recency": {
                    "starting_decay_factor":
                        self.starting_decay_factor,

                    "decay_per_extra_season":
                        self.decay_per_extra_season,

                    "minimum_decay_factor":
                        self.minimum_decay_factor,

                    "minimum_sample_weight":
                        self.minimum_sample_weight,

                    "evaluation_decay_factor":
                        evaluation_decay_factor,

                    "production_decay_factor":
                        production_decay_factor,

                    "production_season_weights":
                        season_weights,
                },

                "repertoire": {
                    "settings":
                        asdict(
                            self.repertoire_settings
                        ),

                    "evaluation_diagnostics":
                        evaluation_repertoire,

                    "production_diagnostics":
                        production_repertoire,
                },
            },

            "top_feature_importances":
                feature_importance,
        }

        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return result