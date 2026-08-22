"""Compare unweighted, current-weighted, and mild-weighted Random Forests."""

from __future__ import annotations

import argparse
import math
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from pitch_prediction.cleaning import PitchDataCleaner
from pitch_prediction.feature_engineering import PitchFeatureEngineer
from pitch_prediction.model import PitchModelTrainer
from pitch_prediction.pipeline import DailyStarterPipeline
from pitch_prediction.schema import StatcastSchema


DEFAULT_OUTPUT_ROOT = Path("Data/daily_pipeline")
DEFAULT_RESULTS_DIR = Path("Data/ablation")


def evaluate_pitcher(
    kg4: pd.DataFrame,
    current_trainer: PitchModelTrainer,
    mild_trainer: PitchModelTrainer,
) -> dict:

    data = kg4.dropna(
        subset=["pitch_type"]
    ).copy()

    if data.empty:
        raise ValueError(
            "Pitcher has no pitches with known pitch_type."
        )

    data["game_date"] = pd.to_datetime(
        data["game_date"],
        errors="raise",
    )

    data["season"] = (
        data["game_date"].dt.year
    )

    game_order = current_trainer._ordered_games(
        data
    )

    if len(game_order) < 2:
        raise ValueError(
            "Pitcher needs at least 2 games "
            "for chronological evaluation."
        )

    # ---------------------------------------------------------
    # SAME chronological 80/20 split for all three models
    # ---------------------------------------------------------

    n_test_games = max(
        1,
        math.ceil(
            len(game_order)
            * current_trainer.test_fraction
        ),
    )

    n_test_games = min(
        n_test_games,
        len(game_order) - 1,
    )

    train_game_ids = set(
        game_order[:-n_test_games]
    )

    test_game_ids = set(
        game_order[-n_test_games:]
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

    X_train = current_trainer._features(
        train
    )

    y_train = (
        train["pitch_type"]
        .astype(str)
    )

    X_test = current_trainer._features(
        test
    )

    y_test = (
        test["pitch_type"]
        .astype(str)
    )

    training_reference_season = int(
        train["season"].max()
    )

    # =========================================================
    # MODEL A: UNWEIGHTED
    # =========================================================

    unweighted_model = (
        current_trainer._build_pipeline(
            X_train
        )
    )

    unweighted_model.fit(
        X_train,
        y_train,
    )

    unweighted_train_predictions = (
        unweighted_model.predict(
            X_train
        )
    )

    unweighted_test_predictions = (
        unweighted_model.predict(
            X_test
        )
    )

    unweighted_train_accuracy = float(
        accuracy_score(
            y_train,
            unweighted_train_predictions,
        )
    )

    unweighted_test_accuracy = float(
        accuracy_score(
            y_test,
            unweighted_test_predictions,
        )
    )

    # =========================================================
    # MODEL B: CURRENT / MORE AGGRESSIVE WEIGHTING
    # =========================================================

    current_weights = (
        current_trainer._season_sample_weights(
            train,
            reference_season=(
                training_reference_season
            ),
        )
    )

    current_weighted_model = (
        current_trainer._build_pipeline(
            X_train
        )
    )

    current_weighted_model.fit(
        X_train,
        y_train,
        classifier__sample_weight=(
            current_weights
        ),
    )

    current_train_predictions = (
        current_weighted_model.predict(
            X_train
        )
    )

    current_test_predictions = (
        current_weighted_model.predict(
            X_test
        )
    )

    current_train_accuracy = float(
        accuracy_score(
            y_train,
            current_train_predictions,
        )
    )

    current_test_accuracy = float(
        accuracy_score(
            y_test,
            current_test_predictions,
        )
    )

    # =========================================================
    # MODEL C: MILD RECENCY WEIGHTING
    # =========================================================

    mild_weights = (
        mild_trainer._season_sample_weights(
            train,
            reference_season=(
                training_reference_season
            ),
        )
    )

    mild_model = (
        mild_trainer._build_pipeline(
            X_train
        )
    )

    mild_model.fit(
        X_train,
        y_train,
        classifier__sample_weight=(
            mild_weights
        ),
    )

    mild_train_predictions = (
        mild_model.predict(
            X_train
        )
    )

    mild_test_predictions = (
        mild_model.predict(
            X_test
        )
    )

    mild_train_accuracy = float(
        accuracy_score(
            y_train,
            mild_train_predictions,
        )
    )

    mild_test_accuracy = float(
        accuracy_score(
            y_test,
            mild_test_predictions,
        )
    )

    # =========================================================
    # COMMON BASELINE
    # =========================================================

    majority_class = (
        y_train.mode().iloc[0]
    )

    baseline_predictions = np.full(
        len(y_test),
        majority_class,
        dtype=object,
    )

    baseline_accuracy = float(
        accuracy_score(
            y_test,
            baseline_predictions,
        )
    )

    # =========================================================
    # EFFECTS
    # =========================================================

    current_effect = (
        current_test_accuracy
        - unweighted_test_accuracy
    )

    mild_effect = (
        mild_test_accuracy
        - unweighted_test_accuracy
    )

    mild_vs_current = (
        mild_test_accuracy
        - current_test_accuracy
    )

    training_seasons = int(
        train["season"].nunique()
    )

    career_seasons = int(
        data["season"].nunique()
    )

    return {
        "career_seasons":
            career_seasons,

        "training_seasons":
            training_seasons,

        "career_games":
            len(game_order),

        "career_pitches":
            len(data),

        "current_decay_factor":
            current_trainer._decay_factor(
                training_seasons
            ),

        "mild_decay_factor":
            mild_trainer._decay_factor(
                training_seasons
            ),

        "baseline_accuracy":
            baseline_accuracy,

        "unweighted_train_accuracy":
            unweighted_train_accuracy,

        "unweighted_test_accuracy":
            unweighted_test_accuracy,

        "current_train_accuracy":
            current_train_accuracy,

        "current_test_accuracy":
            current_test_accuracy,

        "mild_train_accuracy":
            mild_train_accuracy,

        "mild_test_accuracy":
            mild_test_accuracy,

        "current_effect_pp":
            current_effect * 100,

        "mild_effect_pp":
            mild_effect * 100,

        "mild_vs_current_pp":
            mild_vs_current * 100,
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        type=str,
        default=date.today().isoformat(),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    game_date = date.fromisoformat(
        args.date
    )

    # ---------------------------------------------------------
    # Load normal project pipeline pieces
    # ---------------------------------------------------------

    print("Loading schemas...")

    raw_schema = StatcastSchema.from_file(
        Path("config/statcast_columns.txt")
    )

    cleaned_schema = StatcastSchema.from_file(
        Path("config/cleaned_columns.txt")
    )

    cleaner = PitchDataCleaner(
        raw_schema,
        cleaned_schema,
    )

    feature_engineer = PitchFeatureEngineer(
        cleaned_schema
    )

    pipeline = DailyStarterPipeline(
        output_root=DEFAULT_OUTPUT_ROOT,
        schema=raw_schema,
        cleaner=cleaner,
        feature_engineer=feature_engineer,

        # Important:
        # we do not want production models created
        # during this experiment.
        model_trainer=None,
    )

    # ---------------------------------------------------------
    # Current weighting equation
    # ---------------------------------------------------------

    current_trainer = PitchModelTrainer()

    # ---------------------------------------------------------
    # New mild weighting equation
    #
    # decay =
    # max(
    #     0.80,
    #     0.98 - 0.01 * (career seasons - 1)
    # )
    # ---------------------------------------------------------

    mild_trainer = PitchModelTrainer(
        starting_decay_factor=0.98,
        decay_per_extra_season=0.01,
        minimum_decay_factor=0.80,
        minimum_sample_weight=0.10,
    )

    # ---------------------------------------------------------
    # Get probable starters
    # ---------------------------------------------------------

    print(
        f"Finding probable starters for {game_date}..."
    )

    schedule = (
        pipeline.mlb.probable_starters(
            game_date
        )
    )

    unique_starters = list(
        {
            starter.pitcher_id: starter
            for starter in schedule.starters
        }.values()
    )

    print(
        f"Found {len(unique_starters)} "
        "unique probable starters."
    )

    if len(unique_starters) < args.sample_size:
        raise ValueError(
            f"Only {len(unique_starters)} starters "
            f"available for sample size "
            f"{args.sample_size}."
        )

    rng = random.Random(
        args.seed
    )

    rng.shuffle(
        unique_starters
    )

    results = []
    failures = []

    # ---------------------------------------------------------
    # Evaluate pitchers
    # ---------------------------------------------------------

    for starter in unique_starters:

        if len(results) >= args.sample_size:
            break

        print()
        print("=" * 70)

        print(
            f"Pitcher {len(results) + 1}/"
            f"{args.sample_size}: "
            f"{starter.pitcher_name}"
        )

        print("=" * 70)

        try:

            print(
                "Downloading/updating career data..."
            )

            download = pipeline._download_pitcher(
                starter,
                game_date - timedelta(days=1),
            )

            print("Cleaning...")

            cleaning = pipeline._clean_pitcher(
                starter,
                Path(download.path),
            )

            print(
                "Creating KG4..."
            )

            features = pipeline._engineer_pitcher(
                starter,
                Path(cleaning.cleaned_path),
            )

            kg4 = pd.read_csv(
                features.kg4_path,
                low_memory=False,
            )

            print(
                f"KG4: {len(kg4):,} pitches"
            )

            print(
                "Training 3 comparison models..."
            )

            result = evaluate_pitcher(
                kg4,
                current_trainer,
                mild_trainer,
            )

            result["pitcher_id"] = (
                starter.pitcher_id
            )

            result["pitcher_name"] = (
                starter.pitcher_name
            )

            results.append(
                result
            )

            print(
                "Unweighted:      "
                f"{result['unweighted_test_accuracy']:.4f}"
            )

            print(
                "Current weighted:"
                f" {result['current_test_accuracy']:.4f}"
            )

            print(
                "Mild weighted:   "
                f"{result['mild_test_accuracy']:.4f}"
            )

            print(
                "Current effect:  "
                f"{result['current_effect_pp']:+.2f} pp"
            )

            print(
                "Mild effect:     "
                f"{result['mild_effect_pp']:+.2f} pp"
            )

        except Exception as exc:

            failures.append(
                {
                    "pitcher_id":
                        starter.pitcher_id,

                    "pitcher_name":
                        starter.pitcher_name,

                    "error_type":
                        type(exc).__name__,

                    "message":
                        str(exc),
                }
            )

            print(
                "FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

    if not results:
        raise RuntimeError(
            "No pitchers were successfully evaluated."
        )

    # ---------------------------------------------------------
    # Results table
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "pitcher_id",
            "pitcher_name",
            "career_seasons",
            "career_games",
            "career_pitches",

            "current_decay_factor",
            "mild_decay_factor",

            "baseline_accuracy",

            "unweighted_test_accuracy",
            "current_test_accuracy",
            "mild_test_accuracy",

            "current_effect_pp",
            "mild_effect_pp",
            "mild_vs_current_pp",
        ]
    ]

    # ---------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------

    current_wins = int(
        (
            results_df[
                "current_effect_pp"
            ] > 0
        ).sum()
    )

    mild_wins = int(
        (
            results_df[
                "mild_effect_pp"
            ] > 0
        ).sum()
    )

    mild_wins_vs_current = int(
        (
            results_df[
                "mild_vs_current_pp"
            ] > 0
        ).sum()
    )

    mean_unweighted = float(
        results_df[
            "unweighted_test_accuracy"
        ].mean()
    )

    mean_current = float(
        results_df[
            "current_test_accuracy"
        ].mean()
    )

    mean_mild = float(
        results_df[
            "mild_test_accuracy"
        ].mean()
    )

    mean_current_effect = float(
        results_df[
            "current_effect_pp"
        ].mean()
    )

    mean_mild_effect = float(
        results_df[
            "mild_effect_pp"
        ].mean()
    )

    median_current_effect = float(
        results_df[
            "current_effect_pp"
        ].median()
    )

    median_mild_effect = float(
        results_df[
            "mild_effect_pp"
        ].median()
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    DEFAULT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        DEFAULT_RESULTS_DIR
        / (
            "three_way_weight_ablation_"
            f"{game_date.isoformat()}.csv"
        )
    )

    results_df.to_csv(
        result_path,
        index=False,
    )

    if failures:

        failure_path = (
            DEFAULT_RESULTS_DIR
            / (
                "three_way_weight_ablation_"
                f"{game_date.isoformat()}_failures.csv"
            )
        )

        pd.DataFrame(
            failures
        ).to_csv(
            failure_path,
            index=False,
        )

    # ---------------------------------------------------------
    # Pretty output
    # ---------------------------------------------------------

    display = results_df.copy()

    for column in [
        "baseline_accuracy",
        "unweighted_test_accuracy",
        "current_test_accuracy",
        "mild_test_accuracy",
    ]:

        display[column] = (
            display[column]
            * 100
        ).round(2)

    for column in [
        "current_effect_pp",
        "mild_effect_pp",
        "mild_vs_current_pp",
    ]:

        display[column] = (
            display[column]
            .round(2)
        )

    print()
    print()
    print("=" * 100)
    print(
        "THREE-WAY RECENCY WEIGHT ABLATION"
    )
    print("=" * 100)

    print()
    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("-" * 100)
    print("AGGREGATE RESULTS")
    print("-" * 100)

    print(
        f"Successful pitchers: "
        f"{len(results_df)}"
    )

    print()
    print(
        f"Current weighting wins vs unweighted: "
        f"{current_wins}/{len(results_df)}"
    )

    print(
        f"Mild weighting wins vs unweighted:    "
        f"{mild_wins}/{len(results_df)}"
    )

    print(
        f"Mild weighting wins vs current:       "
        f"{mild_wins_vs_current}/{len(results_df)}"
    )

    print()

    print(
        "Mean unweighted accuracy: "
        f"{mean_unweighted * 100:.2f}%"
    )

    print(
        "Mean current accuracy:    "
        f"{mean_current * 100:.2f}%"
    )

    print(
        "Mean mild accuracy:       "
        f"{mean_mild * 100:.2f}%"
    )

    print()

    print(
        "Mean current effect:      "
        f"{mean_current_effect:+.2f} pp"
    )

    print(
        "Median current effect:    "
        f"{median_current_effect:+.2f} pp"
    )

    print()

    print(
        "Mean mild effect:         "
        f"{mean_mild_effect:+.2f} pp"
    )

    print(
        "Median mild effect:       "
        f"{median_mild_effect:+.2f} pp"
    )

    print()
    print(
        f"Results saved to: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()