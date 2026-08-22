"""Ablation experiment for recency and repertoire-aware pitch weighting."""

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


# ============================================================
# REPERTOIRE SETTINGS
# ============================================================
#
# We only make repertoire judgments when we have enough pitches
# in the latest training season.
#
# This prevents us from declaring a pitch "retired" after only
# one or two starts.
# ============================================================

MIN_RECENT_PITCHES = 300
MIN_OLDER_SEASON_PITCHES = 300

# A pitch used less than 1% recently, but at least 8% during
# a previous reliable season, is considered inactive.
INACTIVE_USAGE_THRESHOLD = 0.01
MEANINGFUL_USAGE_THRESHOLD = 0.08

# If current usage is less than 50% of recent historical usage,
# the pitch is considered declining.
DECLINE_RATIO = 0.50

# A pitch at >=5% now that was <=1% historically is emerging.
EMERGING_USAGE_THRESHOLD = 0.05
OLD_EMERGING_USAGE_THRESHOLD = 0.01

# Multipliers applied to OLDER examples of these pitches.
INACTIVE_MULTIPLIER = 0.10
DECLINING_MULTIPLIER = 0.50


def fit_and_score(
    trainer: PitchModelTrainer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Train one Random Forest and return train/test accuracy.

    If every weight equals 1, train normally instead of passing
    sample_weight. This gives us a clean experimental control.
    """

    model = trainer._build_pipeline(
        X_train
    )

    if (
        sample_weights is None
        or np.allclose(sample_weights, 1.0)
    ):
        model.fit(
            X_train,
            y_train,
        )

    else:
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights,
        )

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
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

    return (
        train_accuracy,
        test_accuracy,
    )


def build_repertoire_weights(
    train: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """
    Analyze the pitcher's repertoire using ONLY training data.

    Returns:
        1. one repertoire multiplier per training pitch
        2. diagnostics describing each pitch type

    Important:
    Current/latest-season observations always keep multiplier 1.

    Repertoire penalties are applied only to OLD observations.

    Example:

        Changeup was 15% historically
        Changeup is now 0%

        old changeups -> multiplier 0.10

    This lets the model retain the history while strongly reducing
    the influence of a pitch that appears to have been abandoned.
    """

    weights = np.ones(
        len(train),
        dtype=float,
    )

    seasons = sorted(
        train["season"]
        .dropna()
        .astype(int)
        .unique()
    )

    pitch_types = sorted(
        train["pitch_type"]
        .astype(str)
        .unique()
    )

    # --------------------------------------------------------
    # One-season pitcher:
    # there is no repertoire history to compare against.
    # --------------------------------------------------------

    if len(seasons) < 2:

        diagnostics = []

        latest_season = seasons[-1]

        latest_data = train[
            train["season"]
            == latest_season
        ]

        latest_counts = (
            latest_data["pitch_type"]
            .astype(str)
            .value_counts(
                normalize=True
            )
        )

        for pitch_type in pitch_types:

            diagnostics.append(
                {
                    "pitch_type":
                        pitch_type,

                    "status":
                        "insufficient_history",

                    "recent_usage":
                        float(
                            latest_counts.get(
                                pitch_type,
                                0.0,
                            )
                        ),

                    "prior_usage":
                        0.0,

                    "peak_older_usage":
                        0.0,

                    "older_multiplier":
                        1.0,
                }
            )

        return (
            weights,
            diagnostics,
        )

    latest_season = seasons[-1]

    # --------------------------------------------------------
    # Build season-by-season pitch usage table
    # --------------------------------------------------------

    season_pitch_counts = (
        train
        .groupby(
            [
                "season",
                "pitch_type",
            ]
        )
        .size()
        .rename("pitch_count")
        .reset_index()
    )

    season_totals = (
        train
        .groupby("season")
        .size()
        .rename("season_total")
        .reset_index()
    )

    usage_table = (
        season_pitch_counts
        .merge(
            season_totals,
            on="season",
            how="left",
        )
    )

    usage_table["usage_rate"] = (
        usage_table["pitch_count"]
        / usage_table["season_total"]
    )

    # --------------------------------------------------------
    # Latest season
    # --------------------------------------------------------

    latest_total = int(
        (
            train["season"]
            == latest_season
        ).sum()
    )

    latest_usage = (
        usage_table[
            usage_table["season"]
            == latest_season
        ]
        .set_index("pitch_type")[
            "usage_rate"
        ]
        .to_dict()
    )

    # --------------------------------------------------------
    # If we do not have enough current-season observations,
    # do NOT make strong repertoire judgments.
    # --------------------------------------------------------

    if latest_total < MIN_RECENT_PITCHES:

        diagnostics = []

        for pitch_type in pitch_types:

            diagnostics.append(
                {
                    "pitch_type":
                        pitch_type,

                    "status":
                        "insufficient_recent_data",

                    "recent_usage":
                        float(
                            latest_usage.get(
                                pitch_type,
                                0.0,
                            )
                        ),

                    "prior_usage":
                        0.0,

                    "peak_older_usage":
                        0.0,

                    "older_multiplier":
                        1.0,
                }
            )

        return (
            weights,
            diagnostics,
        )

    # --------------------------------------------------------
    # Determine which older seasons contain enough data to
    # make meaningful comparisons.
    # --------------------------------------------------------

    reliable_older_seasons = []

    for season in seasons[:-1]:

        season_total = int(
            (
                train["season"]
                == season
            ).sum()
        )

        if (
            season_total
            >= MIN_OLDER_SEASON_PITCHES
        ):
            reliable_older_seasons.append(
                season
            )

    if not reliable_older_seasons:

        diagnostics = []

        for pitch_type in pitch_types:

            diagnostics.append(
                {
                    "pitch_type":
                        pitch_type,

                    "status":
                        "insufficient_history",

                    "recent_usage":
                        float(
                            latest_usage.get(
                                pitch_type,
                                0.0,
                            )
                        ),

                    "prior_usage":
                        0.0,

                    "peak_older_usage":
                        0.0,

                    "older_multiplier":
                        1.0,
                }
            )

        return (
            weights,
            diagnostics,
        )

    # --------------------------------------------------------
    # Use up to the previous two reliable seasons to measure
    # what the pitcher had been doing recently before the
    # latest season.
    # --------------------------------------------------------

    prior_seasons = (
        reliable_older_seasons[-2:]
    )

    prior_data = train[
        train["season"].isin(
            prior_seasons
        )
    ]

    prior_usage = (
        prior_data[
            "pitch_type"
        ]
        .astype(str)
        .value_counts(
            normalize=True
        )
        .to_dict()
    )

    # --------------------------------------------------------
    # Find the highest usage each pitch ever had during a
    # reliable older season.
    #
    # This lets us detect pitches that disappeared several
    # years ago, not just pitches that disappeared last year.
    # --------------------------------------------------------

    reliable_usage = usage_table[
        usage_table["season"].isin(
            reliable_older_seasons
        )
    ]

    peak_older_usage = (
        reliable_usage
        .groupby("pitch_type")[
            "usage_rate"
        ]
        .max()
        .to_dict()
    )

    diagnostics = []

    # --------------------------------------------------------
    # Classify every pitch type
    # --------------------------------------------------------

    for pitch_type in pitch_types:

        recent = float(
            latest_usage.get(
                pitch_type,
                0.0,
            )
        )

        prior = float(
            prior_usage.get(
                pitch_type,
                0.0,
            )
        )

        peak = float(
            peak_older_usage.get(
                pitch_type,
                0.0,
            )
        )

        status = "stable"
        multiplier = 1.0

        # ----------------------------------------------------
        # INACTIVE
        #
        # Example:
        #
        # historical usage = 15%
        # latest usage     = 0%
        # ----------------------------------------------------

        if (
            recent
            < INACTIVE_USAGE_THRESHOLD

            and peak
            >= MEANINGFUL_USAGE_THRESHOLD
        ):

            status = "inactive"

            multiplier = (
                INACTIVE_MULTIPLIER
            )

        # ----------------------------------------------------
        # DECLINING
        #
        # Example:
        #
        # prior usage  = 20%
        # recent usage = 6%
        # ----------------------------------------------------

        elif (
            prior
            >= MEANINGFUL_USAGE_THRESHOLD

            and recent
            < (
                prior
                * DECLINE_RATIO
            )
        ):

            status = "declining"

            multiplier = (
                DECLINING_MULTIPLIER
            )

        # ----------------------------------------------------
        # EMERGING / NEW
        #
        # Example:
        #
        # old usage    = ~0%
        # recent usage = 15%
        #
        # We do not need to increase its weight:
        # the latest examples already have the strongest
        # recency weight.
        # ----------------------------------------------------

        elif (
            recent
            >= EMERGING_USAGE_THRESHOLD

            and peak
            <= OLD_EMERGING_USAGE_THRESHOLD
        ):

            status = "emerging"

            multiplier = 1.0

        # ----------------------------------------------------
        # Apply repertoire penalty ONLY to historical rows.
        #
        # Recent examples remain fully relevant.
        # ----------------------------------------------------

        if multiplier < 1.0:

            mask = (
                (
                    train["pitch_type"]
                    .astype(str)
                    == pitch_type
                )
                &
                (
                    train["season"]
                    < latest_season
                )
            )

            weights[mask.to_numpy()] = (
                multiplier
            )

        diagnostics.append(
            {
                "pitch_type":
                    pitch_type,

                "status":
                    status,

                "recent_usage":
                    recent,

                "prior_usage":
                    prior,

                "peak_older_usage":
                    peak,

                "older_multiplier":
                    multiplier,
            }
        )

    return (
        weights,
        diagnostics,
    )


def evaluate_pitcher(
    kg4: pd.DataFrame,
    trainer: PitchModelTrainer,
) -> dict:
    """
    Compare four models on exactly the same train/test split:

    A. Unweighted
    B. Mild season recency
    C. Repertoire only
    D. Mild season recency + repertoire
    """

    data = kg4.dropna(
        subset=["pitch_type"]
    ).copy()

    if data.empty:
        raise ValueError(
            "No known pitch types."
        )

    data["game_date"] = pd.to_datetime(
        data["game_date"],
        errors="raise",
    )

    data["season"] = (
        data["game_date"].dt.year
    )

    game_order = (
        trainer._ordered_games(
            data
        )
    )

    if len(game_order) < 2:
        raise ValueError(
            "At least two games are required."
        )

    # ========================================================
    # SAME chronological 80/20 split for every model
    # ========================================================

    n_test_games = max(
        1,
        math.ceil(
            len(game_order)
            * trainer.test_fraction
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

    X_train = trainer._features(
        train
    )

    y_train = (
        train["pitch_type"]
        .astype(str)
    )

    X_test = trainer._features(
        test
    )

    y_test = (
        test["pitch_type"]
        .astype(str)
    )

    training_reference_season = int(
        train["season"].max()
    )

    # ========================================================
    # MODEL A: UNWEIGHTED
    # ========================================================

    (
        unweighted_train_accuracy,
        unweighted_test_accuracy,
    ) = fit_and_score(
        trainer,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ========================================================
    # MODEL B: MILD RECENCY ONLY
    # ========================================================

    recency_weights = (
        trainer._season_sample_weights(
            train,
            reference_season=(
                training_reference_season
            ),
        )
    )

    (
        recency_train_accuracy,
        recency_test_accuracy,
    ) = fit_and_score(
        trainer,
        X_train,
        y_train,
        X_test,
        y_test,
        recency_weights,
    )

    # ========================================================
    # MODEL C: REPERTOIRE ONLY
    # ========================================================

    (
        repertoire_weights,
        repertoire_diagnostics,
    ) = build_repertoire_weights(
        train
    )

    (
        repertoire_train_accuracy,
        repertoire_test_accuracy,
    ) = fit_and_score(
        trainer,
        X_train,
        y_train,
        X_test,
        y_test,
        repertoire_weights,
    )

    # ========================================================
    # MODEL D: MILD RECENCY + REPERTOIRE
    # ========================================================

    combined_weights = (
        recency_weights
        * repertoire_weights
    )

    (
        combined_train_accuracy,
        combined_test_accuracy,
    ) = fit_and_score(
        trainer,
        X_train,
        y_train,
        X_test,
        y_test,
        combined_weights,
    )

    # ========================================================
    # BASELINE
    # ========================================================

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

    # ========================================================
    # REPERTOIRE SUMMARY
    # ========================================================

    inactive = [
        item["pitch_type"]
        for item
        in repertoire_diagnostics
        if item["status"]
        == "inactive"
    ]

    declining = [
        item["pitch_type"]
        for item
        in repertoire_diagnostics
        if item["status"]
        == "declining"
    ]

    emerging = [
        item["pitch_type"]
        for item
        in repertoire_diagnostics
        if item["status"]
        == "emerging"
    ]

    repertoire_summary = "; ".join(
        (
            f"{item['pitch_type']}:"
            f"{item['status']}"
        )
        for item
        in repertoire_diagnostics
    )

    # ========================================================
    # RESULTS
    # ========================================================

    return {
        "career_seasons":
            int(
                data["season"].nunique()
            ),

        "career_games":
            len(game_order),

        "career_pitches":
            len(data),

        "training_reference_season":
            training_reference_season,

        "mild_decay_factor":
            trainer._decay_factor(
                int(
                    train[
                        "season"
                    ].nunique()
                )
            ),

        "baseline_accuracy":
            baseline_accuracy,

        "unweighted_test_accuracy":
            unweighted_test_accuracy,

        "recency_test_accuracy":
            recency_test_accuracy,

        "repertoire_test_accuracy":
            repertoire_test_accuracy,

        "combined_test_accuracy":
            combined_test_accuracy,

        "recency_effect_pp":
            (
                recency_test_accuracy
                - unweighted_test_accuracy
            )
            * 100,

        "repertoire_effect_pp":
            (
                repertoire_test_accuracy
                - unweighted_test_accuracy
            )
            * 100,

        "combined_effect_pp":
            (
                combined_test_accuracy
                - unweighted_test_accuracy
            )
            * 100,

        # This is particularly important:
        #
        # Does repertoire logic improve the mild-recency
        # model we were already considering?
        "repertoire_added_to_recency_pp":
            (
                combined_test_accuracy
                - recency_test_accuracy
            )
            * 100,

        "inactive_pitches":
            ",".join(inactive),

        "declining_pitches":
            ",".join(declining),

        "emerging_pitches":
            ",".join(emerging),

        "repertoire_summary":
            repertoire_summary,
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

    # ========================================================
    # NORMAL PROJECT COMPONENTS
    # ========================================================

    print(
        "Loading schemas..."
    )

    raw_schema = StatcastSchema.from_file(
        Path(
            "config/statcast_columns.txt"
        )
    )

    cleaned_schema = StatcastSchema.from_file(
        Path(
            "config/cleaned_columns.txt"
        )
    )

    cleaner = PitchDataCleaner(
        raw_schema,
        cleaned_schema,
    )

    feature_engineer = (
        PitchFeatureEngineer(
            cleaned_schema
        )
    )

    # IMPORTANT:
    #
    # No production models are created during this experiment.
    pipeline = DailyStarterPipeline(
        output_root=DEFAULT_OUTPUT_ROOT,
        schema=raw_schema,
        cleaner=cleaner,
        feature_engineer=(
            feature_engineer
        ),
        model_trainer=None,
    )

    # ========================================================
    # MILD RECENCY TRAINER
    # ========================================================
    #
    # decay =
    # max(
    #     0.80,
    #     0.98
    #     - 0.01 * (career seasons - 1)
    # )
    # ========================================================

    trainer = PitchModelTrainer(
        starting_decay_factor=0.98,
        decay_per_extra_season=0.01,
        minimum_decay_factor=0.80,
        minimum_sample_weight=0.10,
    )

    # ========================================================
    # FIND STARTERS
    # ========================================================

    print(
        f"Finding probable starters "
        f"for {game_date}..."
    )

    schedule = (
        pipeline.mlb.probable_starters(
            game_date
        )
    )

    unique_starters = list(
        {
            starter.pitcher_id:
                starter
            for starter
            in schedule.starters
        }.values()
    )

    print(
        f"Found {len(unique_starters)} "
        "unique starters."
    )

    if (
        len(unique_starters)
        < args.sample_size
    ):
        raise ValueError(
            f"Only "
            f"{len(unique_starters)} "
            "starters available."
        )

    rng = random.Random(
        args.seed
    )

    rng.shuffle(
        unique_starters
    )

    results = []
    failures = []

    # ========================================================
    # EVALUATE PITCHERS
    # ========================================================

    for starter in unique_starters:

        if (
            len(results)
            >= args.sample_size
        ):
            break

        print()
        print("=" * 70)

        print(
            f"Pitcher "
            f"{len(results) + 1}/"
            f"{args.sample_size}: "
            f"{starter.pitcher_name}"
        )

        print("=" * 70)

        try:

            print(
                "Downloading/updating "
                "career data..."
            )

            download = (
                pipeline._download_pitcher(
                    starter,
                    game_date
                    - timedelta(days=1),
                )
            )

            print(
                "Cleaning..."
            )

            cleaning = (
                pipeline._clean_pitcher(
                    starter,
                    Path(
                        download.path
                    ),
                )
            )

            print(
                "Creating KG4..."
            )

            features = (
                pipeline._engineer_pitcher(
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

            print(
                f"KG4: "
                f"{len(kg4):,} pitches"
            )

            print(
                "Training 4 "
                "comparison models..."
            )

            result = evaluate_pitcher(
                kg4,
                trainer,
            )

            result[
                "pitcher_id"
            ] = starter.pitcher_id

            result[
                "pitcher_name"
            ] = starter.pitcher_name

            results.append(
                result
            )

            print(
                "Unweighted:        "
                f"{result['unweighted_test_accuracy']:.4f}"
            )

            print(
                "Mild recency:      "
                f"{result['recency_test_accuracy']:.4f}"
            )

            print(
                "Repertoire only:   "
                f"{result['repertoire_test_accuracy']:.4f}"
            )

            print(
                "Recency + rep:     "
                f"{result['combined_test_accuracy']:.4f}"
            )

            print(
                "Rep added to mild: "
                f"{result['repertoire_added_to_recency_pp']:+.2f} pp"
            )

            print(
                "Inactive pitches:  "
                f"{result['inactive_pitches'] or 'none'}"
            )

            print(
                "Declining pitches: "
                f"{result['declining_pitches'] or 'none'}"
            )

            print(
                "Emerging pitches:  "
                f"{result['emerging_pitches'] or 'none'}"
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
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if not results:

        raise RuntimeError(
            "No pitchers were "
            "successfully evaluated."
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

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

            "mild_decay_factor",

            "baseline_accuracy",

            "unweighted_test_accuracy",
            "recency_test_accuracy",
            "repertoire_test_accuracy",
            "combined_test_accuracy",

            "recency_effect_pp",
            "repertoire_effect_pp",
            "combined_effect_pp",
            "repertoire_added_to_recency_pp",

            "inactive_pitches",
            "declining_pitches",
            "emerging_pitches",
            "repertoire_summary",
        ]
    ]

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    n = len(
        results_df
    )

    recency_wins = int(
        (
            results_df[
                "recency_effect_pp"
            ]
            > 0
        ).sum()
    )

    repertoire_wins = int(
        (
            results_df[
                "repertoire_effect_pp"
            ]
            > 0
        ).sum()
    )

    combined_wins = int(
        (
            results_df[
                "combined_effect_pp"
            ]
            > 0
        ).sum()
    )

    repertoire_improves_recency = int(
        (
            results_df[
                "repertoire_added_to_recency_pp"
            ]
            > 0
        ).sum()
    )

    mean_unweighted = float(
        results_df[
            "unweighted_test_accuracy"
        ].mean()
    )

    mean_recency = float(
        results_df[
            "recency_test_accuracy"
        ].mean()
    )

    mean_repertoire = float(
        results_df[
            "repertoire_test_accuracy"
        ].mean()
    )

    mean_combined = float(
        results_df[
            "combined_test_accuracy"
        ].mean()
    )

    mean_rep_added = float(
        results_df[
            "repertoire_added_to_recency_pp"
        ].mean()
    )

    median_rep_added = float(
        results_df[
            "repertoire_added_to_recency_pp"
        ].median()
    )

    # ========================================================
    # SAVE
    # ========================================================

    DEFAULT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        DEFAULT_RESULTS_DIR
        / (
            "repertoire_ablation_"
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
                "repertoire_ablation_"
                f"{game_date.isoformat()}"
                "_failures.csv"
            )
        )

        pd.DataFrame(
            failures
        ).to_csv(
            failure_path,
            index=False,
        )

    # ========================================================
    # PRINT FINAL REPORT
    # ========================================================

    display = (
        results_df.copy()
    )

    for column in [
        "baseline_accuracy",
        "unweighted_test_accuracy",
        "recency_test_accuracy",
        "repertoire_test_accuracy",
        "combined_test_accuracy",
    ]:

        display[column] = (
            display[column]
            * 100
        ).round(2)

    for column in [
        "recency_effect_pp",
        "repertoire_effect_pp",
        "combined_effect_pp",
        "repertoire_added_to_recency_pp",
    ]:

        display[column] = (
            display[column]
            .round(2)
        )

    print()
    print()
    print("=" * 110)
    print(
        "REPERTOIRE WEIGHTING ABLATION"
    )
    print("=" * 110)

    print()

    print(
        display[
            [
                "pitcher_name",
                "career_seasons",
                "unweighted_test_accuracy",
                "recency_test_accuracy",
                "repertoire_test_accuracy",
                "combined_test_accuracy",
                "repertoire_added_to_recency_pp",
                "inactive_pitches",
                "declining_pitches",
                "emerging_pitches",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("-" * 110)
    print(
        "AGGREGATE RESULTS"
    )
    print("-" * 110)

    print(
        f"Successful pitchers: "
        f"{n}"
    )

    print()

    print(
        f"Mild recency beats unweighted: "
        f"{recency_wins}/{n}"
    )

    print(
        f"Repertoire beats unweighted:   "
        f"{repertoire_wins}/{n}"
    )

    print(
        f"Combined beats unweighted:     "
        f"{combined_wins}/{n}"
    )

    print(
        f"Repertoire improves mild model:"
        f" {repertoire_improves_recency}/{n}"
    )

    print()

    print(
        "Mean unweighted accuracy: "
        f"{mean_unweighted * 100:.2f}%"
    )

    print(
        "Mean mild-recency accuracy:"
        f" {mean_recency * 100:.2f}%"
    )

    print(
        "Mean repertoire accuracy:  "
        f"{mean_repertoire * 100:.2f}%"
    )

    print(
        "Mean combined accuracy:    "
        f"{mean_combined * 100:.2f}%"
    )

    print()

    print(
        "Mean repertoire benefit "
        "on top of mild recency: "
        f"{mean_rep_added:+.2f} pp"
    )

    print(
        "Median repertoire benefit "
        "on top of mild recency: "
        f"{median_rep_added:+.2f} pp"
    )

    print()

    print(
        f"Results saved to: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()