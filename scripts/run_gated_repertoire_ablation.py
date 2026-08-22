"""Validation ablation for gated repertoire weighting on a fresh pitcher sample."""

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
# DEVELOPMENT SAMPLE
# ============================================================
#
# These pitchers were already used while designing/tuning
# the weighting rules.
#
# They are excluded from this validation experiment.
# ============================================================

DEVELOPMENT_PITCHER_IDS = {
    680694,  # Kyle Bradish
    543037,  # Gerrit Cole
    669456,  # Shane Bieber
    641743,  # Anthony Kay
    693855,  # Ian Seymour
    677976,  # Randy Dobnak
    680570,  # Grayson Rodriguez
    695611,  # Gage Jump
    669923,  # George Kirby
    594798,  # Jacob deGrom
}


# ============================================================
# DATA REQUIREMENTS
# ============================================================

MIN_RECENT_PITCHES = 300
MIN_OLDER_SEASON_PITCHES = 300


# ============================================================
# CONTINUOUS WEIGHT SETTINGS
# ============================================================

CONTINUOUS_MIN_MULTIPLIER = 0.15
USAGE_SMOOTHING = 0.01

MAX_EMERGING_BOOST = 1.50
BOOST_SCALE = 2.5


# ============================================================
# GATING RULES
# ============================================================
#
# Decline activates only when:
#
# prior usage >= 8%
# AND
# recent usage < 50% of prior usage
#
# Emerging activates only when:
#
# prior usage <= 2%
# AND
# recent usage >= 8%
#
# These rules are now FROZEN for validation.
# ============================================================

GATED_MIN_PRIOR_USAGE = 0.08
GATED_DECLINE_RATIO = 0.50

GATED_MAX_OLD_EMERGING_USAGE = 0.02
GATED_MIN_RECENT_EMERGING_USAGE = 0.08


def fit_and_score(
    trainer: PitchModelTrainer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weights: np.ndarray | None = None,
) -> float:
    """Train one model and return test accuracy."""

    model = trainer._build_pipeline(
        X_train
    )

    if (
        sample_weights is None
        or np.allclose(
            sample_weights,
            1.0,
        )
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

    predictions = model.predict(
        X_test
    )

    return float(
        accuracy_score(
            y_test,
            predictions,
        )
    )


def get_repertoire_context(
    train: pd.DataFrame,
) -> dict | None:
    """
    Build recent-vs-prior repertoire information.

    Uses ONLY training data.
    """

    seasons = sorted(
        train["season"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(seasons) < 2:
        return None

    latest_season = seasons[-1]

    latest_data = train[
        train["season"]
        == latest_season
    ]

    if len(latest_data) < MIN_RECENT_PITCHES:
        return None

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
        return None

    # Use the two most recent reliable older seasons.
    prior_seasons = (
        reliable_older_seasons[-2:]
    )

    prior_data = train[
        train["season"].isin(
            prior_seasons
        )
    ]

    recent_usage = (
        latest_data["pitch_type"]
        .astype(str)
        .value_counts(
            normalize=True
        )
        .to_dict()
    )

    prior_usage = (
        prior_data["pitch_type"]
        .astype(str)
        .value_counts(
            normalize=True
        )
        .to_dict()
    )

    pitch_types = sorted(
        train["pitch_type"]
        .astype(str)
        .unique()
    )

    return {
        "latest_season":
            latest_season,

        "recent_usage":
            recent_usage,

        "prior_usage":
            prior_usage,

        "pitch_types":
            pitch_types,
    }


def build_ungated_continuous_weights(
    train: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """
    Previous ungated continuous weighting.

    Included only as a comparison.
    """

    weights = np.ones(
        len(train),
        dtype=float,
    )

    context = get_repertoire_context(
        train
    )

    if context is None:
        return weights, []

    latest_season = context[
        "latest_season"
    ]

    recent_usage = context[
        "recent_usage"
    ]

    prior_usage = context[
        "prior_usage"
    ]

    diagnostics = []

    for pitch_type in context[
        "pitch_types"
    ]:

        recent = float(
            recent_usage.get(
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

        old_multiplier = 1.0
        recent_multiplier = 1.0
        status = "stable"

        if (
            prior >= 0.03
            and recent < prior
        ):

            usage_ratio = (
                (
                    recent
                    + USAGE_SMOOTHING
                )
                /
                (
                    prior
                    + USAGE_SMOOTHING
                )
            )

            old_multiplier = float(
                np.clip(
                    usage_ratio,
                    CONTINUOUS_MIN_MULTIPLIER,
                    1.0,
                )
            )

            if recent < 0.01:
                status = "nearly_inactive"
            else:
                status = "declining"

        elif (
            recent > prior
            and recent >= 0.05
        ):

            increase = (
                recent - prior
            )

            recent_multiplier = float(
                np.clip(
                    1.0
                    + increase
                    * BOOST_SCALE,
                    1.0,
                    MAX_EMERGING_BOOST,
                )
            )

            if prior <= 0.01:
                status = "emerging"
            else:
                status = "increasing"

        if old_multiplier < 1.0:

            old_mask = (
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

            weights[
                old_mask.to_numpy()
            ] = old_multiplier

        if recent_multiplier > 1.0:

            recent_mask = (
                (
                    train["pitch_type"]
                    .astype(str)
                    == pitch_type
                )
                &
                (
                    train["season"]
                    == latest_season
                )
            )

            weights[
                recent_mask.to_numpy()
            ] = recent_multiplier

        diagnostics.append(
            {
                "pitch_type":
                    pitch_type,

                "recent_usage":
                    recent,

                "prior_usage":
                    prior,

                "status":
                    status,

                "old_multiplier":
                    old_multiplier,

                "recent_multiplier":
                    recent_multiplier,
            }
        )

    return (
        weights,
        diagnostics,
    )


def build_gated_continuous_weights(
    train: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """
    Gated repertoire weighting.

    Small repertoire fluctuations are ignored.

    Weighting activates only for major declines
    or clearly emerging pitches.
    """

    weights = np.ones(
        len(train),
        dtype=float,
    )

    context = get_repertoire_context(
        train
    )

    if context is None:
        return weights, []

    latest_season = context[
        "latest_season"
    ]

    recent_usage = context[
        "recent_usage"
    ]

    prior_usage = context[
        "prior_usage"
    ]

    diagnostics = []

    for pitch_type in context[
        "pitch_types"
    ]:

        recent = float(
            recent_usage.get(
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

        old_multiplier = 1.0
        recent_multiplier = 1.0

        status = "stable"

        # ====================================================
        # MAJOR DECLINE
        # ====================================================

        substantial_decline = (
            prior
            >= GATED_MIN_PRIOR_USAGE

            and recent
            < (
                prior
                * GATED_DECLINE_RATIO
            )
        )

        if substantial_decline:

            usage_ratio = (
                (
                    recent
                    + USAGE_SMOOTHING
                )
                /
                (
                    prior
                    + USAGE_SMOOTHING
                )
            )

            old_multiplier = float(
                np.clip(
                    usage_ratio,
                    CONTINUOUS_MIN_MULTIPLIER,
                    1.0,
                )
            )

            if recent < 0.01:
                status = "inactive"

            else:
                status = (
                    "major_decline"
                )

        # ====================================================
        # EMERGING PITCH
        # ====================================================

        substantial_emergence = (
            prior
            <= GATED_MAX_OLD_EMERGING_USAGE

            and recent
            >= GATED_MIN_RECENT_EMERGING_USAGE
        )

        if substantial_emergence:

            increase = (
                recent - prior
            )

            recent_multiplier = float(
                np.clip(
                    1.0
                    + increase
                    * BOOST_SCALE,
                    1.0,
                    MAX_EMERGING_BOOST,
                )
            )

            status = "emerging"

        # ====================================================
        # APPLY OLD-PITCH PENALTY
        # ====================================================

        if old_multiplier < 1.0:

            old_mask = (
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

            weights[
                old_mask.to_numpy()
            ] = old_multiplier

        # ====================================================
        # APPLY RECENT BOOST
        # ====================================================

        if recent_multiplier > 1.0:

            recent_mask = (
                (
                    train["pitch_type"]
                    .astype(str)
                    == pitch_type
                )
                &
                (
                    train["season"]
                    == latest_season
                )
            )

            weights[
                recent_mask.to_numpy()
            ] = recent_multiplier

        diagnostics.append(
            {
                "pitch_type":
                    pitch_type,

                "recent_usage":
                    recent,

                "prior_usage":
                    prior,

                "status":
                    status,

                "old_multiplier":
                    old_multiplier,

                "recent_multiplier":
                    recent_multiplier,
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
    # SAME CHRONOLOGICAL 80/20 SPLIT
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

    reference_season = int(
        train["season"].max()
    )

    # ========================================================
    # A. UNWEIGHTED
    # ========================================================

    unweighted_accuracy = (
        fit_and_score(
            trainer,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    # ========================================================
    # B. MILD RECENCY
    # ========================================================

    recency_weights = (
        trainer._season_sample_weights(
            train,
            reference_season=(
                reference_season
            ),
        )
    )

    recency_accuracy = (
        fit_and_score(
            trainer,
            X_train,
            y_train,
            X_test,
            y_test,
            recency_weights,
        )
    )

    # ========================================================
    # C. MILD + UNGATED REPERTOIRE
    # ========================================================

    (
        ungated_weights,
        ungated_diagnostics,
    ) = (
        build_ungated_continuous_weights(
            train
        )
    )

    ungated_combined_weights = (
        recency_weights
        * ungated_weights
    )

    ungated_accuracy = (
        fit_and_score(
            trainer,
            X_train,
            y_train,
            X_test,
            y_test,
            ungated_combined_weights,
        )
    )

    # ========================================================
    # D. MILD + GATED REPERTOIRE
    # ========================================================

    (
        gated_weights,
        gated_diagnostics,
    ) = (
        build_gated_continuous_weights(
            train
        )
    )

    gated_combined_weights = (
        recency_weights
        * gated_weights
    )

    gated_accuracy = (
        fit_and_score(
            trainer,
            X_train,
            y_train,
            X_test,
            y_test,
            gated_combined_weights,
        )
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
    # DIAGNOSTICS
    # ========================================================

    ungated_summary = "; ".join(
        (
            f"{item['pitch_type']}:"
            f"{item['status']}"
            f"(old={item['old_multiplier']:.2f},"
            f"new={item['recent_multiplier']:.2f})"
        )
        for item
        in ungated_diagnostics
    )

    gated_summary = "; ".join(
        (
            f"{item['pitch_type']}:"
            f"{item['status']}"
            f"(old={item['old_multiplier']:.2f},"
            f"new={item['recent_multiplier']:.2f})"
        )
        for item
        in gated_diagnostics
    )

    return {
        "career_seasons":
            int(
                data["season"].nunique()
            ),

        "career_games":
            len(game_order),

        "career_pitches":
            len(data),

        "baseline_accuracy":
            baseline_accuracy,

        "unweighted_accuracy":
            unweighted_accuracy,

        "recency_accuracy":
            recency_accuracy,

        "ungated_accuracy":
            ungated_accuracy,

        "gated_accuracy":
            gated_accuracy,

        "recency_effect_pp":
            (
                recency_accuracy
                - unweighted_accuracy
            )
            * 100,

        "ungated_effect_pp":
            (
                ungated_accuracy
                - unweighted_accuracy
            )
            * 100,

        "gated_effect_pp":
            (
                gated_accuracy
                - unweighted_accuracy
            )
            * 100,

        "ungated_added_to_recency_pp":
            (
                ungated_accuracy
                - recency_accuracy
            )
            * 100,

        "gated_added_to_recency_pp":
            (
                gated_accuracy
                - recency_accuracy
            )
            * 100,

        "gated_vs_ungated_pp":
            (
                gated_accuracy
                - ungated_accuracy
            )
            * 100,

        "ungated_summary":
            ungated_summary,

        "gated_summary":
            gated_summary,
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
        default=15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    args = parser.parse_args()

    game_date = (
        date.fromisoformat(
            args.date
        )
    )

    # ========================================================
    # PROJECT SETUP
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
    # FROZEN MILD RECENCY SETTINGS
    # ========================================================

    trainer = PitchModelTrainer(
        starting_decay_factor=0.98,
        decay_per_extra_season=0.01,
        minimum_decay_factor=0.80,
        minimum_sample_weight=0.10,
    )

    # ========================================================
    # GET VALIDATION STARTERS
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
            if (
                starter.pitcher_id
                not in DEVELOPMENT_PITCHER_IDS
            )
        }.values()
    )

    print(
        f"Found "
        f"{len(unique_starters)} "
        "eligible starters after "
        "excluding the development sample."
    )

    if (
        len(unique_starters)
        < args.sample_size
    ):
        raise ValueError(
            f"Only "
            f"{len(unique_starters)} "
            "eligible starters available "
            f"for requested sample size "
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

    # ========================================================
    # RUN VALIDATION
    # ========================================================

    for starter in unique_starters:

        if (
            len(results)
            >= args.sample_size
        ):
            break

        print()
        print("=" * 80)

        print(
            f"Validation pitcher "
            f"{len(results) + 1}/"
            f"{args.sample_size}: "
            f"{starter.pitcher_name}"
        )

        print("=" * 80)

        try:

            print(
                "Downloading/updating career data..."
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
                "Training validation models..."
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
                "Unweighted:       "
                f"{result['unweighted_accuracy']:.4f}"
            )

            print(
                "Mild recency:     "
                f"{result['recency_accuracy']:.4f}"
            )

            print(
                "Ungated rep:      "
                f"{result['ungated_accuracy']:.4f}"
            )

            print(
                "Gated rep:        "
                f"{result['gated_accuracy']:.4f}"
            )

            print(
                "Gated vs mild:    "
                f"{result['gated_added_to_recency_pp']:+.2f} pp"
            )

            print(
                "Gated changes:    "
                f"{result['gated_summary'] or 'none'}"
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
            "No validation pitchers were "
            "successfully evaluated."
        )

    # ========================================================
    # RESULTS
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

            "baseline_accuracy",

            "unweighted_accuracy",
            "recency_accuracy",
            "ungated_accuracy",
            "gated_accuracy",

            "recency_effect_pp",
            "ungated_effect_pp",
            "gated_effect_pp",

            "ungated_added_to_recency_pp",
            "gated_added_to_recency_pp",
            "gated_vs_ungated_pp",

            "ungated_summary",
            "gated_summary",
        ]
    ]

    n = len(
        results_df
    )

    gated_wins_vs_unweighted = int(
        (
            results_df[
                "gated_effect_pp"
            ] > 0
        ).sum()
    )

    gated_wins_vs_recency = int(
        (
            results_df[
                "gated_added_to_recency_pp"
            ] > 0
        ).sum()
    )

    gated_losses_vs_recency = int(
        (
            results_df[
                "gated_added_to_recency_pp"
            ] < 0
        ).sum()
    )

    gated_ties_vs_recency = int(
        (
            results_df[
                "gated_added_to_recency_pp"
            ] == 0
        ).sum()
    )

    mean_unweighted = float(
        results_df[
            "unweighted_accuracy"
        ].mean()
    )

    mean_recency = float(
        results_df[
            "recency_accuracy"
        ].mean()
    )

    mean_ungated = float(
        results_df[
            "ungated_accuracy"
        ].mean()
    )

    mean_gated = float(
        results_df[
            "gated_accuracy"
        ].mean()
    )

    mean_gated_added = float(
        results_df[
            "gated_added_to_recency_pp"
        ].mean()
    )

    median_gated_added = float(
        results_df[
            "gated_added_to_recency_pp"
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
            "fresh_validation_gated_repertoire_"
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
                "fresh_validation_gated_repertoire_"
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
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 115)
    print(
        "FRESH-SAMPLE GATED REPERTOIRE VALIDATION"
    )
    print("=" * 115)

    print()

    display = (
        results_df.copy()
    )

    for column in [
        "baseline_accuracy",
        "unweighted_accuracy",
        "recency_accuracy",
        "ungated_accuracy",
        "gated_accuracy",
    ]:

        display[column] = (
            display[column]
            * 100
        ).round(2)

    for column in [
        "recency_effect_pp",
        "ungated_effect_pp",
        "gated_effect_pp",
        "ungated_added_to_recency_pp",
        "gated_added_to_recency_pp",
        "gated_vs_ungated_pp",
    ]:

        display[column] = (
            display[column]
            .round(2)
        )

    print(
        display[
            [
                "pitcher_name",
                "career_seasons",
                "unweighted_accuracy",
                "recency_accuracy",
                "gated_accuracy",
                "gated_added_to_recency_pp",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("-" * 115)
    print(
        "VALIDATION SUMMARY"
    )
    print("-" * 115)

    print(
        f"Successful validation pitchers: "
        f"{n}"
    )

    print()

    print(
        "Gated beats unweighted: "
        f"{gated_wins_vs_unweighted}/{n}"
    )

    print(
        "Gated improves mild recency: "
        f"{gated_wins_vs_recency}/{n}"
    )

    print(
        "Gated hurts mild recency: "
        f"{gated_losses_vs_recency}/{n}"
    )

    print(
        "Gated ties mild recency: "
        f"{gated_ties_vs_recency}/{n}"
    )

    print()

    print(
        "Mean unweighted accuracy: "
        f"{mean_unweighted * 100:.2f}%"
    )

    print(
        "Mean mild-recency accuracy: "
        f"{mean_recency * 100:.2f}%"
    )

    print(
        "Mean ungated accuracy: "
        f"{mean_ungated * 100:.2f}%"
    )

    print(
        "Mean gated accuracy: "
        f"{mean_gated * 100:.2f}%"
    )

    print()

    print(
        "Mean gated benefit over mild: "
        f"{mean_gated_added:+.2f} pp"
    )

    print(
        "Median gated benefit over mild: "
        f"{median_gated_added:+.2f} pp"
    )

    print()

    print(
        f"Results saved to: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()