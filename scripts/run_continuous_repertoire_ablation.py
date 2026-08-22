"""Compare categorical vs continuous repertoire-aware weighting."""

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

MIN_RECENT_PITCHES = 300
MIN_OLDER_SEASON_PITCHES = 300

# Categorical repertoire settings
INACTIVE_USAGE_THRESHOLD = 0.01
MEANINGFUL_USAGE_THRESHOLD = 0.08
DECLINE_RATIO = 0.50

INACTIVE_MULTIPLIER = 0.10
DECLINING_MULTIPLIER = 0.50

# Continuous repertoire settings
CONTINUOUS_MIN_MULTIPLIER = 0.15

# Small value used to prevent unstable ratios near zero.
USAGE_SMOOTHING = 0.01

# We only downweight a historical pitch if it previously
# represented at least this much of the repertoire.
MIN_PRIOR_USAGE_FOR_DECLINE = 0.03

# Recent pitches that are increasing can receive a boost.
MAX_EMERGING_BOOST = 1.50

# A 20 percentage-point increase receives the maximum 1.5 boost.
BOOST_SCALE = 2.5


def fit_and_score(
    trainer: PitchModelTrainer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Train one model and return train/test accuracy."""

    model = trainer._build_pipeline(X_train)

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

    return (
        float(
            accuracy_score(
                y_train,
                train_predictions,
            )
        ),
        float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        ),
    )


def get_repertoire_context(
    train: pd.DataFrame,
) -> dict | None:
    """
    Build repertoire information using ONLY training data.

    Returns None when there is not enough recent/history data
    to make reliable repertoire adjustments.
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
        train["season"] == latest_season
    ]

    latest_total = len(
        latest_data
    )

    if latest_total < MIN_RECENT_PITCHES:
        return None

    # --------------------------------------------------------
    # Find reliable older seasons
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
        return None

    # Use the previous two reliable seasons as our best
    # estimate of the pitcher's recent historical repertoire.
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

        "prior_seasons":
            prior_seasons,

        "recent_usage":
            recent_usage,

        "prior_usage":
            prior_usage,

        "pitch_types":
            pitch_types,
    }


def build_categorical_repertoire_weights(
    train: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """
    Previous categorical repertoire experiment.

    Old examples receive:
        stable     -> 1.00
        declining  -> 0.50
        inactive   -> 0.10
    """

    weights = np.ones(
        len(train),
        dtype=float,
    )

    context = get_repertoire_context(
        train
    )

    if context is None:
        return (
            weights,
            [],
        )

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

        status = "stable"
        multiplier = 1.0

        # ---------------------------------------------
        # Inactive
        # ---------------------------------------------

        if (
            recent
            < INACTIVE_USAGE_THRESHOLD
            and prior
            >= MEANINGFUL_USAGE_THRESHOLD
        ):
            status = "inactive"
            multiplier = (
                INACTIVE_MULTIPLIER
            )

        # ---------------------------------------------
        # Declining
        # ---------------------------------------------

        elif (
            prior
            >= MEANINGFUL_USAGE_THRESHOLD
            and recent
            < prior * DECLINE_RATIO
        ):
            status = "declining"
            multiplier = (
                DECLINING_MULTIPLIER
            )

        # ---------------------------------------------
        # Apply only to OLD examples
        # ---------------------------------------------

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

            weights[
                mask.to_numpy()
            ] = multiplier

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
                    multiplier,
            }
        )

    return (
        weights,
        diagnostics,
    )


def build_continuous_repertoire_weights(
    train: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """
    Continuous repertoire weighting.

    Declining pitches:
        historical examples are smoothly downweighted based on

            (recent usage + smoothing)
            --------------------------
            (prior usage + smoothing)

    Emerging / increasing pitches:
        current-season examples can receive a modest boost.

    No future/test information is used.
    """

    weights = np.ones(
        len(train),
        dtype=float,
    )

    context = get_repertoire_context(
        train
    )

    if context is None:
        return (
            weights,
            [],
        )

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
        # DECLINING PITCH
        # ====================================================
        #
        # Example:
        #
        # prior  = 20%
        # recent = 10%
        #
        # ratio ≈ 0.52 after smoothing
        #
        # Old examples receive ~0.52 weight.
        # ====================================================

        if (
            prior
            >= MIN_PRIOR_USAGE_FOR_DECLINE
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

        # ====================================================
        # EMERGING / INCREASING PITCH
        # ====================================================
        #
        # Example:
        #
        # prior  = 2%
        # recent = 12%
        #
        # increase = 10 percentage points
        #
        # recent multiplier =
        #     1 + 0.10 * 2.5
        #     = 1.25
        #
        # Maximum boost = 1.50
        # ====================================================

        elif (
            recent > prior
            and recent >= 0.05
        ):

            usage_increase = (
                recent - prior
            )

            recent_multiplier = float(
                np.clip(
                    1.0
                    + usage_increase
                    * BOOST_SCALE,

                    1.0,
                    MAX_EMERGING_BOOST,
                )
            )

            if prior <= 0.01:
                status = "emerging"
            else:
                status = "increasing"

        # ====================================================
        # APPLY HISTORICAL MULTIPLIER
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
        # APPLY CURRENT-SEASON BOOST
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
    """
    Compare four models:

    A. Unweighted
    B. Mild recency
    C. Mild recency + categorical repertoire
    D. Mild recency + continuous repertoire
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
    # IDENTICAL CHRONOLOGICAL 80/20 SPLIT
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
        _,
        unweighted_test_accuracy,
    ) = fit_and_score(
        trainer,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # ========================================================
    # MILD RECENCY WEIGHTS
    # ========================================================

    recency_weights = (
        trainer._season_sample_weights(
            train,
            reference_season=(
                training_reference_season
            ),
        )
    )

    # ========================================================
    # MODEL B: MILD RECENCY ONLY
    # ========================================================

    (
        _,
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
    # MODEL C:
    # MILD RECENCY + CATEGORICAL REPERTOIRE
    # ========================================================

    (
        categorical_weights,
        categorical_diagnostics,
    ) = (
        build_categorical_repertoire_weights(
            train
        )
    )

    categorical_combined_weights = (
        recency_weights
        * categorical_weights
    )

    (
        _,
        categorical_test_accuracy,
    ) = fit_and_score(
        trainer,
        X_train,
        y_train,
        X_test,
        y_test,
        categorical_combined_weights,
    )

    # ========================================================
    # MODEL D:
    # MILD RECENCY + CONTINUOUS REPERTOIRE
    # ========================================================

    (
        continuous_weights,
        continuous_diagnostics,
    ) = (
        build_continuous_repertoire_weights(
            train
        )
    )

    continuous_combined_weights = (
        recency_weights
        * continuous_weights
    )

    (
        _,
        continuous_test_accuracy,
    ) = fit_and_score(
        trainer,
        X_train,
        y_train,
        X_test,
        y_test,
        continuous_combined_weights,
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
    # DIAGNOSTIC STRINGS
    # ========================================================

    categorical_summary = "; ".join(
        (
            f"{item['pitch_type']}:"
            f"{item['status']}"
        )
        for item
        in categorical_diagnostics
    )

    continuous_summary = "; ".join(
        (
            f"{item['pitch_type']}:"
            f"{item['status']}"
            f"(old={item['old_multiplier']:.2f},"
            f"new={item['recent_multiplier']:.2f})"
        )
        for item
        in continuous_diagnostics
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

        "categorical_test_accuracy":
            categorical_test_accuracy,

        "continuous_test_accuracy":
            continuous_test_accuracy,

        "recency_effect_pp":
            (
                recency_test_accuracy
                - unweighted_test_accuracy
            )
            * 100,

        "categorical_effect_pp":
            (
                categorical_test_accuracy
                - unweighted_test_accuracy
            )
            * 100,

        "continuous_effect_pp":
            (
                continuous_test_accuracy
                - unweighted_test_accuracy
            )
            * 100,

        "categorical_added_to_recency_pp":
            (
                categorical_test_accuracy
                - recency_test_accuracy
            )
            * 100,

        "continuous_added_to_recency_pp":
            (
                continuous_test_accuracy
                - recency_test_accuracy
            )
            * 100,

        "continuous_vs_categorical_pp":
            (
                continuous_test_accuracy
                - categorical_test_accuracy
            )
            * 100,

        "categorical_summary":
            categorical_summary,

        "continuous_summary":
            continuous_summary,
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

    print("Loading schemas...")

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
    # MILD RECENCY CONFIGURATION
    # ========================================================

    trainer = PitchModelTrainer(
        starting_decay_factor=0.98,
        decay_per_extra_season=0.01,
        minimum_decay_factor=0.80,
        minimum_sample_weight=0.10,
    )

    # ========================================================
    # GET SAME PROBABLE STARTERS
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
    # EVALUATE
    # ========================================================

    for starter in unique_starters:

        if len(results) >= args.sample_size:
            break

        print()
        print("=" * 75)

        print(
            f"Pitcher "
            f"{len(results) + 1}/"
            f"{args.sample_size}: "
            f"{starter.pitcher_name}"
        )

        print("=" * 75)

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

            print("Cleaning...")

            cleaning = (
                pipeline._clean_pitcher(
                    starter,
                    Path(
                        download.path
                    ),
                )
            )

            print("Creating KG4...")

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
                f"KG4: {len(kg4):,} pitches"
            )

            print(
                "Training 4 comparison models..."
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
                "Unweighted:             "
                f"{result['unweighted_test_accuracy']:.4f}"
            )

            print(
                "Mild recency:           "
                f"{result['recency_test_accuracy']:.4f}"
            )

            print(
                "Categorical repertoire: "
                f"{result['categorical_test_accuracy']:.4f}"
            )

            print(
                "Continuous repertoire:  "
                f"{result['continuous_test_accuracy']:.4f}"
            )

            print(
                "Continuous vs mild:      "
                f"{result['continuous_added_to_recency_pp']:+.2f} pp"
            )

            print(
                "Continuous vs category:  "
                f"{result['continuous_vs_categorical_pp']:+.2f} pp"
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
            "No pitchers successfully evaluated."
        )

    # ========================================================
    # RESULTS TABLE
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
            "categorical_test_accuracy",
            "continuous_test_accuracy",

            "recency_effect_pp",
            "categorical_effect_pp",
            "continuous_effect_pp",

            "categorical_added_to_recency_pp",
            "continuous_added_to_recency_pp",
            "continuous_vs_categorical_pp",

            "categorical_summary",
            "continuous_summary",
        ]
    ]

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    n = len(
        results_df
    )

    continuous_wins_vs_unweighted = int(
        (
            results_df[
                "continuous_effect_pp"
            ]
            > 0
        ).sum()
    )

    continuous_wins_vs_recency = int(
        (
            results_df[
                "continuous_added_to_recency_pp"
            ]
            > 0
        ).sum()
    )

    continuous_wins_vs_categorical = int(
        (
            results_df[
                "continuous_vs_categorical_pp"
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

    mean_categorical = float(
        results_df[
            "categorical_test_accuracy"
        ].mean()
    )

    mean_continuous = float(
        results_df[
            "continuous_test_accuracy"
        ].mean()
    )

    mean_continuous_added = float(
        results_df[
            "continuous_added_to_recency_pp"
        ].mean()
    )

    median_continuous_added = float(
        results_df[
            "continuous_added_to_recency_pp"
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
            "continuous_repertoire_ablation_"
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
                "continuous_repertoire_ablation_"
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
    # PRINT
    # ========================================================

    display = (
        results_df.copy()
    )

    for column in [
        "baseline_accuracy",
        "unweighted_test_accuracy",
        "recency_test_accuracy",
        "categorical_test_accuracy",
        "continuous_test_accuracy",
    ]:

        display[column] = (
            display[column]
            * 100
        ).round(2)

    for column in [
        "recency_effect_pp",
        "categorical_effect_pp",
        "continuous_effect_pp",
        "categorical_added_to_recency_pp",
        "continuous_added_to_recency_pp",
        "continuous_vs_categorical_pp",
    ]:

        display[column] = (
            display[column]
            .round(2)
        )

    print()
    print()
    print("=" * 115)
    print(
        "CONTINUOUS REPERTOIRE ABLATION"
    )
    print("=" * 115)

    print()

    print(
        display[
            [
                "pitcher_name",
                "career_seasons",
                "unweighted_test_accuracy",
                "recency_test_accuracy",
                "categorical_test_accuracy",
                "continuous_test_accuracy",
                "continuous_added_to_recency_pp",
                "continuous_vs_categorical_pp",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("-" * 115)
    print("AGGREGATE RESULTS")
    print("-" * 115)

    print(
        f"Successful pitchers: {n}"
    )

    print()

    print(
        "Continuous beats unweighted: "
        f"{continuous_wins_vs_unweighted}/{n}"
    )

    print(
        "Continuous improves mild recency: "
        f"{continuous_wins_vs_recency}/{n}"
    )

    print(
        "Continuous beats categorical: "
        f"{continuous_wins_vs_categorical}/{n}"
    )

    print()

    print(
        "Mean unweighted accuracy: "
        f"{mean_unweighted * 100:.2f}%"
    )

    print(
        "Mean mild recency accuracy: "
        f"{mean_recency * 100:.2f}%"
    )

    print(
        "Mean categorical accuracy: "
        f"{mean_categorical * 100:.2f}%"
    )

    print(
        "Mean continuous accuracy: "
        f"{mean_continuous * 100:.2f}%"
    )

    print()

    print(
        "Mean continuous benefit over mild: "
        f"{mean_continuous_added:+.2f} pp"
    )

    print(
        "Median continuous benefit over mild: "
        f"{median_continuous_added:+.2f} pp"
    )

    print()

    print(
        f"Results saved to: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()