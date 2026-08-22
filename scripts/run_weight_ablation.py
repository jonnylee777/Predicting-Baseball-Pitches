"""Compare weighted vs unweighted Random Forest training on the same KG4 data."""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from pitch_prediction.schema import StatcastSchema
from pitch_prediction.cleaning import PitchDataCleaner
from pitch_prediction.feature_engineering import PitchFeatureEngineer
from pitch_prediction.model import PitchModelTrainer


PITCHER_ID = 592332
PITCHER_NAME = "Kevin Gausman"

RAW_DATA_PATH = Path(
    "Data/592332_data-2.csv"
)


def main() -> None:

    # ---------------------------------------------------------
    # 1. Load and create KG4 exactly like the real pipeline
    # ---------------------------------------------------------

    print("1. Loading schemas...")

    raw_schema = StatcastSchema.from_file(
        Path("config/statcast_columns.txt")
    )

    cleaned_schema = StatcastSchema.from_file(
        Path("config/cleaned_columns.txt")
    )

    print("2. Loading Kevin Gausman data...")

    raw = pd.read_csv(
        RAW_DATA_PATH,
        low_memory=False,
    )

    print("3. Cleaning data...")

    cleaner = PitchDataCleaner(
        raw_schema,
        cleaned_schema,
    )

    cleaned = cleaner.transform(
        raw,
        str(RAW_DATA_PATH),
    )

    print("4. Creating KG4...")

    feature_engineer = PitchFeatureEngineer(
        cleaned_schema
    )

    datasets = feature_engineer.transform(
        cleaned,
        "Kevin Gausman ablation",
    )

    kg4 = datasets.kg4.copy()

    print(
        f"KG4: {len(kg4):,} pitches, "
        f"{len(kg4.columns)} columns"
    )

    # ---------------------------------------------------------
    # 2. Prepare the data exactly like model.py
    # ---------------------------------------------------------

    trainer = PitchModelTrainer()

    data = kg4.dropna(
        subset=["pitch_type"]
    ).copy()

    data["game_date"] = pd.to_datetime(
        data["game_date"],
        errors="raise",
    )

    data["season"] = (
        data["game_date"].dt.year
    )

    game_order = trainer._ordered_games(
        data
    )

    # Same chronological 80/20 split used by model.py.
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

    print()
    print("Chronological split")
    print("------------------------------")
    print(
        f"Training games: {len(train_game_ids)}"
    )
    print(
        f"Testing games: {len(test_game_ids)}"
    )
    print(
        f"Training pitches: {len(train):,}"
    )
    print(
        f"Testing pitches: {len(test):,}"
    )
    print(
        f"Training seasons: "
        f"{train['season'].min()}-"
        f"{train['season'].max()}"
    )
    print(
        f"Testing seasons: "
        f"{test['season'].min()}-"
        f"{test['season'].max()}"
    )

    # ---------------------------------------------------------
    # 3. MODEL A: NO RECENCY WEIGHTING
    # ---------------------------------------------------------

    print()
    print("5. Training UNWEIGHTED model...")

    unweighted_model = trainer._build_pipeline(
        X_train
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

    unweighted_train_accuracy = (
        accuracy_score(
            y_train,
            unweighted_train_predictions,
        )
    )

    unweighted_test_accuracy = (
        accuracy_score(
            y_test,
            unweighted_test_predictions,
        )
    )

    # ---------------------------------------------------------
    # 4. MODEL B: RECENCY WEIGHTING
    # ---------------------------------------------------------

    print(
        "6. Training RECENCY-WEIGHTED model..."
    )

    weighted_model = trainer._build_pipeline(
        X_train
    )

    training_reference_season = int(
        train["season"].max()
    )

    sample_weights = (
        trainer._season_sample_weights(
            train,
            reference_season=(
                training_reference_season
            ),
        )
    )

    weighted_model.fit(
        X_train,
        y_train,
        classifier__sample_weight=(
            sample_weights
        ),
    )

    weighted_train_predictions = (
        weighted_model.predict(
            X_train
        )
    )

    weighted_test_predictions = (
        weighted_model.predict(
            X_test
        )
    )

    weighted_train_accuracy = (
        accuracy_score(
            y_train,
            weighted_train_predictions,
        )
    )

    weighted_test_accuracy = (
        accuracy_score(
            y_test,
            weighted_test_predictions,
        )
    )

    # ---------------------------------------------------------
    # 5. Baseline
    # ---------------------------------------------------------

    majority_class = (
        y_train.mode().iloc[0]
    )

    baseline_predictions = np.full(
        len(y_test),
        majority_class,
        dtype=object,
    )

    baseline_accuracy = (
        accuracy_score(
            y_test,
            baseline_predictions,
        )
    )

    # ---------------------------------------------------------
    # 6. Compare
    # ---------------------------------------------------------

    difference = (
        weighted_test_accuracy
        - unweighted_test_accuracy
    )

    print()
    print("=" * 50)
    print("ABLATION RESULTS")
    print("=" * 50)

    print()
    print("UNWEIGHTED RANDOM FOREST")
    print(
        f"Train accuracy: "
        f"{unweighted_train_accuracy:.4f}"
    )
    print(
        f"Test accuracy:  "
        f"{unweighted_test_accuracy:.4f}"
    )

    print()
    print("RECENCY-WEIGHTED RANDOM FOREST")
    print(
        f"Train accuracy: "
        f"{weighted_train_accuracy:.4f}"
    )
    print(
        f"Test accuracy:  "
        f"{weighted_test_accuracy:.4f}"
    )

    print()
    print("BASELINE")
    print(
        f"Test accuracy:  "
        f"{baseline_accuracy:.4f}"
    )

    print()
    print("WEIGHTING EFFECT")
    print(
        f"Weighted - Unweighted: "
        f"{difference:+.4f}"
    )

    print()

    if difference > 0:
        print(
            "RESULT: Recency weighting improved "
            "test accuracy."
        )

    elif difference < 0:
        print(
            "RESULT: Recency weighting reduced "
            "test accuracy."
        )

    else:
        print(
            "RESULT: Recency weighting made "
            "no difference."
        )

    print()
    print("Season weights used:")

    weight_table = (
        trainer._season_weight_table(
            train,
            reference_season=(
                training_reference_season
            ),
        )
    )

    for season, weight in (
        weight_table.items()
    ):
        print(
            f"  {season}: {weight:.4f}"
        )


if __name__ == "__main__":
    main()