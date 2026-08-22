from pathlib import Path

import pandas as pd

from pitch_prediction.performance_history import (
    HISTORY_COLUMNS,
    load_performance_history,
    normalize_history,
    record_performance,
    upsert_performance_row,
)


def sample_row(
    *,
    game_pk=1001,
    pitcher_id=2001,
    game_date="2026-08-20",
    model_accuracy=0.60,
):
    return {
        "game_date": game_date,
        "season": 2026,
        "game_pk": game_pk,
        "pitcher_id": pitcher_id,
        "pitcher_name": "Test Pitcher",
        "opponent": "Test Opponent",
        "pitch_count": 100,
        "model_correct": 60,
        "baseline_correct": 40,
        "model_accuracy": model_accuracy,
        "baseline_accuracy": 0.40,
        "accuracy_over_baseline": model_accuracy - 0.40,
        "accuracy_over_baseline_pp": (
            model_accuracy - 0.40
        ) * 100,
        "relative_improvement": (
            model_accuracy - 0.40
        ) / 0.40,
        "relative_improvement_percent": (
            (
                model_accuracy - 0.40
            )
            / 0.40
        ) * 100,
        "baseline_strategy": "stratified",
        "model_version": "rf_v1",
        "feature_version": "kg4_v1",
        "evaluation_version": "postgame_v1",
        "model_path": "model.joblib",
        "predictions_path": "predictions.csv",
        "summary_path": "summary.json",
        "evaluated_at_utc": (
            "2026-08-21T12:00:00+00:00"
        ),
    }


def test_history_contains_version_columns():

    assert "model_version" in HISTORY_COLUMNS
    assert "feature_version" in HISTORY_COLUMNS
    assert "evaluation_version" in HISTORY_COLUMNS


def test_upsert_replaces_same_pitcher_game():

    history = pd.DataFrame(
        [sample_row()]
    )

    replacement = sample_row(
        model_accuracy=0.70
    )

    updated = upsert_performance_row(
        history,
        replacement,
    )

    assert len(updated) == 1

    assert (
        updated.iloc[0]["model_accuracy"]
        == 0.70
    )


def test_different_games_are_preserved():

    history = pd.DataFrame(
        [sample_row()]
    )

    second = sample_row(
        game_pk=1002,
        game_date="2026-08-21",
    )

    updated = upsert_performance_row(
        history,
        second,
    )

    assert len(updated) == 2


def test_history_sorted_chronologically():

    history = pd.DataFrame(
        [
            sample_row(
                game_pk=1002,
                game_date="2026-08-21",
            ),
            sample_row(
                game_pk=1001,
                game_date="2026-08-20",
            ),
        ]
    )

    normalized = normalize_history(
        history
    )

    assert list(
        normalized["game_date"]
    ) == [
        "2026-08-20",
        "2026-08-21",
    ]


def test_missing_season_is_derived_from_date():

    row = sample_row()

    row["season"] = None

    normalized = normalize_history(
        pd.DataFrame(
            [row]
        )
    )

    assert (
        normalized.iloc[0]["season"]
        == 2026
    )


def test_record_performance_is_rerun_safe(
    tmp_path: Path,
):

    path = (
        tmp_path
        / "performance_history.csv"
    )

    record_performance(
        path=path,
        row=sample_row(
            model_accuracy=0.60
        ),
    )

    record_performance(
        path=path,
        row=sample_row(
            model_accuracy=0.70
        ),
    )

    history = load_performance_history(
        path
    )

    assert len(history) == 1

    assert (
        history.iloc[0]["model_accuracy"]
        == 0.70
    )


def test_old_history_without_version_columns_loads(
    tmp_path: Path,
):

    path = (
        tmp_path
        / "old_history.csv"
    )

    old_row = sample_row()

    old_row.pop(
        "model_version"
    )

    old_row.pop(
        "feature_version"
    )

    old_row.pop(
        "evaluation_version"
    )

    pd.DataFrame(
        [old_row]
    ).to_csv(
        path,
        index=False,
    )

    history = load_performance_history(
        path
    )

    assert "model_version" in history.columns
    assert "feature_version" in history.columns
    assert "evaluation_version" in history.columns