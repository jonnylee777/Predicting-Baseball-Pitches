"""Persistent performance history for pitch prediction evaluations.

The permanent history file is:

    Data/daily_pipeline/performance_history.csv

Each row represents one evaluated starting-pitcher game.

The unique identifier is:

    game_pk + pitcher_id

Therefore rerunning the same game replaces the old row instead of
creating a duplicate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HISTORY_COLUMNS = [
    "game_date",
    "season",
    "game_pk",
    "pitcher_id",
    "pitcher_name",
    "opponent",
    "pitch_count",
    "model_correct",
    "baseline_correct",
    "model_accuracy",
    "baseline_accuracy",
    "accuracy_over_baseline",
    "accuracy_over_baseline_pp",
    "relative_improvement",
    "relative_improvement_percent",
    "baseline_strategy",
    "model_path",
    "predictions_path",
    "summary_path",
    "evaluated_at_utc",
]


UNIQUE_KEY_COLUMNS = [
    "game_pk",
    "pitcher_id",
]


def empty_history() -> pd.DataFrame:
    """Return an empty history table with the standard columns."""

    return pd.DataFrame(
        columns=HISTORY_COLUMNS
    )


def load_performance_history(
    path: Path,
) -> pd.DataFrame:
    """Load the cumulative history file."""

    path = Path(
        path
    )

    if not path.exists():

        return empty_history()

    history = pd.read_csv(
        path,
        low_memory=False,
    )

    # Make sure future-added columns exist even when reading
    # an older history file.
    for column in HISTORY_COLUMNS:

        if column not in history.columns:

            history[column] = pd.NA

    return history


def upsert_performance_row(
    history: pd.DataFrame,
    row: dict,
) -> pd.DataFrame:
    """
    Insert or replace one pitcher-game.

    The same game_pk + pitcher_id combination can only appear once.

    This makes rerunning a day's replay safe.
    """

    working = history.copy()

    # Ensure all standard columns exist.
    for column in HISTORY_COLUMNS:

        if column not in working.columns:

            working[column] = pd.NA

    new_row = {
        column:
            row.get(
                column,
                pd.NA,
            )
        for column
        in HISTORY_COLUMNS
    }

    new_frame = pd.DataFrame(
        [new_row]
    )

    if working.empty:

        return new_frame

    existing_game_pk = pd.to_numeric(
        working[
            "game_pk"
        ],
        errors="coerce",
    )

    existing_pitcher_id = pd.to_numeric(
        working[
            "pitcher_id"
        ],
        errors="coerce",
    )

    target_game_pk = pd.to_numeric(
        pd.Series(
            [
                new_row[
                    "game_pk"
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]

    target_pitcher_id = pd.to_numeric(
        pd.Series(
            [
                new_row[
                    "pitcher_id"
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]

    duplicate_mask = (
        existing_game_pk.eq(
            target_game_pk
        )
        &
        existing_pitcher_id.eq(
            target_pitcher_id
        )
    )

    working = working.loc[
        ~duplicate_mask
    ].copy()

    result = pd.concat(
        [
            working,
            new_frame,
        ],
        ignore_index=True,
    )

    return result


def normalize_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and sort the history before saving."""

    working = history.copy()

    for column in HISTORY_COLUMNS:

        if column not in working.columns:

            working[column] = pd.NA

    # Keep only the standard public history columns.
    working = working[
        HISTORY_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    working[
        "game_date"
    ] = pd.to_datetime(
        working[
            "game_date"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # SEASON
    # --------------------------------------------------------

    season_from_date = (
        working[
            "game_date"
        ]
        .dt.year
    )

    working[
        "season"
    ] = pd.to_numeric(
        working[
            "season"
        ],
        errors="coerce",
    )

    working[
        "season"
    ] = (
        working[
            "season"
        ]
        .fillna(
            season_from_date
        )
    )

    # --------------------------------------------------------
    # IDS
    # --------------------------------------------------------

    for column in [
        "game_pk",
        "pitcher_id",
    ]:

        working[
            column
        ] = pd.to_numeric(
            working[
                column
            ],
            errors="coerce",
        )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    working = working.dropna(
        subset=[
            "game_date",
            "game_pk",
            "pitcher_id",
        ]
    )

    # --------------------------------------------------------
    # FINAL DEDUPLICATION
    # --------------------------------------------------------

    working = (
        working
        .drop_duplicates(
            subset=(
                UNIQUE_KEY_COLUMNS
            ),
            keep="last",
        )
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    working = (
        working
        .sort_values(
            [
                "game_date",
                "game_pk",
                "pitcher_name",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    # Save dates as YYYY-MM-DD rather than timestamps.
    working[
        "game_date"
    ] = (
        working[
            "game_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # Nullable integer types keep CSV values cleaner.
    working[
        "season"
    ] = (
        working[
            "season"
        ]
        .astype(
            "Int64"
        )
    )

    working[
        "game_pk"
    ] = (
        working[
            "game_pk"
        ]
        .astype(
            "Int64"
        )
    )

    working[
        "pitcher_id"
    ] = (
        working[
            "pitcher_id"
        ]
        .astype(
            "Int64"
        )
    )

    return working


def save_performance_history(
    history: pd.DataFrame,
    path: Path,
) -> None:
    """Save cumulative performance history."""

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized = (
        normalize_history(
            history
        )
    )

    normalized.to_csv(
        path,
        index=False,
    )


def record_performance(
    *,
    path: Path,
    row: dict,
) -> pd.DataFrame:
    """
    Convenience function used by the daily pipeline.

    Load history
        ↓
    insert/replace pitcher-game
        ↓
    save history
        ↓
    return updated history
    """

    history = (
        load_performance_history(
            path
        )
    )

    history = (
        upsert_performance_row(
            history,
            row,
        )
    )

    save_performance_history(
        history,
        path,
    )

    return history