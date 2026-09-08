"""League-wide batter strike-zone reference.

``sz_top`` and ``sz_bot`` are measured on the pitch itself, so they are not
knowable before a pitch is thrown. They are also the third most important
feature group in the production models, which makes guessing them badly
expensive.

Two properties make them recoverable. Within a game a batter's zone is
constant, so once a batter has seen one pitch the exact value is known. Across
games it is a stable physical attribute, so a batter's historical mean is a
close estimate for their first plate appearance.

This module builds that historical mean for every batter in the pipeline's
data, pooled across all pitchers rather than only the pitcher being predicted.
Pooling matters: a starter has often never faced a given batter, in which case
their own history offers nothing and the estimate would fall back to a league
average four times less precise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


REFERENCE_COLUMNS = ("batter", "sz_top", "sz_bot", "pitches", "games")


def build_batter_zone_reference(
    kg4_dir: Path,
    *,
    through_date: date | None = None,
    minimum_pitches: int = 5,
) -> pd.DataFrame:
    """Average each batter's strike zone across every pitcher's history.

    ``through_date`` excludes pitches from that date onward, so a reference
    built for evaluating a game contains no information from the game itself.
    """

    frames: list[pd.DataFrame] = []
    for path in sorted(kg4_dir.glob("*.csv")):
        frame = pd.read_csv(
            path,
            low_memory=False,
            usecols=["batter", "game_pk", "game_date", "sz_top", "sz_bot"],
        )
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        if through_date is not None:
            frame = frame[frame["game_date"].dt.date < through_date]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=list(REFERENCE_COLUMNS))

    pooled = pd.concat(frames, ignore_index=True).dropna(subset=["sz_top", "sz_bot"])

    # A pitch is duplicated when two of the pipeline's pitchers faced the same
    # batter in the same game, so measurements are averaged per batter-game
    # first to keep heavily covered batters from dominating their own mean.
    per_game = (
        pooled.groupby(["batter", "game_pk"])
        .agg(sz_top=("sz_top", "mean"), sz_bot=("sz_bot", "mean"), pitches=("sz_top", "size"))
        .reset_index()
    )

    reference = (
        per_game.groupby("batter")
        .agg(
            sz_top=("sz_top", "mean"),
            sz_bot=("sz_bot", "mean"),
            pitches=("pitches", "sum"),
            games=("game_pk", "nunique"),
        )
        .reset_index()
    )
    reference = reference[reference["pitches"] >= minimum_pitches]
    reference["batter"] = reference["batter"].astype(int)
    return reference.loc[:, list(REFERENCE_COLUMNS)].sort_values("batter")
