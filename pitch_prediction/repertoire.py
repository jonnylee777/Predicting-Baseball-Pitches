"""Continuous pitch-repertoire weighting.

This module measures how a pitcher's pitch mix has changed over time.

It is intentionally separate from model.py so that:
1. repertoire logic can be tested independently,
2. model.py stays focused on machine-learning training,
3. future models can reuse the same repertoire analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RepertoireSettings:
    """Settings frozen from the repertoire experiments."""

    min_recent_pitches: int = 300
    min_older_season_pitches: int = 300

    # Historical examples of pitches that are declining
    # are smoothly reduced using the usage ratio.
    min_prior_usage_for_decline: float = 0.03
    usage_smoothing: float = 0.01
    minimum_old_multiplier: float = 0.15

    # Increasing/current-season pitches can receive a small boost.
    minimum_recent_usage_for_increase: float = 0.05
    maximum_recent_multiplier: float = 1.50
    increase_boost_scale: float = 2.5


@dataclass(frozen=True)
class RepertoireAdjustment:
    pitch_type: str
    recent_usage: float
    prior_usage: float
    status: str
    old_multiplier: float
    recent_multiplier: float


def continuous_repertoire_weights(
    data: pd.DataFrame,
    settings: RepertoireSettings | None = None,
) -> tuple[np.ndarray, list[RepertoireAdjustment]]:
    """
    Return one repertoire weight per pitch.

    IMPORTANT:
    `data` must contain only information that is legally available
    at training time.

    During historical evaluation this means TRAINING DATA ONLY.
    """

    settings = settings or RepertoireSettings()

    required = {"season", "pitch_type"}

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Repertoire weighting requires columns: {sorted(missing)}"
        )

    weights = np.ones(
        len(data),
        dtype=float,
    )

    seasons = sorted(
        data["season"]
        .dropna()
        .astype(int)
        .unique()
    )

    # We cannot measure repertoire change with only one season.
    if len(seasons) < 2:
        return weights, []

    latest_season = seasons[-1]

    latest_data = data[
        data["season"] == latest_season
    ]

    # Do not draw repertoire conclusions from a tiny current sample.
    if len(latest_data) < settings.min_recent_pitches:
        return weights, []

    reliable_older_seasons = []

    for season in seasons[:-1]:

        season_count = int(
            (
                data["season"] == season
            ).sum()
        )

        if season_count >= settings.min_older_season_pitches:
            reliable_older_seasons.append(
                season
            )

    if not reliable_older_seasons:
        return weights, []

    # The two most recent reliable prior seasons represent
    # the pitcher's recent historical repertoire.
    prior_seasons = reliable_older_seasons[-2:]

    prior_data = data[
        data["season"].isin(
            prior_seasons
        )
    ]

    recent_usage = (
        latest_data["pitch_type"]
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )

    prior_usage = (
        prior_data["pitch_type"]
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )

    pitch_types = sorted(
        data["pitch_type"]
        .astype(str)
        .unique()
    )

    diagnostics: list[RepertoireAdjustment] = []

    pitch_type_series = (
        data["pitch_type"]
        .astype(str)
    )

    for pitch_type in pitch_types:

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

        if (
            prior >= settings.min_prior_usage_for_decline
            and recent < prior
        ):

            usage_ratio = (
                (
                    recent
                    + settings.usage_smoothing
                )
                /
                (
                    prior
                    + settings.usage_smoothing
                )
            )

            old_multiplier = float(
                np.clip(
                    usage_ratio,
                    settings.minimum_old_multiplier,
                    1.0,
                )
            )

            if recent < 0.01:
                status = "nearly_inactive"
            else:
                status = "declining"

        # ====================================================
        # INCREASING / EMERGING PITCH
        # ====================================================

        elif (
            recent > prior
            and recent
            >= settings.minimum_recent_usage_for_increase
        ):

            increase = recent - prior

            recent_multiplier = float(
                np.clip(
                    1.0
                    + increase
                    * settings.increase_boost_scale,
                    1.0,
                    settings.maximum_recent_multiplier,
                )
            )

            if prior <= 0.01:
                status = "emerging"
            else:
                status = "increasing"

        # ====================================================
        # APPLY HISTORICAL DOWNWEIGHTING
        # ====================================================

        if old_multiplier < 1.0:

            old_mask = (
                pitch_type_series.eq(
                    pitch_type
                )
                &
                data["season"].lt(
                    latest_season
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
                pitch_type_series.eq(
                    pitch_type
                )
                &
                data["season"].eq(
                    latest_season
                )
            )

            weights[
                recent_mask.to_numpy()
            ] = recent_multiplier

        diagnostics.append(
            RepertoireAdjustment(
                pitch_type=pitch_type,
                recent_usage=recent,
                prior_usage=prior,
                status=status,
                old_multiplier=old_multiplier,
                recent_multiplier=recent_multiplier,
            )
        )

    return weights, diagnostics


def diagnostics_to_dicts(
    diagnostics: list[RepertoireAdjustment],
) -> list[dict]:
    """Convert diagnostics into JSON-safe dictionaries."""

    return [
        asdict(item)
        for item in diagnostics
    ]