"""Offline validation of the live feature path against Savant ground truth.

Live prediction only has value if the features rebuilt from the GUMBO feed
produce the same predictions as the Savant features the models were trained
on. This module replays a completed game through the live code path and
compares, for the same pitches and the same frozen pre-game model:

* how closely each KG4 column matches Savant, and
* whether the resulting predictions and accuracy differ.

Running this before ever connecting to a live game is what makes the live
accuracy number trustworthy rather than a surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier

from ..feature_engineering import KG4_COLUMNS
from ..model import PitchModelTrainer
from .batter_zones import build_batter_zone_reference
from .context import PregameContext
from .features import UNAVAILABLE_LIVE_COLUMNS, LiveFeatureBuilder
from .gumbo import GumboClient

# Numeric columns compared with a tolerance rather than exactly. Savant and the
# live feed round these differently; the tolerance is the largest difference
# that is immaterial to a tree split.
NUMERIC_TOLERANCES: dict[str, float] = {
    # GUMBO reports hitData.totalDistance a few feet above Savant's
    # hit_distance_sc for the same batted ball.
    "hit_distance_sc_of_prev_pitch": 5.0,
    # Savant carries more decimal places on the strike zone than the feed.
    "sz_top": 0.01,
    "sz_bot": 0.01,
    "strike_zone_height": 0.02,
}
DEFAULT_TOLERANCE = 1e-6


@dataclass
class ColumnComparison:
    column: str
    compared: int
    matched: int
    live_missing_only: int
    savant_missing_only: int

    @property
    def match_rate(self) -> float:
        return self.matched / self.compared if self.compared else float("nan")


@dataclass
class GameVerification:
    game_pk: int
    pitcher_id: int
    game_date: str
    pitch_count: int

    savant_accuracy: float
    live_accuracy: float
    baseline_accuracy: float
    prediction_agreement: float

    columns: list[ColumnComparison] = field(default_factory=list)
    translation_warnings: str = "none"
    zone_sources: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy_delta(self) -> float:
        return self.live_accuracy - self.savant_accuracy

    def relative_improvement(self, accuracy: float) -> float | None:
        if self.baseline_accuracy <= 0:
            return None
        return (accuracy - self.baseline_accuracy) / self.baseline_accuracy


class LiveFeatureVerifier:
    """Replay completed games through the live feature path."""

    def __init__(
        self,
        data_root: Path,
        client: GumboClient | None = None,
        trainer: PitchModelTrainer | None = None,
        use_win_probability: bool = False,
    ) -> None:
        self.data_root = data_root
        # Savant's bat_win_exp varies pitch by pitch; the API publishes one
        # value per plate appearance. Feeding that approximation in measured
        # worse than leaving the column missing for the imputer, so it is off
        # by default.
        self.use_win_probability = use_win_probability
        self.client = client or GumboClient()
        self.trainer = trainer or PitchModelTrainer()
        self._zone_reference_cache: dict[date, dict[int, tuple[float, float]]] = {}

    def batter_zone_reference(self, game_date: date) -> dict[int, tuple[float, float]]:
        """League-wide batter zones built only from games before ``game_date``."""

        if game_date not in self._zone_reference_cache:
            frame = build_batter_zone_reference(
                self.data_root / "features" / "kg4" / "pitchers",
                through_date=game_date,
            )
            self._zone_reference_cache[game_date] = {
                int(row.batter): (float(row.sz_top), float(row.sz_bot))
                for row in frame.itertuples()
            }
        return self._zone_reference_cache[game_date]

    # --------------------------------------------------------------

    def history_path(self, pitcher_id: int) -> Path:
        return (
            self.data_root
            / "features"
            / "kg4"
            / "pitchers"
            / f"{pitcher_id}.csv"
        )

    def model_path(self, game_date: date, pitcher_id: int) -> Path:
        return (
            self.data_root
            / "models"
            / game_date.isoformat()
            / "pitchers"
            / f"{pitcher_id}.joblib"
        )

    # --------------------------------------------------------------

    def verify(
        self, *, game_pk: int, pitcher_id: int, game_date: date
    ) -> GameVerification:
        history = pd.read_csv(self.history_path(pitcher_id), low_memory=False)
        history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")

        savant = (
            history[history["game_pk"] == game_pk]
            .sort_values(["at_bat_number_of_game", "pitch_number_of_ab"])
            .reset_index(drop=True)
        )
        if savant.empty:
            raise ValueError(
                f"No Savant rows for game {game_pk} and pitcher {pitcher_id}"
            )

        context = PregameContext.from_history(
            history,
            pitcher_id=pitcher_id,
            game_date=game_date,
            batter_zone_reference=self.batter_zone_reference(game_date),
        )
        snapshot = self.client.feed_live(game_pk)
        builder = LiveFeatureBuilder(
            context,
            win_probability=(
                self.client.win_probability(game_pk)
                if self.use_win_probability
                else None
            ),
        )
        result = builder.build(snapshot, include_pending=False)

        live = (
            result.features
            .assign(_order=range(len(result.features)))
            .sort_values(["at_bat_number_of_game", "pitch_number_of_ab", "_order"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        aligned_live, aligned_savant = _align(live, savant)

        comparisons = [
            _compare_column(column, aligned_live, aligned_savant)
            for column in KG4_COLUMNS
            if column not in {"pitch_type", *UNAVAILABLE_LIVE_COLUMNS}
        ]

        model = joblib.load(self.model_path(game_date, pitcher_id))
        actual = aligned_savant["pitch_type"].astype(str).to_numpy()

        savant_predictions = self._predict(model, aligned_savant)
        live_predictions = self._predict(model, aligned_live)

        baseline_accuracy = self._baseline_accuracy(
            history, game_date, aligned_savant, actual
        )

        return GameVerification(
            game_pk=game_pk,
            pitcher_id=pitcher_id,
            game_date=game_date.isoformat(),
            pitch_count=len(actual),
            savant_accuracy=float((savant_predictions == actual).mean()),
            live_accuracy=float((live_predictions == actual).mean()),
            baseline_accuracy=baseline_accuracy,
            prediction_agreement=float(
                (savant_predictions == live_predictions).mean()
            ),
            columns=comparisons,
            translation_warnings=result.warnings.describe(),
            zone_sources=result.meta["sz_source"].value_counts().to_dict(),
        )

    # --------------------------------------------------------------

    def _predict(self, model: Any, frame: pd.DataFrame) -> np.ndarray:
        features = self.trainer._features(frame)
        if hasattr(model, "feature_names_in_"):
            features = features.reindex(columns=list(model.feature_names_in_))
        return model.predict(features)

    def _baseline_accuracy(
        self,
        history: pd.DataFrame,
        game_date: date,
        game_rows: pd.DataFrame,
        actual: np.ndarray,
    ) -> float:
        """Stratified baseline over the pitcher's pre-game pitch mix."""

        pregame = history[history["game_date"].dt.date < game_date]
        if pregame.empty:
            return float("nan")

        features = self.trainer._features(pregame)
        baseline = DummyClassifier(
            strategy="stratified", random_state=self.trainer.random_state
        )
        baseline.fit(features, pregame["pitch_type"].astype(str))
        predictions = baseline.predict(
            self.trainer._features(game_rows).reindex(columns=features.columns)
        )
        return float((predictions == actual).mean())


# At-bat and pitch number alone are not unique: when a plate appearance is
# interrupted, two batters can share an at-bat number within a game. Including
# the batter makes the key unique across every row in the pipeline's data.
ALIGNMENT_KEYS = ("at_bat_number_of_game", "pitch_number_of_ab", "batter")


def _align(
    live: pd.DataFrame, savant: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match rows on the pitcher's own at-bat, pitch number, and batter."""

    keys = list(ALIGNMENT_KEYS)
    live_keyed = live.set_index(keys)
    savant_keyed = savant.set_index(keys)
    shared = live_keyed.index.intersection(savant_keyed.index)
    ordered = savant_keyed.index[savant_keyed.index.isin(shared)]
    return (
        live_keyed.loc[ordered].reset_index(),
        savant_keyed.loc[ordered].reset_index(),
    )


def _compare_column(
    column: str, live: pd.DataFrame, savant: pd.DataFrame
) -> ColumnComparison:
    left, right = live[column], savant[column]
    left_missing, right_missing = left.isna(), right.isna()
    both_present = ~left_missing & ~right_missing

    if pd.api.types.is_numeric_dtype(right) and column != "game_date":
        tolerance = NUMERIC_TOLERANCES.get(column, DEFAULT_TOLERANCE)
        left_values = pd.to_numeric(left, errors="coerce")
        right_values = pd.to_numeric(right, errors="coerce")
        matched = int(
            np.isclose(
                left_values[both_present],
                right_values[both_present],
                rtol=0,
                atol=tolerance,
            ).sum()
        )
    else:
        matched = int(
            (left[both_present].astype(str) == right[both_present].astype(str)).sum()
        )

    return ColumnComparison(
        column=column,
        compared=int(both_present.sum()),
        matched=matched,
        live_missing_only=int((left_missing & ~right_missing).sum()),
        savant_missing_only=int((~left_missing & right_missing).sum()),
    )
