"""Pre-game context that the live feed cannot supply on its own.

A handful of KG4 features summarise information from before the first pitch:
career matchup history, the pitcher's rest, and each batter's strike zone.
These are read once from the pitcher's Savant history and then advanced
incrementally as the game unfolds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


# Fallback used only for a batter with no prior Statcast history, such as a
# major-league debut. Derived from 2025-2026 pitch data.
LEAGUE_SZ_TOP = 3.328
LEAGUE_SZ_BOT = 1.615


@dataclass
class PregameContext:
    """Everything known about a pitcher-game before the first pitch."""

    pitcher_id: int
    season: int
    game_date: date

    # Each batter's strike zone is a stable physical attribute. Within a game
    # it is effectively constant, so the live builder prefers a value observed
    # earlier in the same game and falls back to this historical mean for a
    # batter's first plate appearance.
    batter_zones: dict[int, tuple[float, float]] = field(default_factory=dict)

    # League-wide batter zones, pooled across all pitchers. Consulted when the
    # pitcher being predicted has never faced this batter, which is common.
    batter_zone_reference: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )

    # Plate appearances this pitcher has faced against each batter, career, up
    # to but not including the current game.
    career_pa_vs_batter: dict[int, int] = field(default_factory=dict)

    # The pitcher's final pitch types before this game, oldest first. Seeds the
    # prev3 rolling rates, which are computed per pitcher across games rather
    # than being reset each start.
    seed_pitch_types: tuple[str, ...] = ()

    pitcher_days_since_prev_game: float | None = None
    pitcher_birth_year: int | None = None

    # The pitcher's pre-game pitch mix. The project's baseline is a stratified
    # draw from this distribution, so its expected accuracy against a game can
    # be computed directly rather than sampled, which keeps the live
    # relative-improvement figure free of run-to-run randomness.
    pitch_type_prior: dict[str, float] = field(default_factory=dict)

    def expected_baseline_accuracy(self, actual: list[str]) -> float | None:
        """Expected accuracy of a stratified guesser over ``actual`` pitches.

        A stratified guesser predicts class i with probability ``prior[i]``, so
        its chance of being right on a pitch of class i is ``prior[i]``. Over a
        set of pitches that is the mean prior of the classes actually thrown.
        """

        if not self.pitch_type_prior or not actual:
            return None
        return sum(self.pitch_type_prior.get(pitch, 0.0) for pitch in actual) / len(
            actual
        )

    def batter_zone(self, batter_id: int) -> tuple[float, float]:
        """Best available estimate of a batter's zone before their first pitch."""

        zone = self.batter_zones.get(batter_id)
        if zone is not None:
            return zone
        zone = self.batter_zone_reference.get(batter_id)
        if zone is not None:
            return zone
        return (LEAGUE_SZ_TOP, LEAGUE_SZ_BOT)

    def batter_zone_source(self, batter_id: int) -> str:
        if batter_id in self.batter_zones:
            return "pitcher_history"
        if batter_id in self.batter_zone_reference:
            return "league_reference"
        return "league_average"

    def has_batter_zone(self, batter_id: int) -> bool:
        return (
            batter_id in self.batter_zones
            or batter_id in self.batter_zone_reference
        )

    @classmethod
    def from_history(
        cls,
        history: pd.DataFrame,
        *,
        pitcher_id: int,
        game_date: date,
        batter_zone_reference: dict[int, tuple[float, float]] | None = None,
    ) -> "PregameContext":
        """Build context from KG4-shaped history, excluding the target game.

        ``history`` may contain the target game; rows on or after
        ``game_date`` are dropped so that no in-game information leaks into
        pre-game context.
        """

        frame = history.copy()
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        prior = frame[frame["game_date"].dt.date < game_date]

        context = cls(
            pitcher_id=pitcher_id,
            season=game_date.year,
            game_date=game_date,
            batter_zone_reference=batter_zone_reference or {},
        )

        if prior.empty:
            return context

        zones = prior.groupby("batter")[["sz_top", "sz_bot"]].mean()
        context.batter_zones = {
            int(batter): (float(row.sz_top), float(row.sz_bot))
            for batter, row in zones.iterrows()
            if pd.notna(row.sz_top) and pd.notna(row.sz_bot)
        }

        plate_appearances = prior.drop_duplicates(
            subset=["game_pk", "at_bat_number_of_game"]
        )
        context.career_pa_vs_batter = {
            int(batter): int(count)
            for batter, count in plate_appearances["batter"].value_counts().items()
        }

        ordered = prior.sort_values(
            ["game_date", "game_pk", "at_bat_number_of_game", "pitch_number_of_ab"]
        )
        context.seed_pitch_types = tuple(
            str(value) for value in ordered["pitch_type"].dropna().tail(3)
        )

        mix = prior["pitch_type"].dropna().astype(str).value_counts(normalize=True)
        context.pitch_type_prior = {str(k): float(v) for k, v in mix.items()}

        last_game = ordered["game_date"].max()
        if pd.notna(last_game):
            context.pitcher_days_since_prev_game = float(
                (game_date - last_game.date()).days
            )

        return context
