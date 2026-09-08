"""GUMBO to Statcast vocabulary translation.

The live MLB feed (statsapi ``feed/live``) and the Baseball Savant CSV export
describe the same events with different vocabularies. The production models are
trained on Savant values, so every live value must be translated before it can
be handed to a model. A silent mismatch is worse than a crash here: an unmapped
category is imputed by the one-hot encoder and quietly degrades predictions
instead of failing, so unknown values are surfaced through
:class:`TranslationWarnings` rather than being dropped.

Every mapping below was derived from real feeds, not from documentation. See
``scripts/verify_live_features.py`` for the audit that keeps them honest.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


# ------------------------------------------------------------------
# PITCH RESULT
# ------------------------------------------------------------------
# GUMBO ``details.call.code`` -> Savant ``description``.
#
# The first twelve entries are observed in live 2026 regular-season feeds. The
# remainder are rarer calls kept so that an unusual game does not produce an
# unknown code mid-inning.
CALL_CODE_TO_DESCRIPTION: dict[str, str] = {
    "B": "ball",
    "*B": "blocked_ball",
    "C": "called_strike",
    "S": "swinging_strike",
    "W": "swinging_strike_blocked",
    "F": "foul",
    "T": "foul_tip",
    "L": "foul_bunt",
    "X": "hit_into_play",
    "D": "hit_into_play",
    "E": "hit_into_play",
    "H": "hit_by_pitch",
    # Rarer / non-standard calls.
    "M": "missed_bunt",
    "O": "bunt_foul_tip",
    "P": "pitchout",
    "I": "intent_ball",
    "V": "intent_ball",
    "Q": "swinging_strike",
    "R": "foul",
    "Y": "hit_into_play",
    "Z": "hit_into_play",
    # Automated ball-strike challenge outcomes (2026).
    "AB": "ball",
    "AC": "called_strike",
}

# Savant ``description`` -> Savant ``type`` (B / S / X).
DESCRIPTION_TO_TYPE: dict[str, str] = {
    "ball": "B",
    "blocked_ball": "B",
    "intent_ball": "B",
    "pitchout": "B",
    "hit_by_pitch": "B",
    "called_strike": "S",
    "swinging_strike": "S",
    "swinging_strike_blocked": "S",
    "foul": "S",
    "foul_tip": "S",
    "foul_bunt": "S",
    "bunt_foul_tip": "S",
    "missed_bunt": "S",
    "hit_into_play": "X",
}


# ------------------------------------------------------------------
# BATTED BALL TYPE
# ------------------------------------------------------------------
# GUMBO ``hitData.trajectory`` -> Savant ``bb_type``.
#
# Savant has no bunt categories, so bunts collapse into their trajectory.
TRAJECTORY_TO_BB_TYPE: dict[str, str] = {
    "ground_ball": "ground_ball",
    "line_drive": "line_drive",
    "fly_ball": "fly_ball",
    "popup": "popup",
    "bunt_grounder": "ground_ball",
    "bunt_popup": "popup",
    "bunt_line_drive": "line_drive",
    "bunt_fly_ball": "fly_ball",
}


# ------------------------------------------------------------------
# PLATE APPEARANCE RESULT
# ------------------------------------------------------------------
# Savant ``events`` values that terminate a plate appearance. GUMBO's
# ``result.eventType`` uses the same vocabulary, so this is an allow-list
# rather than a rename: it filters out mid-at-bat events such as ``pickoff_1b``
# and ``stolen_base_2b`` that never appear in Savant's ``events`` column.
PLATE_APPEARANCE_EVENTS: frozenset[str] = frozenset(
    {
        "single",
        "double",
        "triple",
        "home_run",
        "walk",
        "intent_walk",
        "strikeout",
        "strikeout_double_play",
        "strikeout_triple_play",
        "hit_by_pitch",
        "field_out",
        "force_out",
        "field_error",
        "fielders_choice",
        "fielders_choice_out",
        "double_play",
        "triple_play",
        "grounded_into_double_play",
        "sac_fly",
        "sac_fly_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
        "catcher_interf",
        "truncated_pa",
        "batter_interference",
        "fan_interference",
        "other_out",
    }
)


@dataclass
class TranslationWarnings:
    """Collects unmapped feed values instead of failing mid-game."""

    unknown_call_codes: set[str] = field(default_factory=set)
    unknown_trajectories: set[str] = field(default_factory=set)
    unknown_events: set[str] = field(default_factory=set)
    unmapped_hit_locations: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not (
            self.unknown_call_codes
            or self.unknown_trajectories
            or self.unknown_events
            or self.unmapped_hit_locations
        )

    def describe(self) -> str:
        parts = []
        if self.unknown_call_codes:
            parts.append(f"call codes={sorted(self.unknown_call_codes)}")
        if self.unknown_trajectories:
            parts.append(f"trajectories={sorted(self.unknown_trajectories)}")
        if self.unknown_events:
            parts.append(f"events={sorted(self.unknown_events)}")
        if self.unmapped_hit_locations:
            parts.append(f"hit locations={sorted(self.unmapped_hit_locations)}")
        return "; ".join(parts) if parts else "none"


def pitch_description(call_code: str | None, warnings: TranslationWarnings) -> str | None:
    """Translate a GUMBO call code into a Savant ``description``."""

    if call_code is None:
        return None
    description = CALL_CODE_TO_DESCRIPTION.get(call_code)
    if description is None:
        warnings.unknown_call_codes.add(call_code)
    return description


def pitch_type_code(description: str | None) -> str | None:
    """Translate a Savant ``description`` into a Savant ``type`` (B/S/X)."""

    if description is None:
        return None
    return DESCRIPTION_TO_TYPE.get(description)


def batted_ball_type(trajectory: str | None, warnings: TranslationWarnings) -> str | None:
    if not trajectory:
        return None
    bb_type = TRAJECTORY_TO_BB_TYPE.get(trajectory)
    if bb_type is None:
        warnings.unknown_trajectories.add(trajectory)
    return bb_type


def hit_location(location: str | int | None, warnings: TranslationWarnings) -> float | None:
    """Translate ``hitData.location`` into Savant's numeric ``hit_location``.

    GUMBO also reports gap locations such as ``"78"`` and ``"89"`` that have no
    Savant equivalent, because Savant records the fielder credited with the
    ball. Those become missing rather than being coerced to a single fielder.
    """

    if location is None or location == "":
        return None
    text = str(location).strip()
    if text in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
        return float(text)
    warnings.unmapped_hit_locations.add(text)
    return None


def plate_appearance_event(
    event_type: str | None, warnings: TranslationWarnings
) -> str | None:
    """Return ``event_type`` when it is a Savant plate-appearance result."""

    if not event_type:
        return None
    if event_type in PLATE_APPEARANCE_EVENTS:
        return event_type
    warnings.unknown_events.add(event_type)
    return None


# ------------------------------------------------------------------
# LAUNCH SPEED / ANGLE CLASSIFICATION
# ------------------------------------------------------------------
# Savant's ``launch_speed_angle`` is a 1-6 batted-ball quality bucket (weak,
# topped, under, flare/burner, solid contact, barrel) that the live feed does
# not publish. It is a deterministic function of exit velocity and launch
# angle, so it is recovered from a lookup table rather than dropped.
#
# The table in ``config/launch_speed_angle_grid.csv`` was built by binning
# 107,455 historical batted balls to 1 mph x 1 degree. Mean cell purity is
# 0.9989, confirming the function is deterministic, and the table reproduces
# Savant's value on 99.8% of held-out batted balls. Cells outside the table
# fall back to the nearest populated cell.
LAUNCH_SPEED_ANGLE_GRID_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "launch_speed_angle_grid.csv"
)


@lru_cache(maxsize=1)
def _launch_speed_angle_grid() -> dict[tuple[int, int], int]:
    if not LAUNCH_SPEED_ANGLE_GRID_PATH.exists():
        return {}
    grid: dict[tuple[int, int], int] = {}
    with LAUNCH_SPEED_ANGLE_GRID_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["launch_speed_mph"]), int(row["launch_angle_deg"]))
            grid[key] = int(row["launch_speed_angle"])
    return grid


def launch_speed_angle(
    launch_speed: float | None, launch_angle: float | None
) -> float | None:
    """Recover Savant's 1-6 batted-ball classification from exit velo/angle."""

    if launch_speed is None or launch_angle is None:
        return None
    try:
        speed = float(launch_speed)
        angle = float(launch_angle)
    except (TypeError, ValueError):
        return None
    if speed != speed or angle != angle:  # NaN guard
        return None

    grid = _launch_speed_angle_grid()
    if not grid:
        return None

    key = (round(speed), round(angle))
    exact = grid.get(key)
    if exact is not None:
        return float(exact)

    # Nearest populated cell. The table covers the realistic batted-ball
    # envelope, so this only fires on extreme outliers.
    target_speed, target_angle = key
    nearest = min(
        grid,
        key=lambda cell: (cell[0] - target_speed) ** 2 + (cell[1] - target_angle) ** 2,
    )
    return float(grid[nearest])
