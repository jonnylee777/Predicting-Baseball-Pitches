"""Combine what was predicted live with what was replayed afterwards.

The project predicts pitches two ways, and they cover different ground:

* **Live** -- predicted before the pitch was thrown, from the MLB feed. It
  covers most pitches but not all: a pitch whose situation could not be
  anticipated, or that the feed published together with another, gets no live
  prediction.
* **Replay** -- the postgame pass over the completed game. It covers *every*
  pitch, using the same frozen pre-game model, but it is not a live prediction.

Neither alone is the whole story. This module joins them into one row per
pitch, so a game log can show what was called before the pitch and fill the
gaps with the replay, marking which is which.

The two sources are joined on the pitcher's own pitch count within the game,
which both compute the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("Data/daily_pipeline")

JOIN_KEYS = ("game_pk", "pitcher", "pitch_number_of_game")


@dataclass(frozen=True)
class GameLogPaths:
    replay: Path
    live: Path

    @classmethod
    def for_game(
        cls,
        game_pk: int,
        pitcher_id: int,
        game_date: str,
        data_root: Path = DATA_ROOT,
    ) -> "GameLogPaths":
        return cls(
            replay=(
                data_root
                / "predictions"
                / "postgame"
                / game_date
                / f"{game_pk}_{pitcher_id}.csv"
            ),
            live=(
                data_root
                / "predictions"
                / "live"
                / f"slate_{game_pk}_{pitcher_id}_live.jsonl"
            ),
        )


def load_live(path: Path) -> pd.DataFrame:
    """Live predictions for one pitcher-game, one row per scored pitch."""

    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keep = {
        "game_pk": "game_pk",
        "pitcher_id": "pitcher",
        "pitch_number_of_game": "pitch_number_of_game",
        "predicted_pitch_type": "live_prediction",
        "correct": "live_correct",
        "confidence": "live_confidence",
        "timing": "live_timing",
        "prediction_lead_seconds": "live_lead_seconds",
        "predicted_ahead": "live_predicted_ahead",
    }
    present = {k: v for k, v in keep.items() if k in frame.columns}
    return frame.loc[:, list(present)].rename(columns=present)


def load_replay(path: Path) -> pd.DataFrame:
    """Postgame predictions for one pitcher-game, one row per pitch thrown."""

    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    keep = {
        "game_pk": "game_pk",
        "pitcher": "pitcher",
        "pitcher_name": "pitcher_name",
        "pitch_number_of_game": "pitch_number_of_game",
        "at_bat_number_of_game": "at_bat_number_of_game",
        "pitch_number_of_ab": "pitch_number_of_ab",
        "inning": "inning",
        "inning_topbot": "inning_topbot",
        "balls": "balls",
        "strikes": "strikes",
        "count": "count",
        "outs_when_up": "outs",
        "batter": "batter",
        "actual_pitch": "actual_pitch",
        "model_prediction": "replay_prediction",
        "model_correct": "replay_correct",
        "model_confidence": "replay_confidence",
        "baseline_prediction": "baseline_prediction",
        "baseline_correct": "baseline_correct",
    }
    present = {k: v for k, v in keep.items() if k in frame.columns}
    return frame.loc[:, list(present)].rename(columns=present)


def combine(replay: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """One row per pitch: the replay record, overlaid with the live call.

    The replay is the spine because it covers every pitch. ``source`` says
    which prediction a viewer would actually have seen before the pitch:

    * ``live``   -- predicted before the pitch was thrown
    * ``late``   -- predicted live, but the prediction arrived after the pitch
    * ``replay`` -- no live prediction; only the postgame pass covers it
    """

    if replay.empty:
        return replay

    keys = [key for key in JOIN_KEYS if key in replay.columns]
    combined = (
        replay.merge(live, on=keys, how="left")
        if not live.empty
        else replay.assign(
            live_prediction=pd.NA,
            live_correct=pd.NA,
            live_timing=pd.NA,
            live_lead_seconds=pd.NA,
            live_predicted_ahead=pd.NA,
        )
    )

    def source(row: pd.Series) -> str:
        if pd.isna(row.get("live_prediction")):
            return "replay"
        return "live" if row.get("live_timing") == "before_pitch" else "late"

    combined["source"] = combined.apply(source, axis=1)
    # What a viewer would have seen before the pitch, where anything existed.
    combined["shown_prediction"] = combined.apply(
        lambda r: r["live_prediction"] if r["source"] == "live" else pd.NA, axis=1
    )
    return combined.sort_values("pitch_number_of_game").reset_index(drop=True)


def load_game_log(
    game_pk: int,
    pitcher_id: int,
    game_date: str,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    paths = GameLogPaths.for_game(game_pk, pitcher_id, game_date, data_root)
    return combine(load_replay(paths.replay), load_live(paths.live))


def coverage(combined: pd.DataFrame) -> dict:
    """How much of the game each mode accounted for."""

    if combined.empty:
        return {"pitches": 0}

    total = len(combined)
    counts = combined["source"].value_counts()
    live = int(counts.get("live", 0))
    late = int(counts.get("late", 0))
    replay_only = int(counts.get("replay", 0))

    live_rows = combined[combined["source"] == "live"]
    return {
        "pitches": total,
        "live": live,
        "late": late,
        "replay_only": replay_only,
        "live_share": live / total,
        # Accuracy of what a viewer saw before the pitch.
        "live_accuracy": (
            float(live_rows["live_correct"].mean()) if live else None
        ),
        # Accuracy over every pitch, which only the replay can measure.
        "replay_accuracy": float(combined["replay_correct"].mean()),
        "baseline_accuracy": (
            float(combined["baseline_correct"].mean())
            if "baseline_correct" in combined
            else None
        ),
    }
