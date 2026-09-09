"""Train pre-game models that can be run a pitch ahead.

The production models use the previous pitch's Statcast measurements, which MLB
publishes about 19 seconds after the pitch while pitches arrive about 20 seconds
apart. A predictor that waits for them has roughly a second of headroom.

These models withhold those columns, which leaves only enumerable unknowns, so
the live engine can score every state the next pitch might arrive in before the
current pitch is thrown. See ``pitch_prediction.live.feature_sets``.

Run after ``scripts.run_daily_pipeline`` for the same date, which downloads the
histories these models train on:

    python -m scripts.train_live_models --date 2026-09-08

Models are written beside the production ones, under
``models/<date>/pitchers_live/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from pitch_prediction.live.feature_sets import PREDICT_AHEAD_EXCLUSIONS
from pitch_prediction.model import PitchModelTrainer

DATA_ROOT = Path("Data/daily_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain pitchers that already have a live model",
    )
    return parser.parse_args()


def starters_for(data_root: Path, game_date: date) -> list[tuple[int, str]]:
    """Read the day's starters from the pipeline manifest."""

    manifest_path = data_root / "runs" / game_date.isoformat() / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No pipeline manifest at {manifest_path}. Run "
            f"`python -m scripts.run_daily_pipeline --date {game_date}` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen: dict[int, str] = {}
    for starter in manifest.get("starters") or []:
        seen.setdefault(int(starter["pitcher_id"]), starter.get("pitcher_name", "?"))
    return sorted(seen.items())


def main() -> None:
    args = parse_args()
    game_date = date.fromisoformat(args.date)
    model_dir = (
        args.data_root / "models" / game_date.isoformat() / "pitchers_live"
    )

    starters = starters_for(args.data_root, game_date)
    print(
        f"Training live-compatible models for {len(starters)} starters on "
        f"{game_date}, withholding {len(PREDICT_AHEAD_EXCLUSIONS)} late columns."
    )

    trainer = PitchModelTrainer(exclude_features=PREDICT_AHEAD_EXCLUSIONS)
    trained = skipped = failed = 0

    for pitcher_id, pitcher_name in starters:
        target = model_dir / f"{pitcher_id}.joblib"
        if target.exists() and not args.force:
            skipped += 1
            continue

        history_path = (
            args.data_root / "features" / "kg4" / "pitchers" / f"{pitcher_id}.csv"
        )
        if not history_path.exists():
            print(f"  [skip] {pitcher_name} ({pitcher_id}): no KG4 history")
            failed += 1
            continue

        history = pd.read_csv(history_path, low_memory=False)
        history["game_date"] = pd.to_datetime(history["game_date"], errors="raise")
        # Only data before the game date, so the model stays a pre-game model.
        pregame = history[history["game_date"].dt.date < game_date]
        if pregame.empty:
            print(f"  [skip] {pitcher_name} ({pitcher_id}): no pre-game history")
            failed += 1
            continue

        try:
            result = trainer.train(
                pregame,
                pitcher_id=pitcher_id,
                pitcher_name=pitcher_name,
                reference_season=game_date.year,
                model_dir=model_dir,
            )
        except Exception as error:  # noqa: BLE001 - keep training the rest
            print(f"  [fail] {pitcher_name} ({pitcher_id}): {error}")
            failed += 1
            continue

        trained += 1
        print(
            f"  [ok]   {pitcher_name} ({pitcher_id}): "
            f"{result.career_pitches:,} pitches, "
            f"test {result.test_accuracy:.1%} vs baseline "
            f"{result.baseline_accuracy:.1%}"
        )

    print(f"\ntrained {trained}, skipped {skipped}, failed {failed}")
    print(f"models: {model_dir}")


if __name__ == "__main__":
    main()
