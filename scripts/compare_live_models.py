"""Run the production and live models side by side on the same live game.

Both engines see the same feed snapshots, so any difference is the model and
its prediction strategy, not the game:

* **production** -- the full 65-feature model. It cannot be run ahead, because
  it needs the previous pitch's measurements, so it predicts only once the feed
  publishes them.
* **live** -- the 54-feature model, which enumerates the states the next pitch
  could arrive in and scores them before the current pitch is thrown.

Reports, for the pitches both engines scored, how often each predicted before
the pitch actually happened and how often it was right.

    python -m scripts.compare_live_models --game-pk 824956 --pitcher-id 12345

Requires both `scripts.run_daily_pipeline` and `scripts.train_live_models` to
have run for the game's date, before first pitch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

import joblib

from pitch_prediction.live.batter_zones import load_or_build
from pitch_prediction.live.engine import (
    LivePredictionEngine,
    PitchPrediction,
    build_context,
)
from pitch_prediction.live.feature_sets import PREDICT_AHEAD_EXCLUSIONS
from pitch_prediction.live.gumbo import GumboClient
from pitch_prediction.model import PitchModelTrainer

DATA_ROOT = Path("Data/daily_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-pk", type=int, required=True)
    parser.add_argument("--pitcher-id", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--minutes", type=float, default=None, help="Stop after this long"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build(args, game_date: date, pitcher_name: str) -> dict[str, LivePredictionEngine]:
    root = args.data_root / "models" / game_date.isoformat()
    paths = {
        "production": root / "pitchers" / f"{args.pitcher_id}.joblib",
        "live": root / "pitchers_live" / f"{args.pitcher_id}.joblib",
    }
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(
                f"Missing the {name} model at {path}. Run "
                f"scripts.run_daily_pipeline and scripts.train_live_models for "
                f"{game_date} first."
            )

    reference = load_or_build(
        args.data_root / "features" / "kg4" / "pitchers",
        through_date=game_date,
        cache_dir=args.data_root / "reference",
    )
    context = build_context(
        args.data_root, args.pitcher_id, game_date, batter_zone_reference=reference
    )

    engines = {}
    for name, path in paths.items():
        exclusions = PREDICT_AHEAD_EXCLUSIONS if name == "live" else ()
        engines[name] = LivePredictionEngine(
            model=joblib.load(path),
            context=context,
            pitcher_name=pitcher_name,
            trainer=PitchModelTrainer(exclude_features=exclusions),
            log_path=(
                args.data_root
                / "predictions"
                / "live"
                / f"{args.game_pk}_{args.pitcher_id}_{name}.jsonl"
            ),
        )
    return engines


def summarize(name: str, records: list[PitchPrediction]) -> dict:
    if not records:
        return {"model": name, "pitches": 0}
    leads = [
        r.prediction_lead_seconds
        for r in records
        if r.prediction_lead_seconds is not None
    ]
    in_time = [r for r in records if r.timing == "before_pitch"]
    correct_in_time = [r for r in in_time if r.correct]
    return {
        "model": name,
        "pitches": len(records),
        "before_pitch": len(in_time),
        "before_pitch_rate": len(in_time) / len(records),
        "computed_ahead": sum(1 for r in records if r.predicted_ahead),
        "accuracy_all": sum(1 for r in records if r.correct) / len(records),
        "accuracy_in_time": (
            len(correct_in_time) / len(in_time) if in_time else None
        ),
        "median_lead": statistics.median(leads) if leads else None,
    }


def report(rows: list[dict]) -> None:
    print()
    print("=" * 78)
    print("PRODUCTION vs LIVE MODEL, same pitches")
    print("=" * 78)
    print(
        f"{'model':<12}{'pitches':>8}{'before pitch':>14}{'ahead':>7}"
        f"{'median lead':>13}{'acc (all)':>11}{'acc (in time)':>14}"
    )
    for row in rows:
        if not row["pitches"]:
            print(f"{row['model']:<12}{'0':>8}   no pitches scored")
            continue
        in_time = (
            f"{row['accuracy_in_time']:.1%}"
            if row["accuracy_in_time"] is not None
            else "--"
        )
        print(
            f"{row['model']:<12}{row['pitches']:>8}"
            f"{row['before_pitch']}/{row['pitches']} ({row['before_pitch_rate']:.0%})".rjust(14)
            + f"{row['computed_ahead']:>7}"
            f"{row['median_lead']:>+12.1f}s"
            f"{row['accuracy_all']:>11.1%}{in_time:>14}"
        )
    print()
    print("  before pitch  = prediction provably preceded the pitch")
    print("  ahead         = computed before the *previous* pitch was thrown")
    print("  acc (in time) = accuracy over only those predictions")


def main() -> None:
    args = parse_args()
    client = GumboClient()
    probe = client.feed_live(args.game_pk)
    game_date = date.fromisoformat(probe.official_date)
    pitcher_name = probe.player(args.pitcher_id).get("fullName")
    if not pitcher_name:
        raise SystemExit(
            f"Pitcher {args.pitcher_id} is not in game {args.game_pk}"
        )

    engines = build(args, game_date, pitcher_name)
    print(f"Comparing both models on {pitcher_name}, game {args.game_pk}.")
    print("Ctrl-C to stop and print the comparison.\n")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    try:
        while True:
            snapshot = client.feed_live(args.game_pk)  # one fetch, both engines
            for name, engine in engines.items():
                for prediction in engine.observe(snapshot):
                    mark = "OK " if prediction.correct else "   "
                    print(
                        f"  [{name:<10}] {mark} inn {prediction.inning} "
                        f"{prediction.balls}-{prediction.strikes} "
                        f"{prediction.batter_name:<20} "
                        f"pred {prediction.predicted_pitch_type:<3} "
                        f"actual {str(prediction.actual_pitch_type):<3} "
                        f"lead {prediction.prediction_lead_seconds:+6.1f}s "
                        f"{'ahead' if prediction.predicted_ahead else ''}"
                    )
            if snapshot.is_final or (deadline and time.time() > deadline):
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")

    rows = [summarize(name, engine.history) for name, engine in engines.items()]
    report(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
