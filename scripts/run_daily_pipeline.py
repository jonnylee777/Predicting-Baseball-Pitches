#!/usr/bin/env python3

"""Command-line entry point for the daily probable-starter pipeline."""

from __future__ import annotations
from pitch_prediction.model import PitchModelTrainer

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pitch_prediction.cleaning import PitchDataCleaner  # noqa: E402
from pitch_prediction.clients import BaseballSavantClient, MlbStatsClient  # noqa: E402
from pitch_prediction.feature_engineering import PitchFeatureEngineer  # noqa: E402
from pitch_prediction.pipeline import DailyStarterPipeline, PipelineRunError  # noqa: E402
from pitch_prediction.schema import StatcastSchema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find a day's MLB probable starters and incrementally download each "
            "pitcher's career pitch-detail CSV from Baseball Savant, then clean "
            "and feature-engineer it into the canonical KG2/KG3/KG4 formats."
        )
    )
    parser.add_argument("--date", type=date.fromisoformat, help="Game date (YYYY-MM-DD)")
    parser.add_argument(
        "--timezone",
        default="America/Chicago",
        help="Timezone used to determine today's date (default: America/Chicago)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "Data" / "daily_pipeline",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "statcast_columns.txt",
    )
    parser.add_argument(
        "--cleaned-schema",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "cleaned_columns.txt",
    )
    parser.add_argument("--refresh-days", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve probable starters and write the dated manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        zone = ZoneInfo(args.timezone)
    except Exception as exc:
        print(f"Invalid timezone {args.timezone!r}: {exc}", file=sys.stderr)
        return 2
    game_date = args.date or datetime.now(zone).date()

    raw_schema = StatcastSchema.from_file(args.schema)
    cleaned_schema = StatcastSchema.from_file(args.cleaned_schema)
    pipeline = DailyStarterPipeline(
        output_root=args.output_root,
        schema=raw_schema,
        cleaner=PitchDataCleaner(raw_schema, cleaned_schema),
        feature_engineer=PitchFeatureEngineer(cleaned_schema),
        model_trainer=PitchModelTrainer(),
        mlb_client=MlbStatsClient(timeout_seconds=min(args.timeout, 30)),
        savant_client=BaseballSavantClient(timeout_seconds=args.timeout),
        refresh_days=args.refresh_days,
    )
    try:
        manifest = pipeline.run(game_date=game_date, dry_run=args.dry_run)
    except PipelineRunError as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    print(f"Pipeline complete: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
