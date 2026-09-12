"""Delete frozen pre-game models that are no longer needed.

Each model is tens of megabytes and the pipeline writes one per starter per
day, so the models directory grows without bound. A model is only needed until
the postgame replay for its date has run: after that, the pitch-by-pitch
results live in ``predictions/postgame/`` and the summary in
``performance_history.csv``, neither of which needs the model again.

    python -m scripts.prune_models --keep-days 3 --dry-run
    python -m scripts.prune_models --keep-days 3

By default a date is only pruned once it has evaluated results, so an
unevaluated date is never destroyed to save space. Pass ``--force`` to prune
regardless.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("Data/daily_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-days",
        type=int,
        default=3,
        help="Keep models for dates within this many days of today (default 3)",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be deleted"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Prune even dates that have no evaluated results yet",
    )
    return parser.parse_args()


def evaluated_dates(data_root: Path) -> set[date]:
    path = data_root / "performance_history.csv"
    if not path.exists():
        return set()
    history = pd.read_csv(path, usecols=["game_date"])
    stamps = pd.to_datetime(history["game_date"], errors="coerce").dropna()
    return {stamp.date() for stamp in stamps}


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    args = parse_args()
    model_root = args.data_root / "models"
    if not model_root.exists():
        raise SystemExit(f"No models directory at {model_root}")

    cutoff = date.today() - timedelta(days=args.keep_days)
    evaluated = evaluated_dates(args.data_root)

    total_size = directory_size(model_root)
    print(f"models directory: {total_size / 1e9:.2f} GB")
    print(f"keeping dates after {cutoff.isoformat()}\n")

    freed = kept = 0
    for child in sorted(model_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            day = date.fromisoformat(child.name)
        except ValueError:
            continue

        size = directory_size(child)
        if day > cutoff:
            kept += size
            print(f"  keep  {child.name}  {size / 1e6:8.0f} MB  (within window)")
            continue
        if day not in evaluated and not args.force:
            kept += size
            print(
                f"  keep  {child.name}  {size / 1e6:8.0f} MB  "
                f"(no evaluated results yet; --force to override)"
            )
            continue

        freed += size
        print(f"  prune {child.name}  {size / 1e6:8.0f} MB")
        if not args.dry_run:
            shutil.rmtree(child)

    print(f"\n{'would free' if args.dry_run else 'freed'} {freed / 1e9:.2f} GB")
    print(f"retained {kept / 1e9:.2f} GB")
    if args.dry_run and freed:
        print("\nRe-run without --dry-run to delete.")


if __name__ == "__main__":
    main()
