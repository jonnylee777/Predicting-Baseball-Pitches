"""Audit the live feature path against Savant ground truth.

Replays completed pitcher-games through the GUMBO-based live feature builder
and reports, per column, how closely the reconstructed features match the
Savant features the models were trained on, plus the effect on prediction
accuracy when the same frozen pre-game model scores both.

    python -m scripts.verify_live_features --date 2026-08-20
    python -m scripts.verify_live_features --date 2026-08-20 --pitcher-id 680694
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from pitch_prediction.live.gumbo import GumboClient
from pitch_prediction.live.verification import GameVerification, LiveFeatureVerifier


DATA_ROOT = Path("Data/daily_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", required=True, help="Completed game date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--pitcher-id", type=int, default=None, help="Limit to one pitcher"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum pitcher-games to verify"
    )
    parser.add_argument(
        "--data-root", type=Path, default=DATA_ROOT, help="Pipeline data root"
    )
    parser.add_argument(
        "--with-win-probability",
        action="store_true",
        help="Approximate bat_win_exp from the per-at-bat win probability "
        "endpoint (measured slightly worse than leaving it missing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the full report",
    )
    return parser.parse_args()


def discover_targets(
    data_root: Path, game_date: date, pitcher_id: int | None
) -> list[tuple[int, int]]:
    """Find (game_pk, pitcher_id) pairs that have a frozen pre-game model."""

    model_dir = data_root / "models" / game_date.isoformat() / "pitchers"
    if not model_dir.exists():
        raise SystemExit(f"No frozen models for {game_date}: {model_dir} is missing")

    targets: list[tuple[int, int]] = []
    for model_path in sorted(model_dir.glob("*.joblib")):
        pitcher = int(model_path.stem)
        if pitcher_id is not None and pitcher != pitcher_id:
            continue
        history_path = (
            data_root / "features" / "kg4" / "pitchers" / f"{pitcher}.csv"
        )
        if not history_path.exists():
            continue
        history = pd.read_csv(
            history_path, low_memory=False, usecols=["game_date", "game_pk"]
        )
        history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
        same_day = history[history["game_date"].dt.date == game_date]
        for game_pk in sorted(same_day["game_pk"].dropna().unique()):
            targets.append((int(game_pk), pitcher))
    return targets


def print_report(results: list[GameVerification]) -> None:
    if not results:
        print("No pitcher-games verified.")
        return

    total_pitches = sum(result.pitch_count for result in results)

    print()
    print("=" * 78)
    print("PER-GAME PREDICTION COMPARISON")
    print("=" * 78)
    print(
        f"{'pitcher':>8} {'game':>8} {'pitches':>8} "
        f"{'savant':>8} {'live':>8} {'delta':>8} {'agree':>8}"
    )
    for result in results:
        print(
            f"{result.pitcher_id:>8} {result.game_pk:>8} {result.pitch_count:>8} "
            f"{result.savant_accuracy:>7.1%} {result.live_accuracy:>7.1%} "
            f"{result.accuracy_delta:>+7.1%} {result.prediction_agreement:>7.1%}"
        )

    # Pitch-weighted, matching how the project reports headline accuracy.
    def weighted(attribute: str) -> float:
        return (
            sum(getattr(r, attribute) * r.pitch_count for r in results) / total_pitches
        )

    savant_accuracy = weighted("savant_accuracy")
    live_accuracy = weighted("live_accuracy")
    baseline_accuracy = weighted("baseline_accuracy")
    agreement = weighted("prediction_agreement")

    print("-" * 78)
    print(
        f"{'TOTAL':>8} {'':>8} {total_pitches:>8} "
        f"{savant_accuracy:>7.1%} {live_accuracy:>7.1%} "
        f"{live_accuracy - savant_accuracy:>+7.1%} {agreement:>7.1%}"
    )

    print()
    print("=" * 78)
    print("HEADLINE METRIC: RELATIVE IMPROVEMENT OVER BASELINE")
    print("=" * 78)
    savant_relative = (savant_accuracy - baseline_accuracy) / baseline_accuracy
    live_relative = (live_accuracy - baseline_accuracy) / baseline_accuracy
    print(f"  baseline accuracy          {baseline_accuracy:>8.1%}")
    print(f"  postgame (Savant features) {savant_relative:>+8.1%}")
    print(f"  live (GUMBO features)      {live_relative:>+8.1%}")
    print(f"  cost of going live         {live_relative - savant_relative:>+8.1%}")

    print()
    print("=" * 78)
    print("COLUMN FIDELITY (pitches pooled across games)")
    print("=" * 78)
    pooled: dict[str, dict[str, int]] = defaultdict(
        lambda: {"compared": 0, "matched": 0, "live_missing": 0, "savant_missing": 0}
    )
    for result in results:
        for column in result.columns:
            entry = pooled[column.column]
            entry["compared"] += column.compared
            entry["matched"] += column.matched
            entry["live_missing"] += column.live_missing_only
            entry["savant_missing"] += column.savant_missing_only

    rows = []
    for name, entry in pooled.items():
        rate = entry["matched"] / entry["compared"] if entry["compared"] else float("nan")
        rows.append((rate, name, entry))
    rows.sort(key=lambda item: (item[0], item[1]))

    print(f"{'column':<36} {'match':>8} {'compared':>9} {'live NA':>9} {'sav NA':>8}")
    imperfect = [row for row in rows if not row[0] >= 0.9999]
    for rate, name, entry in imperfect:
        print(
            f"{name:<36} {rate:>7.1%} {entry['compared']:>9} "
            f"{entry['live_missing']:>9} {entry['savant_missing']:>8}"
        )
    print(f"\n  {len(rows) - len(imperfect)} of {len(rows)} columns match exactly.")

    zone_sources: dict[str, int] = defaultdict(int)
    for result in results:
        for source, count in result.zone_sources.items():
            zone_sources[source] += count
    print(f"\n  strike zone source: {dict(zone_sources)}")

    warnings = {r.translation_warnings for r in results} - {"none"}
    print(f"  translation warnings: {warnings or 'none'}")


def main() -> None:
    args = parse_args()
    game_date = date.fromisoformat(args.date)

    targets = discover_targets(args.data_root, game_date, args.pitcher_id)
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit(f"No verifiable pitcher-games found for {game_date}")

    print(f"Verifying {len(targets)} pitcher-game(s) for {game_date}...")
    verifier = LiveFeatureVerifier(
        args.data_root,
        client=GumboClient(),
        use_win_probability=args.with_win_probability,
    )

    results: list[GameVerification] = []
    for game_pk, pitcher_id in targets:
        try:
            result = verifier.verify(
                game_pk=game_pk, pitcher_id=pitcher_id, game_date=game_date
            )
        except Exception as error:  # noqa: BLE001 - report and continue the audit
            print(f"  [skip] game {game_pk} pitcher {pitcher_id}: {error}")
            continue
        results.append(result)
        print(
            f"  [ok]   game {game_pk} pitcher {pitcher_id}: "
            f"{result.pitch_count} pitches, "
            f"live {result.live_accuracy:.1%} vs savant {result.savant_accuracy:.1%}"
        )

    print_report(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                **{
                    key: value
                    for key, value in vars(result).items()
                    if key != "columns"
                },
                "columns": [vars(column) for column in result.columns],
            }
            for result in results
        ]
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
