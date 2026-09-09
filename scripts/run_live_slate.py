"""Follow every game on a slate, comparing the production and live models.

One process handles the whole slate. Each live game's feed is fetched once per
cycle and shared by all four engines belonging to it -- both starting pitchers,
each scored by both models -- so a slate of fifteen games costs fifteen
requests per cycle rather than sixty. Fetches run on a small thread pool to
keep the cycle short without bursting.

For each model it records how often a prediction provably preceded its pitch
and how often it was right, on identical pitches.

    python -m scripts.run_live_slate --date 2026-09-09 --until 23:30

Requires scripts.run_daily_pipeline and scripts.train_live_models for the date.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
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
VARIANTS = {"production": (), "live": PREDICT_AHEAD_EXCLUSIONS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--until", default=None, help="Stop at this UTC time today (HH:MM)"
    )
    parser.add_argument("--cycle-seconds", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--report-every", type=float, default=120.0)
    parser.add_argument(
        "--output", type=Path, default=Path("Data/ablation/live_slate.json")
    )
    return parser.parse_args()


def model_path(root: Path, game_date: date, variant: str, pitcher_id: int) -> Path:
    folder = "pitchers" if variant == "production" else "pitchers_live"
    return root / "models" / game_date.isoformat() / folder / f"{pitcher_id}.joblib"


class SlateRunner:
    def __init__(self, args: argparse.Namespace, game_date: date) -> None:
        self.args = args
        self.game_date = game_date
        self.client = GumboClient()
        self.zone_reference = load_or_build(
            args.data_root / "features" / "kg4" / "pitchers",
            through_date=game_date,
            cache_dir=args.data_root / "reference",
        )
        # game_pk -> pitcher_id -> variant -> engine
        self.engines: dict[int, dict[int, dict[str, LivePredictionEngine]]] = {}
        self.names: dict[int, str] = {}
        self.done: set[int] = set()

    def discover(self) -> list[dict]:
        games = self.client.schedule(self.game_date.isoformat())
        eligible = []
        for game in games:
            starters = {}
            for side in ("away", "home"):
                pitcher = game["teams"][side].get("probablePitcher")
                if not pitcher:
                    continue
                pid = int(pitcher["id"])
                if all(
                    model_path(self.args.data_root, self.game_date, v, pid).exists()
                    for v in VARIANTS
                ):
                    starters[pid] = pitcher["fullName"]
            if starters:
                eligible.append({"game_pk": int(game["gamePk"]), "starters": starters})
        return eligible

    def ensure_engines(self, game_pk: int, starters: dict[int, str]) -> None:
        if game_pk in self.engines:
            return
        built: dict[int, dict[str, LivePredictionEngine]] = {}
        for pid, name in starters.items():
            self.names[pid] = name
            context = build_context(
                self.args.data_root,
                pid,
                self.game_date,
                batter_zone_reference=self.zone_reference,
            )
            per_variant = {}
            for variant, exclusions in VARIANTS.items():
                per_variant[variant] = LivePredictionEngine(
                    model=joblib.load(
                        model_path(self.args.data_root, self.game_date, variant, pid)
                    ),
                    context=context,
                    pitcher_name=name,
                    trainer=PitchModelTrainer(exclude_features=exclusions),
                    log_path=(
                        self.args.data_root
                        / "predictions"
                        / "live"
                        / f"slate_{game_pk}_{pid}_{variant}.jsonl"
                    ),
                )
            built[pid] = per_variant
        self.engines[game_pk] = built
        print(
            f"  [engines] game {game_pk}: "
            + ", ".join(starters.values()),
            flush=True,
        )

    def cycle(self, live: list[dict]) -> None:
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = {
                pool.submit(self.client.feed_live, entry["game_pk"]): entry
                for entry in live
            }
            for future, entry in futures.items():
                try:
                    snapshot = future.result()
                except Exception:
                    continue
                game_pk = entry["game_pk"]
                self.ensure_engines(game_pk, entry["starters"])
                for pid, per_variant in self.engines[game_pk].items():
                    for variant, engine in per_variant.items():
                        try:
                            for prediction in engine.observe(snapshot):
                                self.log(variant, prediction)
                        except Exception as error:  # noqa: BLE001
                            print(
                                f"  [error] {variant} {pid}: {error}", flush=True
                            )
                if snapshot.is_final:
                    self.done.add(game_pk)

    @staticmethod
    def log(variant: str, prediction: PitchPrediction) -> None:
        mark = "OK " if prediction.correct else "   "
        lead = prediction.prediction_lead_seconds
        print(
            f"  [{variant:<10}] {mark}{prediction.pitcher_name[:18]:<18} "
            f"inn {prediction.inning} {prediction.balls}-{prediction.strikes} "
            f"pred {prediction.predicted_pitch_type:<3} "
            f"act {str(prediction.actual_pitch_type):<3} "
            f"lead {lead:+6.1f}s "
            f"{'ahead' if prediction.predicted_ahead else ''}",
            flush=True,
        )

    def records(self, variant: str) -> list[PitchPrediction]:
        return [
            prediction
            for game in self.engines.values()
            for per_variant in game.values()
            for prediction in per_variant[variant].history
        ]

    def summary(self) -> list[dict]:
        rows = []
        for variant in VARIANTS:
            records = self.records(variant)
            if not records:
                rows.append({"model": variant, "pitches": 0})
                continue
            in_time = [r for r in records if r.timing == "before_pitch"]
            leads = [
                r.prediction_lead_seconds
                for r in records
                if r.prediction_lead_seconds is not None
            ]
            rows.append(
                {
                    "model": variant,
                    "pitches": len(records),
                    "pitchers": len({r.pitcher_id for r in records}),
                    "before_pitch": len(in_time),
                    "before_pitch_rate": len(in_time) / len(records),
                    "computed_ahead": sum(1 for r in records if r.predicted_ahead),
                    "accuracy_all": sum(1 for r in records if r.correct) / len(records),
                    "accuracy_in_time": (
                        sum(1 for r in in_time if r.correct) / len(in_time)
                        if in_time
                        else None
                    ),
                    "median_lead": statistics.median(leads) if leads else None,
                }
            )
        return rows

    def report(self) -> None:
        rows = self.summary()
        print("\n" + "=" * 82, flush=True)
        print(
            f"SLATE SO FAR  {datetime.now(timezone.utc):%H:%M}Z   "
            f"games with engines: {len(self.engines)}  final: {len(self.done)}"
        )
        print("=" * 82)
        print(
            f"{'model':<12}{'pitches':>8}{'pitchers':>9}{'before pitch':>15}"
            f"{'ahead':>7}{'med lead':>10}{'acc':>8}{'acc in-time':>13}"
        )
        for row in rows:
            if not row["pitches"]:
                print(f"{row['model']:<12}{0:>8}   nothing scored yet")
                continue
            in_time = (
                f"{row['accuracy_in_time']:.1%}"
                if row["accuracy_in_time"] is not None
                else "--"
            )
            print(
                f"{row['model']:<12}{row['pitches']:>8}{row['pitchers']:>9}"
                + f"{row['before_pitch']}/{row['pitches']} ({row['before_pitch_rate']:.0%})".rjust(15)
                + f"{row['computed_ahead']:>7}{row['median_lead']:>+9.1f}s"
                f"{row['accuracy_all']:>8.1%}{in_time:>13}"
            )
        print(flush=True)
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def run(self, deadline: datetime | None) -> None:
        last_report = time.time()
        while True:
            if deadline and datetime.now(timezone.utc) > deadline:
                print("reached --until", flush=True)
                break
            try:
                slate = self.discover()
            except Exception as error:  # noqa: BLE001
                print(f"  [schedule error] {error}", flush=True)
                time.sleep(30)
                continue

            live = []
            for entry in slate:
                if entry["game_pk"] in self.done:
                    continue
                live.append(entry)
            if not live:
                print("no games left to follow", flush=True)
                break

            # Only fetch games that are actually underway.
            states = {}
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                futures = {
                    pool.submit(self.client.feed_live, e["game_pk"]): e for e in live
                }
                for future, entry in futures.items():
                    try:
                        snapshot = future.result()
                    except Exception:
                        continue
                    states[entry["game_pk"]] = snapshot.abstract_state
                    if snapshot.is_final:
                        self.done.add(entry["game_pk"])

            underway = [e for e in live if states.get(e["game_pk"]) == "Live"]
            if underway:
                self.cycle(underway)
            if time.time() - last_report > self.args.report_every:
                self.report()
                last_report = time.time()
            time.sleep(self.args.cycle_seconds)

        self.report()


def main() -> None:
    args = parse_args()
    game_date = date.fromisoformat(args.date)
    deadline = None
    if args.until:
        hour, minute = (int(part) for part in args.until.split(":"))
        deadline = datetime.now(timezone.utc).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if deadline < datetime.now(timezone.utc):
            deadline += timedelta(days=1)

    runner = SlateRunner(args, game_date)
    slate = runner.discover()
    print(
        f"Slate for {game_date}: {len(slate)} games with both models trained, "
        f"{sum(len(entry['starters']) for entry in slate)} starters."
    )
    if deadline:
        print(f"Running until {deadline:%H:%M}Z.")
    runner.run(deadline)


if __name__ == "__main__":
    main()
