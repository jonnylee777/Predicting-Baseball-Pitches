"""Predict pitches for a game in progress, or replay a completed one.

Live mode polls the GUMBO feed and predicts each pitch before it is thrown:

    python -m scripts.run_live_prediction --game-pk 824802 --pitcher-id 680694

Replay mode drives the same engine through a completed game's archived
snapshots, which are the exact sequence of states a live client would have
observed. Because each prediction is timestamped with the snapshot's own
timecode, the reported lead over each pitch is the real lead the engine would
have had. This is how the live loop is tested without waiting for a game:

    python -m scripts.run_live_prediction --game-pk 824802 --pitcher-id 680694 --replay
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import date
from pathlib import Path

from pitch_prediction.live.batter_zones import build_batter_zone_reference
from pitch_prediction.live.engine import (
    LivePredictionEngine,
    build_context,
    load_pregame_model,
    parse_timecode,
)
from pitch_prediction.live.gumbo import GumboClient

DATA_ROOT = Path("Data/daily_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-pk", type=int, required=True)
    parser.add_argument("--pitcher-id", type=int, required=True)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay a completed game through its archived snapshots",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Live polling interval; the feed advances every few seconds",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Stop replay after this many snapshots",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL path for resolved predictions "
        "(default Data/daily_pipeline/predictions/live/<game>_<pitcher>.jsonl)",
    )
    parser.add_argument(
        "--with-win-probability",
        action="store_true",
        help="Approximate bat_win_exp from the per-at-bat endpoint",
    )
    return parser.parse_args()


def build_engine(args: argparse.Namespace, client: GumboClient, game_date: date):
    model = load_pregame_model(args.data_root, game_date.isoformat(), args.pitcher_id)
    reference_frame = build_batter_zone_reference(
        args.data_root / "features" / "kg4" / "pitchers", through_date=game_date
    )
    reference = {
        int(row.batter): (float(row.sz_top), float(row.sz_bot))
        for row in reference_frame.itertuples()
    }
    context = build_context(
        args.data_root, args.pitcher_id, game_date, batter_zone_reference=reference
    )
    return model, context


def summarize(engine: LivePredictionEngine, label: str) -> None:
    stats = engine.stats
    print()
    print("=" * 70)
    print(f"{label}: {engine.pitcher_name}")
    print("=" * 70)
    print(f"  predictions issued      {stats.predictions_made}")
    print(f"  superseded by new state {stats.superseded}")
    print(f"  pitches scored          {stats.resolved}")

    if not stats.resolved:
        print("  no pitches were scored")
        return

    print(f"  correct                 {stats.correct}")
    if stats.accuracy is not None:
        print(f"  accuracy (all scored)   {stats.accuracy:.1%}")

    honest = stats.honest_accuracy(engine.history)
    print()
    print("  TIMING AUDIT")
    print(f"    predicted before pitch  {stats.before_pitch}/{stats.resolved}"
          f" ({stats.before_pitch / stats.resolved:.0%})")
    print(f"    predicted late          {stats.late}/{stats.resolved}")
    if honest is not None:
        print(f"    accuracy, in-time only  {honest:.1%}")
    if stats.leads:
        leads = sorted(stats.leads)
        print(f"    lead over pitch (s)     median {statistics.median(leads):+.1f}"
              f"  p10 {leads[int(0.1 * len(leads))]:+.1f}"
              f"  p90 {leads[int(0.9 * len(leads))]:+.1f}")


def main() -> None:
    args = parse_args()
    client = GumboClient()

    probe = client.feed_live(args.game_pk)
    if probe.official_date is None:
        raise SystemExit(f"Game {args.game_pk} has no official date")
    game_date = date.fromisoformat(probe.official_date)

    pitcher_name = probe.player(args.pitcher_id).get("fullName")
    if not pitcher_name:
        raise SystemExit(
            f"Pitcher {args.pitcher_id} does not appear in game {args.game_pk}"
        )

    model, context = build_engine(args, client, game_date)
    win_probability = (
        client.win_probability(args.game_pk) if args.with_win_probability else None
    )

    log_path = args.log or (
        args.data_root
        / "predictions"
        / "live"
        / f"{args.game_pk}_{args.pitcher_id}.jsonl"
    )

    if args.replay:
        timecodes = client.timestamps(args.game_pk)
        if args.max_snapshots:
            timecodes = timecodes[: args.max_snapshots]
        print(
            f"Replaying game {args.game_pk} for {pitcher_name} "
            f"through {len(timecodes)} archived snapshots..."
        )
        # Predictions are stamped with the snapshot's own timecode so that the
        # lead over each pitch reflects the game's real timing.
        engine_state = {"timecode": timecodes[0]}
        engine = LivePredictionEngine(
            model=model,
            context=context,
            pitcher_name=pitcher_name,
            win_probability=win_probability,
            log_path=log_path,
            clock=lambda: parse_timecode(engine_state["timecode"]),
        )
        for position, timecode in enumerate(timecodes, start=1):
            engine_state["timecode"] = timecode
            snapshot = client.feed_live(args.game_pk, timecode=timecode)
            for prediction in engine.observe(snapshot):
                mark = "OK " if prediction.correct else "   "
                lead = prediction.prediction_lead_seconds
                lead_text = f"{lead:+5.1f}s" if lead is not None else "    ?"
                print(
                    f"  {mark} inn {prediction.inning} "
                    f"{prediction.balls}-{prediction.strikes} "
                    f"{prediction.batter_name:<22} "
                    f"pred {prediction.predicted_pitch_type:<3} "
                    f"actual {str(prediction.actual_pitch_type):<3} "
                    f"conf {prediction.confidence:.2f} lead {lead_text}"
                )
            if position % 50 == 0:
                print(f"  ... {position}/{len(timecodes)} snapshots")
        summarize(engine, "REPLAY COMPLETE")
    else:
        engine = LivePredictionEngine(
            model=model,
            context=context,
            pitcher_name=pitcher_name,
            win_probability=win_probability,
            log_path=log_path,
        )
        print(f"Following game {args.game_pk} for {pitcher_name}. Ctrl-C to stop.")
        try:
            while True:
                snapshot = client.feed_live(args.game_pk)
                for prediction in engine.observe(snapshot):
                    mark = "OK " if prediction.correct else "   "
                    print(
                        f"  {mark} inn {prediction.inning} "
                        f"{prediction.balls}-{prediction.strikes} "
                        f"{prediction.batter_name:<22} "
                        f"pred {prediction.predicted_pitch_type:<3} "
                        f"actual {str(prediction.actual_pitch_type):<3} "
                        f"({prediction.timing})"
                    )
                if engine.pending is not None:
                    pending = engine.pending
                    print(
                        f"  -> next pitch to {pending.batter_name} "
                        f"({pending.balls}-{pending.strikes}): "
                        f"{pending.predicted_pitch_type} "
                        f"[{pending.confidence:.0%}]"
                    )
                if snapshot.is_final:
                    break
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("\nStopped.")
        summarize(engine, "LIVE SESSION")

    print(f"\n  prediction log: {log_path}")


if __name__ == "__main__":
    main()
