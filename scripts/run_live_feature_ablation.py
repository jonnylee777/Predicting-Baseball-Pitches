"""Does excluding live-unavailable columns from training improve live accuracy?

The production models are trained on Savant's full KG4 frame, including three
columns the live feed cannot supply. ``bat_win_exp`` is the concerning one:
Savant never omits it, so a model trained on Savant has no experience of it
being absent, yet every live row arrives with it missing and lands on the
imputer's sentinel value.

This script retrains each pitcher's pre-game model under several exclusion
sets and scores the same live features against each, to measure whether a
model that never learns to depend on those columns predicts live better.

    python -m scripts.run_live_feature_ablation --date 2026-08-20

Models are written to a scratch directory and do not touch the frozen models
used by the daily pipeline.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from pitch_prediction.live.batter_zones import build_batter_zone_reference
from pitch_prediction.live.context import PregameContext
from pitch_prediction.live.features import LiveFeatureBuilder
from pitch_prediction.live.gumbo import GumboClient
from pitch_prediction.live.verification import _align
from pitch_prediction.model import LIVE_UNAVAILABLE_COLUMNS, PitchModelTrainer


DATA_ROOT = Path("Data/daily_pipeline")

# Each variant is a set of feature columns withheld from training AND from
# prediction, so the model never sees them in either place.
VARIANTS: dict[str, tuple[str, ...]] = {
    "full": (),
    "no_bat_win_exp": ("bat_win_exp",),
    "no_alignments": ("if_fielding_alignment", "of_fielding_alignment"),
    "live_compatible": LIVE_UNAVAILABLE_COLUMNS,
}


@dataclass
class VariantResult:
    variant: str
    pitcher_id: int
    game_pk: int
    pitches: int
    savant_accuracy: float
    live_accuracy: float
    baseline_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, action="append", dest="dates")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def discover(data_root: Path, game_date: date) -> list[tuple[int, int]]:
    model_dir = data_root / "models" / game_date.isoformat() / "pitchers"
    targets: list[tuple[int, int]] = []
    for model_path in sorted(model_dir.glob("*.joblib")):
        pitcher_id = int(model_path.stem)
        history_path = (
            data_root / "features" / "kg4" / "pitchers" / f"{pitcher_id}.csv"
        )
        if not history_path.exists():
            continue
        history = pd.read_csv(
            history_path, low_memory=False, usecols=["game_date", "game_pk"]
        )
        history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
        for game_pk in sorted(
            history.loc[history["game_date"].dt.date == game_date, "game_pk"]
            .dropna()
            .unique()
        ):
            targets.append((int(game_pk), pitcher_id))
    return targets


def evaluate_pitcher_game(
    *,
    data_root: Path,
    client: GumboClient,
    game_pk: int,
    pitcher_id: int,
    game_date: date,
    zone_reference: dict[int, tuple[float, float]],
    scratch: Path,
) -> list[VariantResult]:
    history = pd.read_csv(
        data_root / "features" / "kg4" / "pitchers" / f"{pitcher_id}.csv",
        low_memory=False,
    )
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")

    savant = (
        history[history["game_pk"] == game_pk]
        .sort_values(["at_bat_number_of_game", "pitch_number_of_ab"])
        .reset_index(drop=True)
    )
    pregame = history[history["game_date"].dt.date < game_date].copy()
    if savant.empty or pregame.empty:
        return []

    # The live features are identical across variants; only which columns each
    # model consumes differs, so the feed is walked once.
    context = PregameContext.from_history(
        history,
        pitcher_id=pitcher_id,
        game_date=game_date,
        batter_zone_reference=zone_reference,
    )
    snapshot = client.feed_live(game_pk)
    built = LiveFeatureBuilder(context).build(snapshot, include_pending=False)
    live = (
        built.features.assign(_order=range(len(built.features)))
        .sort_values(["at_bat_number_of_game", "pitch_number_of_ab", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    aligned_live, aligned_savant = _align(live, savant)
    actual = aligned_savant["pitch_type"].astype(str).to_numpy()
    if not len(actual):
        return []

    results: list[VariantResult] = []
    for name, excluded in VARIANTS.items():
        trainer = PitchModelTrainer(exclude_features=excluded)
        model_dir = scratch / name / game_date.isoformat()
        trainer.train(
            pregame,
            pitcher_id=pitcher_id,
            pitcher_name=str(pitcher_id),
            reference_season=game_date.year,
            model_dir=model_dir,
        )
        model = joblib.load(model_dir / f"{pitcher_id}.joblib")

        def predict(frame: pd.DataFrame):
            features = trainer._features(frame)
            if hasattr(model, "feature_names_in_"):
                features = features.reindex(columns=list(model.feature_names_in_))
            return model.predict(features)

        baseline = DummyClassifier(
            strategy="stratified", random_state=trainer.random_state
        )
        pregame_features = trainer._features(pregame)
        baseline.fit(pregame_features, pregame["pitch_type"].astype(str))
        baseline_predictions = baseline.predict(
            trainer._features(aligned_savant).reindex(columns=pregame_features.columns)
        )

        results.append(
            VariantResult(
                variant=name,
                pitcher_id=pitcher_id,
                game_pk=game_pk,
                pitches=len(actual),
                savant_accuracy=float((predict(aligned_savant) == actual).mean()),
                live_accuracy=float((predict(aligned_live) == actual).mean()),
                baseline_accuracy=float((baseline_predictions == actual).mean()),
            )
        )
    return results


def report(results: list[VariantResult]) -> pd.DataFrame:
    frame = pd.DataFrame([vars(result) for result in results])
    if frame.empty:
        print("No results.")
        return frame

    rows = []
    for name in VARIANTS:
        subset = frame[frame["variant"] == name]
        if subset.empty:
            continue
        pitches = subset["pitches"].sum()

        def weighted(column: str) -> float:
            return float((subset[column] * subset["pitches"]).sum() / pitches)

        live = weighted("live_accuracy")
        savant = weighted("savant_accuracy")
        baseline = weighted("baseline_accuracy")
        rows.append(
            {
                "variant": name,
                "features_dropped": len(VARIANTS[name]),
                "pitches": int(pitches),
                "savant_acc": savant,
                "live_acc": live,
                "live_relative": (live - baseline) / baseline,
                "savant_relative": (savant - baseline) / baseline,
                "live_minus_savant": live - savant,
            }
        )

    summary = pd.DataFrame(rows)
    reference = summary.loc[summary["variant"] == "full", "live_relative"]
    baseline_relative = float(reference.iloc[0]) if len(reference) else float("nan")
    summary["vs_full"] = summary["live_relative"] - baseline_relative

    print()
    print("=" * 92)
    print("LIVE ACCURACY BY TRAINING FEATURE SET")
    print("=" * 92)
    print(
        f"{'variant':<18}{'dropped':>8}{'pitches':>9}"
        f"{'savant acc':>12}{'live acc':>10}"
        f"{'live rel':>11}{'live-savant':>13}{'vs full':>10}"
    )
    for row in summary.itertuples():
        print(
            f"{row.variant:<18}{row.features_dropped:>8}{row.pitches:>9}"
            f"{row.savant_acc:>11.2%}{row.live_acc:>10.2%}"
            f"{row.live_relative:>+11.1%}{row.live_minus_savant:>+13.2%}"
            f"{row.vs_full:>+10.1%}"
        )
    print()
    print("  live rel    = relative improvement over baseline, using live features")
    print("  live-savant = cost of live features under that same model")
    print("  vs full     = change in live relative improvement against production")
    return summary


def main() -> None:
    args = parse_args()
    client = GumboClient()
    scratch_root = Path(tempfile.mkdtemp(prefix="live_ablation_"))
    print(f"Scratch models: {scratch_root}")

    results: list[VariantResult] = []
    for raw_date in args.dates:
        game_date = date.fromisoformat(raw_date)
        targets = discover(args.data_root, game_date)
        if args.limit:
            targets = targets[: args.limit]
        zone_reference_frame = build_batter_zone_reference(
            args.data_root / "features" / "kg4" / "pitchers", through_date=game_date
        )
        zone_reference = {
            int(row.batter): (float(row.sz_top), float(row.sz_bot))
            for row in zone_reference_frame.itertuples()
        }
        print(f"\n{game_date}: {len(targets)} pitcher-game(s)")
        for game_pk, pitcher_id in targets:
            try:
                produced = evaluate_pitcher_game(
                    data_root=args.data_root,
                    client=client,
                    game_pk=game_pk,
                    pitcher_id=pitcher_id,
                    game_date=game_date,
                    zone_reference=zone_reference,
                    scratch=scratch_root,
                )
            except Exception as error:  # noqa: BLE001 - keep the sweep going
                print(f"  [skip] {game_pk}/{pitcher_id}: {error}")
                continue
            results.extend(produced)
            if produced:
                live = {r.variant: r.live_accuracy for r in produced}
                print(
                    f"  [ok]   {game_pk}/{pitcher_id} "
                    f"({produced[0].pitches} pitches) "
                    + "  ".join(f"{k}={v:.1%}" for k, v in live.items())
                )

    summary = report(results)
    if args.output and not summary.empty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([vars(r) for r in results]).to_csv(args.output, index=False)
        print(f"\nWrote per-game results to {args.output}")


if __name__ == "__main__":
    main()
