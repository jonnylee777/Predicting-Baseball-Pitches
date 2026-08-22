"""Run post-game pitch prediction replay for one starter."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from pitch_prediction.cleaning import (
    PitchDataCleaner,
)
from pitch_prediction.feature_engineering import (
    PitchFeatureEngineer,
)
from pitch_prediction.model import (
    PitchModelTrainer,
)
from pitch_prediction.pipeline import (
    DailyStarterPipeline,
)
from pitch_prediction.postgame_replay import (
    PostgameReplayer,
)
from pitch_prediction.schema import (
    StatcastSchema,
)


OUTPUT_ROOT = Path(
    "Data/daily_pipeline"
)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Game date YYYY-MM-DD",
    )

    parser.add_argument(
        "--pitcher-id",
        required=True,
        type=int,
        help="MLB pitcher ID",
    )

    parser.add_argument(
        "--game-pk",
        type=int,
        default=None,
        help=(
            "Optional MLB game_pk. "
            "Useful for doubleheaders."
        ),
    )

    args = parser.parse_args()

    game_date = date.fromisoformat(
        args.date
    )

    # ========================================================
    # PROJECT COMPONENTS
    # ========================================================

    raw_schema = (
        StatcastSchema.from_file(
            Path(
                "config/statcast_columns.txt"
            )
        )
    )

    cleaned_schema = (
        StatcastSchema.from_file(
            Path(
                "config/cleaned_columns.txt"
            )
        )
    )

    cleaner = PitchDataCleaner(
        raw_schema,
        cleaned_schema,
    )

    feature_engineer = (
        PitchFeatureEngineer(
            cleaned_schema
        )
    )

    trainer = PitchModelTrainer()

    pipeline = DailyStarterPipeline(
        output_root=OUTPUT_ROOT,
        schema=raw_schema,
        cleaner=cleaner,
        feature_engineer=(
            feature_engineer
        ),
        model_trainer=None,
    )

    # ========================================================
    # FIND STARTER
    # ========================================================

    schedule = (
        pipeline.mlb.probable_starters(
            game_date
        )
    )

    matches = [
        starter
        for starter
        in schedule.starters
        if (
            starter.pitcher_id
            == args.pitcher_id
        )
    ]

    if args.game_pk is not None:

        matches = [
            starter
            for starter
            in matches
            if (
                starter.game_pk
                == args.game_pk
            )
        ]

    if not matches:

        raise ValueError(
            f"No probable starter with pitcher ID "
            f"{args.pitcher_id} was found on "
            f"{game_date}."
        )

    if len(matches) > 1:

        game_pks = [
            starter.game_pk
            for starter
            in matches
        ]

        raise ValueError(
            "Multiple matching games found. "
            "Specify --game-pk. "
            f"Possible game_pks: {game_pks}"
        )

    starter = matches[0]

    print()
    print(
        "Post-game replay:"
    )

    print(
        f"Pitcher: "
        f"{starter.pitcher_name}"
    )

    print(
        f"Game PK: "
        f"{starter.game_pk}"
    )

    print(
        f"Opponent: "
        f"{starter.opponent_name}"
    )

    print(
        f"Status: "
        f"{starter.game_status}"
    )

    # ========================================================
    # REPLAY
    # ========================================================

    replayer = PostgameReplayer(
        pipeline=pipeline,
        model_trainer=trainer,
        output_root=OUTPUT_ROOT,
    )

    result = replayer.replay(
        starter=starter,
        game_date=game_date,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "POST-GAME REPLAY RESULTS"
    )
    print("=" * 70)

    print(
        f"Pitcher: "
        f"{result.pitcher_name}"
    )

    print(
        f"Pitches predicted: "
        f"{result.pitch_count}"
    )

    print()

    print(
        f"Model accuracy: "
        f"{result.model_accuracy:.2%}"
    )

    print(
        f"Stratified baseline accuracy: "
        f"{result.baseline_accuracy:.2%}"
    )

    print()

    print(
        f"Absolute lift: "
        f"{result.accuracy_over_baseline * 100:+.2f} pp"
    )

    if (
        result.relative_improvement
        is not None
    ):

        print(
            f"Relative improvement: "
            f"{result.relative_improvement:+.2%}"
        )

    else:

        print(
            "Relative improvement: undefined"
        )

    print()

    print(
        f"Baseline strategy: "
        f"{result.baseline_strategy}"
    )

    print()

    print(
        f"Prediction log: "
        f"{result.predictions_path}"
    )

    print(
        f"Summary: "
        f"{result.summary_path}"
    )


if __name__ == "__main__":
    main()