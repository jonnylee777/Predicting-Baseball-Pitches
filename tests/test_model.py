import pandas as pd

from pitch_prediction.model import (
    PitchModelTrainer,
)


def test_current_season_has_highest_weight():

    trainer = PitchModelTrainer()

    data = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                [
                    "2026-05-01",
                    "2025-05-01",
                    "2024-05-01",
                ]
            )
        }
    )

    weights = trainer._season_sample_weights(
        data,
        reference_season=2026,
    )

    assert weights[0] == 1.0
    assert weights[0] > weights[1] > weights[2]


def test_long_career_has_faster_decay():

    trainer = PitchModelTrainer()

    short_career_decay = (
        trainer._decay_factor(3)
    )

    long_career_decay = (
        trainer._decay_factor(10)
    )

    assert (
        long_career_decay
        < short_career_decay
    )


def test_weight_never_goes_below_minimum():

    trainer = PitchModelTrainer(
        minimum_sample_weight=0.05
    )

    data = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                [
                    "2026-05-01",
                    "2000-05-01",
                ]
            )
        }
    )

    weights = trainer._season_sample_weights(
        data,
        reference_season=2026,
    )

    assert weights.min() >= 0.05