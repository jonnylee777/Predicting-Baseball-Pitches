from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from pitch_prediction.feature_engineering import (
    KG2_ADDED_COLUMNS,
    KG2_DROP_COLUMNS,
    KG3_DROP_COLUMNS,
    ROLLING_PITCH_COLUMNS,
    PitchFeatureEngineer,
)
from pitch_prediction.schema import StatcastSchema


class GausmanFeatureEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.cleaned = pd.read_csv(root / "Data" / "kg1_cleaned.csv", low_memory=False)
        cls.engineer = PitchFeatureEngineer(
            StatcastSchema.from_file(root / "config" / "cleaned_columns.txt")
        )
        cls.datasets = cls.engineer.transform(cls.cleaned, "Gausman feature fixture")

    def test_stage_shapes_and_column_orders_are_stable(self) -> None:
        self.assertEqual(self.datasets.kg2.shape, (24_840, 90))
        self.assertEqual(self.datasets.kg3.shape, (24_840, 87))
        self.assertEqual(self.datasets.kg4.shape, (24_840, 69))
        self.assertEqual(tuple(self.datasets.kg2.columns), self.engineer.kg2_schema.columns)
        self.assertEqual(tuple(self.datasets.kg3.columns), self.engineer.kg3_schema.columns)
        self.assertEqual(tuple(self.datasets.kg4.columns), self.engineer.kg4_schema.columns)

    def test_kg2_drops_and_adds_notebook_features(self) -> None:
        self.assertFalse(set(KG2_DROP_COLUMNS) & set(self.datasets.kg2.columns))
        self.assertTrue(set(KG2_ADDED_COLUMNS) <= set(self.datasets.kg2.columns))
        self.assertIn("pitch_number_of_ab", self.datasets.kg2)
        self.assertIn("at_bat_number_of_game", self.datasets.kg2)
        self.assertNotIn("pitch_number", self.datasets.kg2)
        self.assertNotIn("at_bat_number", self.datasets.kg2)

    def test_kg3_drops_notebook_features_and_has_fixed_rolling_schema(self) -> None:
        self.assertFalse(set(KG3_DROP_COLUMNS) & set(self.datasets.kg3.columns))
        self.assertTrue(set(ROLLING_PITCH_COLUMNS) <= set(self.datasets.kg3.columns))
        subset = self.engineer.transform(
            self.cleaned.iloc[:100].copy(), "limited repertoire fixture"
        )
        self.assertEqual(tuple(subset.kg3.columns), tuple(self.datasets.kg3.columns))
        self.assertEqual(tuple(subset.kg4.columns), tuple(self.datasets.kg4.columns))

    def test_pitch_types_remain_uncombined_through_every_stage(self) -> None:
        expected = self.cleaned["pitch_type"].value_counts().sort_index()
        for frame in (self.datasets.kg2, self.datasets.kg3, self.datasets.kg4):
            pd.testing.assert_series_equal(
                frame["pitch_type"].value_counts().sort_index(), expected
            )
            self.assertEqual(int((frame["pitch_type"] == "SI").sum()), 195)

    def test_previous_pitch_type_excludes_current_pitch(self) -> None:
        kg2 = self.datasets.kg2
        expected = kg2.groupby("game_pk")["pitch_type"].shift(1)
        pd.testing.assert_series_equal(
            kg2["pitch_type_of_prev_pitch"],
            expected,
            check_names=False,
        )

    def test_count_features_follow_notebook_classification(self) -> None:
        states = {
            (0, 0): "even_count",
            (0, 2): "put_away_count",
            (2, 0): "hitters_count",
            (1, 0): "pitcher_behind",
            (0, 1): "pitcher_ahead",
            (3, 2): "full_count",
        }
        kg2 = self.datasets.kg2
        for (balls, strikes), expected_state in states.items():
            row = kg2.loc[kg2["balls"].eq(balls) & kg2["strikes"].eq(strikes)].iloc[0]
            self.assertEqual(row["count"], f"{balls}-{strikes}")
            self.assertEqual(row["count_state"], expected_state)

    def test_rolling_rates_only_use_the_previous_three_pitches(self) -> None:
        kg3 = self.datasets.kg3
        first_pitch = kg3.groupby("pitcher", sort=False).head(1).index
        self.assertTrue(kg3.loc[first_pitch, list(ROLLING_PITCH_COLUMNS)].isna().all().all())

        row_index = 10
        prior_types = kg3.iloc[row_index - 3 : row_index]["pitch_type"]
        expected_rates = prior_types.value_counts(normalize=True)
        for pitch_type, expected_rate in expected_rates.items():
            column = f"prev3_pitch_rate_{pitch_type}"
            self.assertAlmostEqual(kg3.loc[row_index, column], expected_rate)

    def test_unknown_pitch_code_uses_other_rolling_bucket(self) -> None:
        sample = self.cleaned.iloc[:10].copy()
        sample.loc[sample.index[0], "pitch_type"] = "ZZ"
        kg3 = self.engineer.transform(sample, "unknown pitch fixture").kg3
        self.assertEqual(kg3.loc[1, "prev3_pitch_rate_OTHER"], 1.0)

    def test_empty_rookie_history_still_has_canonical_schemas(self) -> None:
        empty = pd.DataFrame(columns=self.engineer.cleaned_schema.columns)
        datasets = self.engineer.transform(empty, "empty rookie fixture")
        self.assertEqual(datasets.kg2.shape, (0, 90))
        self.assertEqual(datasets.kg3.shape, (0, 87))
        self.assertEqual(datasets.kg4.shape, (0, 69))


if __name__ == "__main__":
    unittest.main()
