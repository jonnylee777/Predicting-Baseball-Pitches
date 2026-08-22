from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from pitch_prediction.cleaning import PitchDataCleaner
from pitch_prediction.schema import StatcastSchema


class GausmanCleaningRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.raw = pd.read_csv(root / "Data" / "592332_data-2.csv", low_memory=False)
        cls.notebook_result = pd.read_csv(
            root / "Data" / "kg1_cleaned.csv", low_memory=False
        )
        cls.cleaned_schema = StatcastSchema.from_file(
            root / "config" / "cleaned_columns.txt"
        )
        cleaner = PitchDataCleaner(
            StatcastSchema.from_file(root / "config" / "statcast_columns.txt"),
            cls.cleaned_schema,
        )
        cls.actual = cleaner.transform(cls.raw, "Kevin Gausman regression fixture")
        cls.summary = cleaner.summarize(cls.raw, cls.actual)

    def test_output_shape_and_order_match_cleaning_notebook(self) -> None:
        self.assertEqual(self.actual.shape, (24_840, 118))
        self.assertEqual(tuple(self.actual.columns), self.cleaned_schema.columns)

    def test_all_values_match_cleaning_notebook_output(self) -> None:
        pd.testing.assert_frame_equal(
            self.actual,
            self.notebook_result,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_pitch_types_are_not_combined(self) -> None:
        expected_counts = self.raw.dropna(subset=["pitch_type"])[
            "pitch_type"
        ].value_counts().sort_index()
        actual_counts = self.actual["pitch_type"].value_counts().sort_index()
        pd.testing.assert_series_equal(actual_counts, expected_counts)
        self.assertEqual(int(actual_counts["SI"]), 195)

    def test_summary_reports_dropped_unclassified_pitches(self) -> None:
        self.assertEqual(self.summary.rows_before, 25_000)
        self.assertEqual(self.summary.rows_after, 24_840)
        self.assertEqual(self.summary.rows_dropped_without_pitch_type, 160)
        self.assertEqual(self.summary.column_count, 118)


if __name__ == "__main__":
    unittest.main()
