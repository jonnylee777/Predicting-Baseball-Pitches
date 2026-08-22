from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pitch_prediction.schema import SchemaMismatchError, StatcastSchema


class StatcastSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = StatcastSchema(("pitch_type", "game_date", "pitcher"))

    def test_normalize_reorders_columns(self) -> None:
        frame = pd.DataFrame(
            {"pitcher": [1], "pitch_type": ["FF"], "game_date": ["2025-01-01"]}
        )
        normalized = self.schema.normalize(frame, "test frame")
        self.assertEqual(list(normalized.columns), list(self.schema.columns))

    def test_normalize_rejects_missing_and_extra_columns(self) -> None:
        frame = pd.DataFrame({"pitch_type": [], "unexpected": [], "pitcher": []})
        with self.assertRaisesRegex(SchemaMismatchError, "missing=.*game_date"):
            self.schema.normalize(frame, "test frame")

    def test_validate_csv_requires_exact_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-order.csv"
            pd.DataFrame(columns=("game_date", "pitch_type", "pitcher")).to_csv(
                path, index=False
            )
            with self.assertRaises(SchemaMismatchError):
                self.schema.validate_csv(path)

    def test_repository_schema_matches_kevin_gausman_export(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        canonical = StatcastSchema.from_file(
            repository_root / "config" / "statcast_columns.txt"
        )
        with (repository_root / "Data" / "592332_data-2.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            gausman_header = tuple(next(csv.reader(handle)))
        self.assertEqual(canonical.columns, gausman_header)


if __name__ == "__main__":
    unittest.main()
