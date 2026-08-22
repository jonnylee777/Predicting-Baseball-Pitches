from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from pitch_prediction.clients import ScheduleResult, Starter
from pitch_prediction.cleaning import CleaningSummary
from pitch_prediction.feature_engineering import FeatureDatasets
from pitch_prediction.pipeline import DailyStarterPipeline
from pitch_prediction.schema import StatcastSchema


COLUMNS = (
    "pitch_type",
    "game_date",
    "pitcher",
    "game_pk",
    "at_bat_number",
    "pitch_number",
)


class FakeMlbClient:
    def probable_starters(self, game_date: date) -> ScheduleResult:
        starter = Starter(
            game_pk=999,
            official_date=game_date.isoformat(),
            game_time_utc=f"{game_date}T18:00:00Z",
            game_status="Scheduled",
            side="away",
            team_id=1,
            team_name="Away",
            opponent_id=2,
            opponent_name="Home",
            pitcher_id=42,
            pitcher_name="Example Pitcher",
        )
        return ScheduleResult(1, [starter], [])

    def mlb_debut_date(self, player_id: int) -> date:
        return date(2025, 4, 1)


class FakeSavantClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, date, date]] = []

    def pitcher_pitches(
        self, player_id: int, start_date: date, end_date: date
    ) -> pd.DataFrame:
        self.calls.append((player_id, start_date, end_date))
        # Deliberately return a non-canonical order to exercise normalization.
        return pd.DataFrame(
            {
                "pitcher": [42, 42],
                "pitch_number": [1, 1],
                "pitch_type": ["FF", "FF"],
                "game_pk": [100, 100],
                "game_date": ["2025-05-31", "2025-05-31"],
                "at_bat_number": [1, 1],
            }
        )


class FakeCleaner:
    def __init__(self) -> None:
        self.cleaned_schema = StatcastSchema(COLUMNS)

    def transform(self, raw: pd.DataFrame, source: str) -> pd.DataFrame:
        return self.cleaned_schema.normalize(raw, source)

    def summarize(
        self, raw: pd.DataFrame, cleaned: pd.DataFrame
    ) -> CleaningSummary:
        return CleaningSummary(len(raw), len(cleaned), 0, len(cleaned.columns))


class FakeFeatureEngineer:
    def __init__(self) -> None:
        self.kg2_schema = StatcastSchema(COLUMNS)
        self.kg3_schema = StatcastSchema(COLUMNS)
        self.kg4_schema = StatcastSchema(COLUMNS)

    def transform(self, cleaned: pd.DataFrame, source: str) -> FeatureDatasets:
        frame = self.kg2_schema.normalize(cleaned, source)
        return FeatureDatasets(frame.copy(), frame.copy(), frame.copy())


class PipelineTests(unittest.TestCase):
    def test_run_writes_canonical_deduplicated_csv_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            savant = FakeSavantClient()
            pipeline = DailyStarterPipeline(
                output_root=root,
                schema=StatcastSchema(COLUMNS),
                cleaner=FakeCleaner(),
                feature_engineer=FakeFeatureEngineer(),
                mlb_client=FakeMlbClient(),
                savant_client=savant,
            )

            manifest_path = pipeline.run(date(2025, 6, 1))

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["downloads"]), 1)
            self.assertEqual(len(manifest["cleaning"]), 1)
            self.assertEqual(len(manifest["feature_engineering"]), 1)
            csv_path = Path(manifest["downloads"][0]["path"])
            frame = pd.read_csv(csv_path)
            self.assertEqual(tuple(frame.columns), COLUMNS)
            self.assertEqual(len(frame), 1)
            cleaned_path = Path(manifest["cleaning"][0]["cleaned_path"])
            self.assertEqual(tuple(pd.read_csv(cleaned_path).columns), COLUMNS)
            feature_result = manifest["feature_engineering"][0]
            for stage in ("kg2", "kg3", "kg4"):
                self.assertEqual(
                    tuple(pd.read_csv(feature_result[f"{stage}_path"]).columns),
                    COLUMNS,
                )
            self.assertEqual(savant.calls, [(42, date(2025, 4, 1), date(2025, 5, 31))])

    def test_dry_run_does_not_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            savant = FakeSavantClient()
            pipeline = DailyStarterPipeline(
                output_root=Path(directory),
                schema=StatcastSchema(COLUMNS),
                cleaner=FakeCleaner(),
                feature_engineer=FakeFeatureEngineer(),
                mlb_client=FakeMlbClient(),
                savant_client=savant,
            )
            pipeline.run(date(2025, 6, 1), dry_run=True)
            self.assertEqual(savant.calls, [])


if __name__ == "__main__":
    unittest.main()
