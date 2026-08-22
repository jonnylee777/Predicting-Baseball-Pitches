"""Orchestration for the nightly probable-starter data pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cleaning import PitchDataCleaner
from .clients import BaseballSavantClient, MlbStatsClient, Starter
from .feature_engineering import PitchFeatureEngineer
from .model import PitchModelTrainer
from .schema import StatcastSchema


STATCAST_FIRST_DATE = date(2008, 3, 25)
DEDUPLICATION_COLUMNS = ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
SORT_COLUMNS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]


@dataclass(frozen=True)
class PitcherDownload:
    pitcher_id: int
    pitcher_name: str
    path: str
    requested_start: str | None
    requested_end: str
    rows_before: int
    rows_after: int
    rows_added: int


@dataclass(frozen=True)
class PitcherCleaning:
    pitcher_id: int
    pitcher_name: str
    raw_path: str
    cleaned_path: str
    rows_before: int
    rows_after: int
    rows_dropped_without_pitch_type: int
    column_count: int


@dataclass(frozen=True)
class PitcherFeatureEngineering:
    pitcher_id: int
    pitcher_name: str
    cleaned_path: str
    kg2_path: str
    kg2_rows: int
    kg2_columns: int
    kg3_path: str
    kg3_rows: int
    kg3_columns: int
    kg4_path: str
    kg4_rows: int
    kg4_columns: int


class PipelineRunError(RuntimeError):
    pass


def _date_chunks(
    start: date,
    end: date,
    maximum_days: int = 2190,
) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=maximum_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv.tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary_path = Path(handle.name)
        frame.to_csv(handle, index=False)

    os.replace(temporary_path, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json.tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    os.replace(temporary_path, path)


class DailyStarterPipeline:
    def __init__(
        self,
        output_root: Path,
        schema: StatcastSchema,
        cleaner: PitchDataCleaner,
        feature_engineer: PitchFeatureEngineer,
        model_trainer: PitchModelTrainer | None = None,
        mlb_client: MlbStatsClient | None = None,
        savant_client: BaseballSavantClient | None = None,
        refresh_days: int = 7,
    ) -> None:
        if refresh_days < 1:
            raise ValueError("refresh_days must be at least 1")

        self.output_root = output_root
        self.schema = schema
        self.cleaner = cleaner
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.mlb = mlb_client or MlbStatsClient()
        self.savant = savant_client or BaseballSavantClient()
        self.refresh_days = refresh_days

    def run(self, game_date: date, dry_run: bool = False) -> Path:
        started_at = datetime.now(timezone.utc)
        schedule = self.mlb.probable_starters(game_date)

        run_dir = self.output_root / "runs" / game_date.isoformat()
        manifest_path = run_dir / "manifest.json"

        manifest: dict = {
            "pipeline_version": 4,
            "game_date": game_date.isoformat(),
            "history_through": (game_date - timedelta(days=1)).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "schema_column_count": len(self.schema.columns),
            "schema_sha256": self.schema.fingerprint,
            "cleaned_schema_column_count": len(self.cleaner.cleaned_schema.columns),
            "cleaned_schema_sha256": self.cleaner.cleaned_schema.fingerprint,
            "kg2_schema_sha256": self.feature_engineer.kg2_schema.fingerprint,
            "kg3_schema_sha256": self.feature_engineer.kg3_schema.fingerprint,
            "kg4_schema_sha256": self.feature_engineer.kg4_schema.fingerprint,
            "game_count": schedule.game_count,
            "starters": [
                asdict(starter)
                for starter in schedule.starters
            ],
            "missing_probable_pitchers": [
                asdict(item)
                for item in schedule.missing_probables
            ],
            "downloads": [],
            "cleaning": [],
            "feature_engineering": [],
            "modeling": [],
            "errors": [],
            "dry_run": dry_run,
        }

        if not dry_run:
            unique_starters = {
                starter.pitcher_id: starter
                for starter in schedule.starters
            }

            for starter in unique_starters.values():

                # -------------------------------------------------
                # 1. Download pitcher history
                # -------------------------------------------------
                try:
                    download = self._download_pitcher(
                        starter,
                        game_date - timedelta(days=1),
                    )

                    manifest["downloads"].append(
                        asdict(download)
                    )

                except Exception as exc:
                    manifest["errors"].append(
                        {
                            "stage": "download",
                            "pitcher_id": starter.pitcher_id,
                            "pitcher_name": starter.pitcher_name,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    continue

                # -------------------------------------------------
                # 2. Clean pitcher history
                # -------------------------------------------------
                try:
                    cleaning = self._clean_pitcher(
                        starter,
                        Path(download.path),
                    )

                    manifest["cleaning"].append(
                        asdict(cleaning)
                    )

                except Exception as exc:
                    manifest["errors"].append(
                        {
                            "stage": "cleaning",
                            "pitcher_id": starter.pitcher_id,
                            "pitcher_name": starter.pitcher_name,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    continue

                # -------------------------------------------------
                # 3. Feature engineering
                # -------------------------------------------------
                try:
                    features = self._engineer_pitcher(
                        starter,
                        Path(cleaning.cleaned_path),
                    )

                    manifest["feature_engineering"].append(
                        asdict(features)
                    )

                except Exception as exc:
                    manifest["errors"].append(
                        {
                            "stage": "feature_engineering",
                            "pitcher_id": starter.pitcher_id,
                            "pitcher_name": starter.pitcher_name,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    continue

                # -------------------------------------------------
                # 4. Modeling
                # -------------------------------------------------
                if self.model_trainer is not None:

                    try:
                        kg4 = pd.read_csv(
                            features.kg4_path,
                            low_memory=False,
                        )

                        model_result = self.model_trainer.train(
                            kg4,
                            pitcher_id=starter.pitcher_id,
                            pitcher_name=starter.pitcher_name,
                            reference_season=game_date.year,
                            model_dir=(
                                self.output_root
                                / "models"
                                / game_date.isoformat()
                                / "pitchers"
                            ),
                        )

                        manifest["modeling"].append(
                            asdict(model_result)
                        )

                    except Exception as exc:
                        manifest["errors"].append(
                            {
                                "stage": "modeling",
                                "pitcher_id": starter.pitcher_id,
                                "pitcher_name": starter.pitcher_name,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )

            # -----------------------------------------------------
            # 5. Validate all output CSV schemas
            # -----------------------------------------------------
            try:
                self._validate_all_output_csvs()

            except Exception as exc:
                manifest["errors"].append(
                    {
                        "stage": "cross_pitcher_schema_validation",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

        manifest["finished_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        _atomic_json(
            manifest,
            manifest_path,
        )

        if manifest["errors"]:
            raise PipelineRunError(
                f"{len(manifest['errors'])} pipeline error(s); "
                f"see {manifest_path}"
            )

        return manifest_path

    def _validate_all_output_csvs(self) -> None:
        raw_dir = (
            self.output_root
            / "raw"
            / "pitchers"
        )

        for csv_path in sorted(
            raw_dir.glob("*.csv")
        ):
            self.schema.validate_csv(
                csv_path
            )

        cleaned_dir = (
            self.output_root
            / "processed"
            / "pitchers"
        )

        for csv_path in sorted(
            cleaned_dir.glob("*.csv")
        ):
            self.cleaner.cleaned_schema.validate_csv(
                csv_path
            )

        for stage, schema in (
            (
                "kg2",
                self.feature_engineer.kg2_schema,
            ),
            (
                "kg3",
                self.feature_engineer.kg3_schema,
            ),
            (
                "kg4",
                self.feature_engineer.kg4_schema,
            ),
        ):

            feature_dir = (
                self.output_root
                / "features"
                / stage
                / "pitchers"
            )

            for csv_path in sorted(
                feature_dir.glob("*.csv")
            ):
                schema.validate_csv(
                    csv_path
                )

    def _clean_pitcher(
        self,
        starter: Starter,
        raw_path: Path,
    ) -> PitcherCleaning:

        cleaned_dir = (
            self.output_root
            / "processed"
            / "pitchers"
        )

        cleaned_path = (
            cleaned_dir
            / f"{starter.pitcher_id}.csv"
        )

        metadata_path = (
            cleaned_dir
            / f"{starter.pitcher_id}.json"
        )

        raw = pd.read_csv(
            raw_path,
            low_memory=False,
        )

        cleaned = self.cleaner.transform(
            raw,
            str(raw_path),
        )

        summary = self.cleaner.summarize(
            raw,
            cleaned,
        )

        _atomic_csv(
            cleaned,
            cleaned_path,
        )

        self.cleaner.cleaned_schema.validate_csv(
            cleaned_path
        )

        metadata = {
            "pitcher_id": starter.pitcher_id,
            "pitcher_name": starter.pitcher_name,
            "raw_path": str(raw_path),
            "cleaned_path": str(cleaned_path),
            "cleaned_schema_sha256":
                self.cleaner.cleaned_schema.fingerprint,
            **asdict(summary),
            "updated_at_utc":
                datetime.now(timezone.utc).isoformat(),
        }

        _atomic_json(
            metadata,
            metadata_path,
        )

        return PitcherCleaning(
            pitcher_id=starter.pitcher_id,
            pitcher_name=starter.pitcher_name,
            raw_path=str(raw_path),
            cleaned_path=str(cleaned_path),
            **asdict(summary),
        )

    def _engineer_pitcher(
        self,
        starter: Starter,
        cleaned_path: Path,
    ) -> PitcherFeatureEngineering:

        cleaned = pd.read_csv(
            cleaned_path,
            low_memory=False,
        )

        datasets = self.feature_engineer.transform(
            cleaned,
            str(cleaned_path),
        )

        stage_data = {
            "kg2": (
                datasets.kg2,
                self.feature_engineer.kg2_schema,
            ),
            "kg3": (
                datasets.kg3,
                self.feature_engineer.kg3_schema,
            ),
            "kg4": (
                datasets.kg4,
                self.feature_engineer.kg4_schema,
            ),
        }

        paths: dict[str, Path] = {}

        for stage, (
            frame,
            stage_schema,
        ) in stage_data.items():

            path = (
                self.output_root
                / "features"
                / stage
                / "pitchers"
                / f"{starter.pitcher_id}.csv"
            )

            _atomic_csv(
                frame,
                path,
            )

            stage_schema.validate_csv(
                path
            )

            paths[stage] = path

        result = PitcherFeatureEngineering(
            pitcher_id=starter.pitcher_id,
            pitcher_name=starter.pitcher_name,
            cleaned_path=str(cleaned_path),

            kg2_path=str(paths["kg2"]),
            kg2_rows=len(datasets.kg2),
            kg2_columns=len(datasets.kg2.columns),

            kg3_path=str(paths["kg3"]),
            kg3_rows=len(datasets.kg3),
            kg3_columns=len(datasets.kg3.columns),

            kg4_path=str(paths["kg4"]),
            kg4_rows=len(datasets.kg4),
            kg4_columns=len(datasets.kg4.columns),
        )

        metadata = {
            **asdict(result),

            "kg2_schema_sha256":
                self.feature_engineer.kg2_schema.fingerprint,

            "kg3_schema_sha256":
                self.feature_engineer.kg3_schema.fingerprint,

            "kg4_schema_sha256":
                self.feature_engineer.kg4_schema.fingerprint,

            "updated_at_utc":
                datetime.now(timezone.utc).isoformat(),
        }

        _atomic_json(
            metadata,
            self.output_root
            / "features"
            / "manifests"
            / f"{starter.pitcher_id}.json",
        )

        return result

    def _download_pitcher(
        self,
        starter: Starter,
        history_through: date,
    ) -> PitcherDownload:

        pitcher_dir = (
            self.output_root
            / "raw"
            / "pitchers"
        )

        # MLBAM ID remains stable even if the
        # player's displayed name changes.
        csv_path = (
            pitcher_dir
            / f"{starter.pitcher_id}.csv"
        )

        metadata_path = (
            pitcher_dir
            / f"{starter.pitcher_id}.json"
        )

        if csv_path.exists():

            existing = self.schema.normalize(
                pd.read_csv(
                    csv_path,
                    low_memory=False,
                ),
                str(csv_path),
            )

            self.schema.validate_csv(
                csv_path
            )

        else:

            existing = pd.DataFrame(
                columns=self.schema.columns
            )

        mlb_debut = self.mlb.mlb_debut_date(
            starter.pitcher_id
        )

        history_start = max(
            mlb_debut,
            STATCAST_FIRST_DATE,
        )

        covered_through: date | None = None

        if (
            csv_path.exists()
            and metadata_path.exists()
        ):

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            if metadata.get(
                "covered_through"
            ):

                covered_through = (
                    date.fromisoformat(
                        metadata[
                            "covered_through"
                        ]
                    )
                )

        elif not existing.empty:

            covered_through = (
                pd.to_datetime(
                    existing["game_date"]
                )
                .max()
                .date()
            )

        request_start = history_start

        if covered_through is not None:

            request_start = max(
                history_start,

                covered_through
                - timedelta(
                    days=self.refresh_days - 1
                ),
            )

        incoming_frames: list[
            pd.DataFrame
        ] = []

        if request_start <= history_through:

            for (
                chunk_start,
                chunk_end,
            ) in _date_chunks(
                request_start,
                history_through,
            ):

                incoming = (
                    self.savant.pitcher_pitches(
                        starter.pitcher_id,
                        chunk_start,
                        chunk_end,
                    )
                )

                incoming = self.schema.normalize(
                    incoming,
                    (
                        "Baseball Savant "
                        f"pitcher={starter.pitcher_id} "
                        f"dates={chunk_start}..{chunk_end}"
                    ),
                )

                if not incoming.empty:

                    pitcher_ids = set(
                        pd.to_numeric(
                            incoming["pitcher"],
                            errors="coerce",
                        )
                        .dropna()
                        .astype(int)
                        .unique()
                    )

                    if pitcher_ids != {
                        starter.pitcher_id
                    }:

                        raise ValueError(
                            "Savant returned pitcher IDs "
                            f"{sorted(pitcher_ids)} for "
                            "requested pitcher "
                            f"{starter.pitcher_id}"
                        )

                incoming_frames.append(
                    incoming
                )

        rows_before = len(existing)

        if (
            incoming_frames
            and not existing.empty
        ):

            existing_dates = (
                pd.to_datetime(
                    existing["game_date"],
                    errors="coerce",
                )
                .dt
                .date
            )

            # Replacing the overlap incorporates
            # corrections and removed rows,
            # not just newly appended records.
            retained_existing = existing.loc[
                existing_dates
                < request_start
            ]

        else:

            retained_existing = existing

        frames = [
            retained_existing,
            *incoming_frames,
        ]

        combined = (
            pd.concat(
                frames,
                ignore_index=True,
            )
            if frames
            else existing
        )

        combined = self.schema.normalize(
            combined,
            f"merged pitcher {starter.pitcher_id}",
        )

        if not combined.empty:

            combined = combined.drop_duplicates(
                subset=DEDUPLICATION_COLUMNS,
                keep="last",
            )

            combined = (
                combined
                .sort_values(
                    SORT_COLUMNS,
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        _atomic_csv(
            combined,
            csv_path,
        )

        self.schema.validate_csv(
            csv_path
        )

        metadata = {
            "pitcher_id":
                starter.pitcher_id,

            "pitcher_name":
                starter.pitcher_name,

            "mlb_debut_date":
                mlb_debut.isoformat(),

            "history_start_date":
                history_start.isoformat(),

            "covered_through":
                max(
                    history_through,
                    covered_through
                    or history_through,
                ).isoformat(),

            "schema_sha256":
                self.schema.fingerprint,

            "row_count":
                len(combined),

            "updated_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        _atomic_json(
            metadata,
            metadata_path,
        )

        return PitcherDownload(
            pitcher_id=starter.pitcher_id,
            pitcher_name=starter.pitcher_name,

            path=str(
                csv_path
            ),

            requested_start=(
                request_start.isoformat()
                if request_start
                <= history_through
                else None
            ),

            requested_end=(
                history_through.isoformat()
            ),

            rows_before=rows_before,

            rows_after=len(
                combined
            ),

            rows_added=(
                len(combined)
                - rows_before
            ),
        )