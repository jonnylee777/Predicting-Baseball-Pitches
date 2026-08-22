"""Canonical Statcast schema validation and column ordering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class SchemaMismatchError(ValueError):
    """Raised when a Savant export no longer matches the canonical schema."""


@dataclass(frozen=True)
class StatcastSchema:
    columns: tuple[str, ...]

    @classmethod
    def from_file(cls, path: Path) -> "StatcastSchema":
        columns = tuple(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if not columns:
            raise ValueError(f"Schema file is empty: {path}")
        duplicates = sorted({name for name in columns if columns.count(name) > 1})
        if duplicates:
            raise ValueError(f"Schema contains duplicate columns: {duplicates}")
        return cls(columns)

    @property
    def fingerprint(self) -> str:
        joined = "\n".join(self.columns).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()

    def normalize(self, frame: pd.DataFrame, source: str) -> pd.DataFrame:
        actual = tuple(str(column).strip().strip('"') for column in frame.columns)
        duplicate_columns = sorted({name for name in actual if actual.count(name) > 1})
        expected_set = set(self.columns)
        actual_set = set(actual)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if duplicate_columns or missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            if duplicate_columns:
                details.append(f"duplicates={duplicate_columns}")
            raise SchemaMismatchError(
                f"Statcast schema mismatch for {source}: " + "; ".join(details)
            )
        normalized = frame.copy()
        normalized.columns = actual
        return normalized.loc[:, self.columns]

    def validate_csv(self, path: Path) -> None:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if tuple(header) != self.columns:
            raise SchemaMismatchError(f"CSV column order is not canonical: {path}")
