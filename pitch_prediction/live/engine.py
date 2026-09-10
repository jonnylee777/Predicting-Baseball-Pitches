"""Live pitch prediction engine.

Predicts the next pitch during a game in progress and records whether the
prediction actually beat the pitch.

Timing is the hard constraint. A pitch appears in the GUMBO feed a few seconds
after it is thrown, and the next pitch follows within roughly 15 seconds, so
the window between "we know the current state" and "the next pitch is thrown"
is narrow and variable. Two design choices make the engine robust to that
rather than dependent on winning a race:

* **Continuous re-prediction.** Every time the observable state for the pending
  pitch changes, the pitch is re-predicted and the previous prediction for it
  is superseded. Whatever prediction stands when the pitch is thrown is the one
  that counts, so a slow feed degrades the prediction's freshness rather than
  losing it.
* **Self-auditing.** Each prediction records the wall-clock instant it was
  made. When the pitch arrives, its feed ``startTime`` is compared against that
  instant and the prediction is marked ``before_pitch`` or ``late``. The
  accuracy log can then report only predictions that genuinely preceded their
  pitch, so a latency problem shows up as an honest number instead of an
  inflated one.

The engine never revises a prediction after the pitch is known.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from ..model import PitchModelTrainer
from .context import PregameContext
from .features import LiveFeatureBuilder, count_state, rolling_rates
from .gumbo import GumboClient, GumboSnapshot


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timecode(timecode: str) -> datetime:
    """Convert a GUMBO timecode (``YYYYMMDD_HHMMSS``) to a UTC datetime."""

    return datetime.strptime(timecode, "%Y%m%d_%H%M%S").replace(
        tzinfo=timezone.utc
    )


def _parse_feed_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass
class PitchPrediction:
    """One prediction for one pitch, plus its outcome once known."""

    game_pk: int
    pitcher_id: int
    pitcher_name: str

    at_bat_index: int
    pitch_number_of_ab: int
    pitch_number_of_game: int

    inning: int
    inning_topbot: str
    balls: int
    strikes: int
    outs: int
    batter_id: int
    batter_name: str

    predicted_pitch_type: str
    confidence: float
    probabilities: dict[str, float]

    predicted_at_utc: str
    feed_timestamp: str | None

    # Filled in once the pitch has actually been thrown.
    actual_pitch_type: str | None = None
    correct: bool | None = None
    pitch_start_utc: str | None = None
    prediction_lead_seconds: float | None = None
    timing: str | None = None

    # Diagnostics for the strike-zone estimate used on this pitch.
    strike_zone_source: str | None = None

    # True when this prediction was computed before the *previous* pitch was
    # thrown, by enumerating the states this pitch could arrive in.
    predicted_ahead: bool = False

    @property
    def key(self) -> tuple[int, int]:
        return (self.at_bat_index, self.pitch_number_of_ab)

    def resolve(self, actual: str | None, pitch_start: datetime | None) -> None:
        """Attach the thrown pitch and judge whether the prediction was in time."""

        self.actual_pitch_type = actual
        self.correct = (
            None if actual is None else actual == self.predicted_pitch_type
        )
        predicted_at = _parse_feed_time(self.predicted_at_utc)
        if pitch_start is not None:
            self.pitch_start_utc = pitch_start.isoformat()
            if predicted_at is not None:
                lead = (pitch_start - predicted_at).total_seconds()
                self.prediction_lead_seconds = lead
                self.timing = "before_pitch" if lead > 0 else "late"


@dataclass
class EngineStats:
    predictions_made: int = 0
    resolved: int = 0
    correct: int = 0
    before_pitch: int = 0
    late: int = 0
    superseded: int = 0
    predicted_ahead: int = 0
    leads: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.resolved if self.resolved else None

    def honest_accuracy(self, predictions: Iterable[PitchPrediction]) -> float | None:
        """Accuracy over predictions that provably preceded their pitch."""

        eligible = [
            prediction
            for prediction in predictions
            if prediction.timing == "before_pitch" and prediction.correct is not None
        ]
        if not eligible:
            return None
        return sum(1 for p in eligible if p.correct) / len(eligible)


class LivePredictionEngine:
    """Consume feed snapshots and emit predictions for the next pitch."""

    def __init__(
        self,
        *,
        model: Any,
        context: PregameContext,
        pitcher_name: str,
        trainer: PitchModelTrainer | None = None,
        win_probability: dict[int, float] | None = None,
        log_path: Path | None = None,
        clock: Any = _utc_now,
        speculate: bool = True,
        repertoire_size: int = 8,
        repertoire_min_share: float = 0.002,
    ) -> None:
        # Replaying a completed game substitutes a clock that returns the
        # snapshot's own timecode, so the timing audit measures the lead the
        # engine would really have had rather than comparing a 2026 pitch to
        # the present moment.
        self.clock = clock
        self.model = model
        self.context = context
        self.pitcher_name = pitcher_name
        self.trainer = trainer or PitchModelTrainer()
        self.builder = LiveFeatureBuilder(context, win_probability=win_probability)
        self.log_path = log_path

        # Predicting a pitch ahead only works when the model ignores the
        # previous pitch's measurements, because those cannot be enumerated.
        self.speculate = speculate and bool(self.trainer.exclude_features)
        # A candidate is only useful if the pitch actually gets thrown, so the
        # cutoffs are deliberately generous: a missed candidate costs a
        # fallback prediction, while an extra one costs about a millisecond.
        self.repertoire_size = repertoire_size
        self.repertoire_min_share = repertoire_min_share

        self.pending: PitchPrediction | None = None
        self.history: list[PitchPrediction] = []
        self.stats = EngineStats()

        self._pending_signature: tuple[Any, ...] | None = None
        self._resolved_keys: set[tuple[int, int]] = set()
        # Candidate state -> prediction computed before the previous pitch was
        # thrown. Cleared each time a real pitch lands.
        self._speculative: dict[tuple[Any, ...], PitchPrediction] = {}

    # --------------------------------------------------------------

    def observe(self, snapshot: GumboSnapshot) -> list[PitchPrediction]:
        """Process one snapshot: resolve thrown pitches, predict the next one.

        Returns the predictions resolved by this snapshot.
        """

        result = self.builder.build(snapshot, include_pending=True)
        if result.features.empty:
            return []

        thrown = self._resolve_thrown(snapshot, result)
        self._update_pending(snapshot, result)
        if self.speculate:
            self._speculate(snapshot, result)
        return thrown

    # --------------------------------------------------------------

    def _resolve_thrown(
        self, snapshot: GumboSnapshot, result: Any
    ) -> list[PitchPrediction]:
        """Match newly thrown pitches against the prediction that was standing."""

        resolved: list[PitchPrediction] = []
        pitch_times = self._pitch_start_times(snapshot)

        thrown_meta = result.meta[~result.meta["is_pending"]]
        for _, row in thrown_meta.iterrows():
            key = (int(row["gumbo_at_bat_index"]), int(row["gumbo_pitch_number"]))
            if key in self._resolved_keys:
                continue
            prediction = self.pending
            if prediction is None or prediction.key != key:
                # No standing prediction for this pitch: the engine started
                # mid-at-bat or the feed advanced by more than one pitch
                # between polls. Nothing to score.
                self._resolved_keys.add(key)
                continue

            prediction.resolve(row["actual_pitch_type"], pitch_times.get(key))
            self._record(prediction)
            resolved.append(prediction)
            self._resolved_keys.add(key)
            self.pending = None
            self._pending_signature = None
            # Candidates for the pitch just thrown, and anything before it,
            # can never be realised now. The candidates for the *next* pitch
            # must survive: they are the whole point, having been computed
            # before this pitch was thrown.
            self._prune_speculative(key)

        return resolved

    def _update_pending(self, snapshot: GumboSnapshot, result: Any) -> None:
        """Predict the pending pitch, re-predicting when its state changes."""

        index = result.pending_index
        if index is None:
            return

        features = result.features.loc[[index]]
        meta = result.meta.loc[index]
        key = (int(meta["gumbo_at_bat_index"]), int(meta["gumbo_pitch_number"]))
        if key in self._resolved_keys:
            return

        signature = (key, *self._state_signature(features.iloc[0]))
        if signature == self._pending_signature:
            return  # Nothing observable changed; the standing prediction holds.

        if self.pending is not None and self.pending.key == key:
            self.stats.superseded += 1

        adopted = self._speculative.pop(
            self._candidate_key(key, features.iloc[0]), None
        )
        # Only one state materialised, so the alternatives for this pitch go.
        for candidate in [c for c in self._speculative if c[:2] == key]:
            del self._speculative[candidate]
        if adopted is not None:
            # Computed before the previous pitch was thrown, so it keeps that
            # earlier timestamp and counts as predicted ahead.
            adopted.predicted_ahead = True
            self.pending = adopted
            self.stats.predicted_ahead += 1
        else:
            self.pending = self._predict(snapshot, features, meta, key)
            self.stats.predictions_made += 1
        self._pending_signature = signature

    # --------------------------------------------------------------

    def _predict(
        self,
        snapshot: GumboSnapshot,
        features: pd.DataFrame,
        meta: pd.Series,
        key: tuple[int, int],
    ) -> PitchPrediction:
        prepared = self.trainer._features(features)
        if hasattr(self.model, "feature_names_in_"):
            prepared = prepared.reindex(columns=list(self.model.feature_names_in_))

        predicted = str(self.model.predict(prepared)[0])
        probabilities: dict[str, float] = {}
        confidence = float("nan")
        if hasattr(self.model, "predict_proba"):
            raw = self.model.predict_proba(prepared)[0]
            probabilities = {
                str(label): float(value)
                for label, value in zip(self.model.classes_, raw)
            }
            confidence = float(np.max(raw))

        row = features.iloc[0]
        batter_id = int(row["batter"])
        return PitchPrediction(
            game_pk=snapshot.game_pk,
            pitcher_id=self.context.pitcher_id,
            pitcher_name=self.pitcher_name,
            at_bat_index=key[0],
            pitch_number_of_ab=key[1],
            pitch_number_of_game=int(row["pitch_number_of_game"]),
            inning=int(row["inning"]),
            inning_topbot=str(row["inning_topbot"]),
            balls=int(row["balls"]),
            strikes=int(row["strikes"]),
            outs=int(row["outs_when_up"]),
            batter_id=batter_id,
            batter_name=snapshot.player(batter_id).get("fullName", "Unknown"),
            predicted_pitch_type=predicted,
            confidence=confidence,
            probabilities=probabilities,
            predicted_at_utc=self.clock().isoformat(),
            feed_timestamp=snapshot.timestamp,
            strike_zone_source=str(meta["sz_source"]),
        )

    # --------------------------------------------------------------
    # PREDICTING A PITCH AHEAD
    # --------------------------------------------------------------

    @staticmethod
    def _candidate_key(
        key: tuple[int, int], row: pd.Series
    ) -> tuple[Any, ...]:
        """Identity of a pitch plus every state feature a candidate assumes.

        A candidate copies the current row and varies only the count and the
        previous pitch type, so it silently assumes the base state, outs and
        score are unchanged. They usually are inside an at-bat, but a steal,
        pickoff or wild pitch moves them. Those fields are therefore part of
        the key: if they move, no candidate matches and the pitch gets a fresh
        prediction. A prediction computed from a state that did not happen
        must never be served, even at the cost of a lower match rate.
        """

        def optional(value: Any) -> Any:
            if value is None:
                return None
            try:
                return None if pd.isna(value) else value
            except (TypeError, ValueError):
                return value

        previous = optional(row["pitch_type_of_prev_pitch"])
        return (
            key[0],
            key[1],
            int(row["batter"]),
            int(row["balls"]),
            int(row["strikes"]),
            None if previous is None else str(previous),
            int(row["outs_when_up"]),
            optional(row["on_1b"]),
            optional(row["on_2b"]),
            optional(row["on_3b"]),
            int(row["bat_score"]),
            int(row["fld_score"]),
        )

    def _prune_speculative(self, resolved: tuple[int, int]) -> None:
        """Drop candidates for pitches at or before ``resolved``."""

        at_bat, pitch_number = resolved
        for candidate in [
            c
            for c in self._speculative
            if c[0] < at_bat or (c[0] == at_bat and c[1] <= pitch_number)
        ]:
            del self._speculative[candidate]

    def _candidate_counts(self, balls: int, strikes: int) -> list[tuple[int, int]]:
        """Counts the next pitch of this at-bat could arrive on.

        A ball advances the count unless it walks the batter; a strike advances
        it unless it strikes them out; with two strikes a foul leaves the count
        alone. Outcomes that end the plate appearance are not enumerated: the
        next batter brings a longer gap, so the ordinary path has time.
        """

        candidates = []
        if balls < 3:
            candidates.append((balls + 1, strikes))
        if strikes < 2:
            candidates.append((balls, strikes + 1))
        else:
            candidates.append((balls, strikes))
        return candidates

    def _candidate_pitch_types(self) -> list[str]:
        prior = self.context.pitch_type_prior
        if not prior:
            return []
        ranked = sorted(prior.items(), key=lambda item: -item[1])
        return [
            name
            for name, share in ranked[: self.repertoire_size]
            if share > self.repertoire_min_share
        ]

    def _speculate(self, snapshot: GumboSnapshot, result: Any) -> None:
        """Predict the pitch after the pending one, for every state it may face.

        The feed publishes a pitch about as long after it is thrown as the gap
        to the next one, so waiting for it leaves no headroom. Everything the
        model still needs is enumerable once the previous pitch's measurements
        are excluded, so each candidate is scored now and kept until the feed
        reveals which one happened.
        """

        index = result.pending_index
        if index is None:
            return

        row = result.features.loc[index]
        meta = result.meta.loc[index]
        at_bat_index = int(meta["gumbo_at_bat_index"])
        pitch_number = int(meta["gumbo_pitch_number"])
        balls, strikes = int(row["balls"]), int(row["strikes"])

        pitch_types = self._candidate_pitch_types()
        if not pitch_types:
            return

        recent = list(result.recent_pitch_types)
        candidates: dict[tuple[Any, ...], pd.DataFrame] = {}
        for next_balls, next_strikes in self._candidate_counts(balls, strikes):
            for pitch_type in pitch_types:
                candidate = row.copy()
                candidate["balls"] = next_balls
                candidate["strikes"] = next_strikes
                candidate["count"] = f"{next_balls}-{next_strikes}"
                candidate["count_state"] = count_state(next_balls, next_strikes)
                candidate["pitch_number_of_ab"] = pitch_number + 1
                candidate["pitch_number_of_game"] = (
                    int(row["pitch_number_of_game"]) + 1
                )
                candidate["pitch_type_of_prev_pitch"] = pitch_type
                for name, value in rolling_rates((recent + [pitch_type])[-3:]).items():
                    candidate[name] = value
                key = self._candidate_key(
                    (at_bat_index, pitch_number + 1), candidate
                )
                candidates[key] = candidate.to_frame().T

        candidates.update(
            self._next_at_bat_candidates(snapshot, result, row, at_bat_index)
        )

        if not candidates:
            return

        keys = list(candidates)
        frame = pd.concat([candidates[k] for k in keys], ignore_index=True)
        # A candidate for the next batter belongs to the next at-bat index, so
        # each prediction takes the index from its own key. Stamping them all
        # with the current one would leave the record's key mismatched and the
        # pitch would never resolve against it.
        predictions = self._predict_frame(
            snapshot, frame, [key[0] for key in keys]
        )
        for key, prediction in zip(keys, predictions):
            self._speculative.setdefault(key, prediction)

    def _next_at_bat_candidates(
        self,
        snapshot: GumboSnapshot,
        result: Any,
        row: pd.Series,
        at_bat_index: int,
    ) -> dict[tuple[Any, ...], pd.DataFrame]:
        """Candidates for the first pitch to the *next* batter.

        The gap before a new batter's first pitch averages 31 seconds, so on a
        dedicated poller the ordinary path has time. Sharing one polling cycle
        across a slate erodes that margin, and first pitches became the largest
        single source of late predictions, so they are enumerated too.

        Only the two common endings are covered: the batter is retired, or the
        batter reaches first. Anything else -- an extra-base hit, a run
        scoring, the third out -- falls back to the ordinary path.
        """

        offense = (
            snapshot.payload.get("liveData", {}).get("linescore", {}).get("offense", {})
        )
        on_deck = offense.get("onDeck") or {}
        batter_id = on_deck.get("id")
        if not batter_id:
            return {}
        batter_id = int(batter_id)

        pitch_types = self._candidate_pitch_types()
        if not pitch_types:
            return {}

        player = snapshot.player(batter_id)
        birth_year = snapshot.birth_year(batter_id)
        stand = (player.get("batSide") or {}).get("code")
        sz_top, sz_bot = self.context.batter_zone(batter_id)

        outs = int(row["outs_when_up"])
        at_bat_number = int(row["at_bat_number_of_game"]) + 1
        recent = list(result.recent_pitch_types)

        base_row = row.copy()
        base_row["batter"] = batter_id
        base_row["balls"] = 0
        base_row["strikes"] = 0
        base_row["count"] = "0-0"
        base_row["count_state"] = count_state(0, 0)
        base_row["pitch_number_of_ab"] = 1
        base_row["at_bat_number_of_game"] = at_bat_number
        base_row["pitch_number_of_game"] = int(row["pitch_number_of_game"]) + 1
        base_row["n_thruorder_pitcher"] = (at_bat_number - 1) // 9 + 1
        if stand:
            base_row["stand"] = stand
        if birth_year is not None:
            base_row["age_bat"] = self.context.season - birth_year
        if sz_top is not None and sz_bot is not None:
            base_row["sz_top"] = sz_top
            base_row["sz_bot"] = sz_bot
            base_row["strike_zone_height"] = sz_top - sz_bot
        base_row["prior_pa_vs_pitcher_career"] = result.career_pa_vs_batter.get(
            batter_id, 0
        )
        base_row["n_priorpa_thisgame_player_at_bat"] = result.game_pa_by_batter.get(
            batter_id, 0
        )

        branches = self._at_bat_ending_branches(row, outs)

        candidates: dict[tuple[Any, ...], pd.DataFrame] = {}
        for branch in branches:
            for pitch_type in pitch_types:
                candidate = base_row.copy()
                for field, value in branch.items():
                    candidate[field] = value
                candidate["pitch_type_of_prev_pitch"] = pitch_type
                for name, value in rolling_rates((recent + [pitch_type])[-3:]).items():
                    candidate[name] = value
                key = self._candidate_key((at_bat_index + 1, 1), candidate)
                candidates[key] = candidate.to_frame().T
        return candidates

    def _at_bat_ending_branches(
        self, row: pd.Series, outs: int
    ) -> list[dict[str, Any]]:
        """The states the next batter could face, given how this at-bat ends.

        Runs are derived from the base state rather than guessed: a double
        scores whoever is on second and third, a home run scores everyone. That
        keeps the candidate count down while still covering the endings that
        actually happen.

        Endings not covered here -- a runner thrown out on the bases, a triple,
        a bases-loaded walk -- fall back to the ordinary path.
        """

        occupied = self._bases(row)
        first, second, third = occupied
        on_base = sum(1 for base in occupied if base is not None)
        batter_id = int(row["batter"])
        bat_score = int(row["bat_score"])
        fld_score = int(row["fld_score"])
        inning = int(row["inning"])

        def state(
            *,
            outs_when_up: int,
            bases: tuple[Any, Any, Any],
            runs: int = 0,
            inning_value: int | None = None,
        ) -> dict[str, Any]:
            scored = bat_score + runs
            return {
                "outs_when_up": outs_when_up,
                "on_1b": bases[0],
                "on_2b": bases[1],
                "on_3b": bases[2],
                "bat_score": scored,
                "pitcher_team_score_diff": fld_score - scored,
                "inning": inning if inning_value is None else inning_value,
            }

        branches: list[dict[str, Any]] = []

        if outs + 1 < 3:
            # Retired, inning continues. A runner on third can score on the way.
            branches.append(state(outs_when_up=outs + 1, bases=occupied))
            if third is not None:
                branches.append(
                    state(
                        outs_when_up=outs + 1,
                        bases=(first, second, None),
                        runs=1,
                    )
                )

        # Reached first: a walk or a single, forcing occupied bases along.
        forced = self._force_advance(row, batter_id)
        if forced is not None:
            branches.append(
                state(
                    outs_when_up=outs,
                    bases=(forced["on_1b"], forced["on_2b"], forced["on_3b"]),
                )
            )
        if third is not None:
            # A single that scores the runner from third.
            branches.append(
                state(
                    outs_when_up=outs,
                    bases=(batter_id, second, None),
                    runs=1,
                )
            )

        # A double: the batter is on second, the runner from first reaches
        # third, and anyone from second or third scores.
        branches.append(
            state(
                outs_when_up=outs,
                bases=(None, batter_id, first),
                runs=(1 if second is not None else 0)
                + (1 if third is not None else 0),
            )
        )

        # A home run clears the bases.
        branches.append(
            state(
                outs_when_up=outs,
                bases=(None, None, None),
                runs=1 + on_base,
            )
        )

        # The inning can end here too -- on the third out, or on a double play
        # from one out -- and the pitcher returns next inning to a clean slate.
        # The pitcher's own team bats in between and may score, and the key
        # pins the score, so the run total is enumerated.
        if outs >= 1:
            for scored in (0, 1, 2, 3):
                branch = state(
                    outs_when_up=0,
                    bases=(None, None, None),
                    inning_value=inning + 1,
                )
                branch["fld_score"] = fld_score + scored
                branch["pitcher_team_score_diff"] = (
                    fld_score + scored - bat_score
                )
                branches.append(branch)

        return branches

    @staticmethod
    def _bases(row: pd.Series) -> tuple[Any, Any, Any]:
        def occupant(column: str) -> Any:
            value = row[column]
            if value is None:
                return None
            try:
                return None if pd.isna(value) else int(value)
            except (TypeError, ValueError):
                return None

        return (occupant("on_1b"), occupant("on_2b"), occupant("on_3b"))

    @staticmethod
    def _force_advance(row: pd.Series, batter_id: int) -> dict[str, Any] | None:
        """Base state after the batter reaches first, advancing forced runners."""

        def occupant(column: str) -> int | None:
            value = row[column]
            if value is None:
                return None
            try:
                return None if pd.isna(value) else int(value)
            except (TypeError, ValueError):
                return None

        first, second, third = (
            occupant("on_1b"),
            occupant("on_2b"),
            occupant("on_3b"),
        )
        if first is not None and second is not None and third is not None:
            # Bases loaded: a walk scores a run, which this does not model.
            return None
        if first is None:
            return {"on_1b": batter_id, "on_2b": second, "on_3b": third}
        if second is None:
            return {"on_1b": batter_id, "on_2b": first, "on_3b": third}
        return {"on_1b": batter_id, "on_2b": first, "on_3b": second}

    def _predict_frame(
        self,
        snapshot: GumboSnapshot,
        frame: pd.DataFrame,
        at_bat_indices: int | list[int],
    ) -> list[PitchPrediction]:
        """Score many candidate states in one pass."""

        if isinstance(at_bat_indices, int):
            at_bat_indices = [at_bat_indices] * len(frame)

        prepared = self.trainer._features(frame)
        if hasattr(self.model, "feature_names_in_"):
            prepared = prepared.reindex(columns=list(self.model.feature_names_in_))

        labels = self.model.predict(prepared)
        probabilities = (
            self.model.predict_proba(prepared)
            if hasattr(self.model, "predict_proba")
            else None
        )
        stamped = self.clock().isoformat()

        records = []
        for position in range(len(frame)):
            row = frame.iloc[position]
            batter_id = int(row["batter"])
            if probabilities is not None:
                raw = probabilities[position]
                mapping = {
                    str(label): float(value)
                    for label, value in zip(self.model.classes_, raw)
                }
                confidence = float(np.max(raw))
            else:
                mapping, confidence = {}, float("nan")
            records.append(
                PitchPrediction(
                    game_pk=snapshot.game_pk,
                    pitcher_id=self.context.pitcher_id,
                    pitcher_name=self.pitcher_name,
                    at_bat_index=at_bat_indices[position],
                    pitch_number_of_ab=int(row["pitch_number_of_ab"]),
                    pitch_number_of_game=int(row["pitch_number_of_game"]),
                    inning=int(row["inning"]),
                    inning_topbot=str(row["inning_topbot"]),
                    balls=int(row["balls"]),
                    strikes=int(row["strikes"]),
                    outs=int(row["outs_when_up"]),
                    batter_id=batter_id,
                    batter_name=snapshot.player(batter_id).get("fullName", "Unknown"),
                    predicted_pitch_type=str(labels[position]),
                    confidence=confidence,
                    probabilities=mapping,
                    predicted_at_utc=stamped,
                    feed_timestamp=snapshot.timestamp,
                    strike_zone_source=None,
                )
            )
        return records

    @staticmethod
    def _state_signature(row: pd.Series) -> tuple[Any, ...]:
        """The observable state that should trigger a fresh prediction.

        Missing values are normalised because ``nan != nan``: leaving them raw
        would make every comparison unequal, so the engine would re-predict on
        every poll and re-stamp the prediction time, shrinking the measured
        lead over the pitch.
        """

        def normalize(value: Any) -> Any:
            if value is None:
                return None
            try:
                return None if pd.isna(value) else value
            except (TypeError, ValueError):
                return value

        return tuple(
            normalize(row[column])
            for column in (
                "balls",
                "strikes",
                "outs_when_up",
                "batter",
                "pitch_type_of_prev_pitch",
                "release_speed_of_prev_pitch",
                "zone_of_prev_pitch",
            )
        )

    @staticmethod
    def _pitch_start_times(
        snapshot: GumboSnapshot,
    ) -> dict[tuple[int, int], datetime | None]:
        times: dict[tuple[int, int], datetime | None] = {}
        for play in snapshot.all_plays:
            at_bat_index = int(play.get("about", {}).get("atBatIndex", -1))
            for event in play.get("playEvents", []):
                if not event.get("isPitch"):
                    continue
                key = (at_bat_index, int(event.get("pitchNumber", 0)))
                times[key] = _parse_feed_time(event.get("startTime"))
        return times

    def _record(self, prediction: PitchPrediction) -> None:
        self.history.append(prediction)
        self.stats.resolved += 1
        if prediction.correct:
            self.stats.correct += 1
        if prediction.timing == "before_pitch":
            self.stats.before_pitch += 1
        elif prediction.timing == "late":
            self.stats.late += 1
        if prediction.prediction_lead_seconds is not None:
            self.stats.leads.append(prediction.prediction_lead_seconds)

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(prediction)) + "\n")


def load_pregame_model(
    data_root: Path,
    game_date: str,
    pitcher_id: int,
    prefer_live: bool = True,
) -> tuple[Any, bool]:
    """Load a frozen pre-game model, preferring the ahead-capable one.

    Returns ``(model, is_live_model)``. The live model withholds the previous
    pitch's measurements so the engine can predict a pitch ahead; the
    production model is the fallback and can only predict once the feed
    publishes the previous pitch.
    """

    root = data_root / "models" / game_date
    live_path = root / "pitchers_live" / f"{pitcher_id}.joblib"
    if prefer_live and live_path.exists():
        return joblib.load(live_path), True

    path = root / "pitchers" / f"{pitcher_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen pre-game model for pitcher {pitcher_id} on {game_date}. "
            f"Run scripts.run_daily_pipeline for that date first."
        )
    return joblib.load(path), False


def build_context(
    data_root: Path,
    pitcher_id: int,
    game_date: Any,
    batter_zone_reference: dict[int, tuple[float, float]] | None = None,
) -> PregameContext:
    history_path = (
        data_root / "features" / "kg4" / "pitchers" / f"{pitcher_id}.csv"
    )
    if not history_path.exists():
        raise FileNotFoundError(
            f"No KG4 history for pitcher {pitcher_id} at {history_path}"
        )
    history = pd.read_csv(history_path, low_memory=False)
    return PregameContext.from_history(
        history,
        pitcher_id=pitcher_id,
        game_date=game_date,
        batter_zone_reference=batter_zone_reference,
    )
