# MLB Pitch Prediction

An end-to-end machine learning system for predicting the next pitch type thrown by MLB starting pitchers using Statcast pitch-by-pitch data.

The project began as a notebook-based case study on Kevin Gausman and has since been expanded into a modular pipeline that dynamically retrieves pitcher data, engineers temporally valid features, trains pitcher-specific models, evaluates completed games, tracks performance over time, and serves results through a Streamlit dashboard.

---

## Results

<!-- RESULTS:START -->

The headline metric is **relative improvement over baseline** — how much further the model gets than a stratified baseline drawing from the same pitcher's historical pitch mix:

```text
relative improvement = (model accuracy − baseline accuracy) / baseline accuracy
```

Results come from automated postgame replay of every eligible MLB starting pitcher, pitch-weighted across all pitcher-games in the window.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="Docs/assets/recent_performance_dark.png">
    <img alt="Relative improvement over baseline, last 14 days: +58.6% overall, shown as daily columns against the period average" src="Docs/assets/recent_performance_light.png" width="900">
  </picture>
</p>

**Trailing 14 days** · 4 evaluated game dates (August 29 – September 11, 2026) · 119 pitcher-games · 97 pitchers · 9,897 pitches

| Game date | Pitcher-games | Pitches | Relative improvement over baseline |
|---|---:|---:|---:|
| Aug 29 | 32 | 2,415 | +57.2% |
| Aug 30 | 27 | 2,455 | +54.2% |
| Sep 8 | 30 | 2,519 | +68.7% |
| Sep 11 | 30 | 2,508 | +75.0% |
| **14-day total** | **119** | **9,897** | **+63.7%** |

The model finished ahead of the baseline in 114 of 119 pitcher-games (96%).

<!-- RESULTS:END -->

---

## Project Summary

The goal is to predict a pitcher's next pitch using information available before the pitch is thrown.

Pitch selection is modeled as a multiclass classification problem:

```text
Game Context + Pitch History + Batter/Pitcher Information
                         ↓
                Predicted Pitch Type
```

The system is designed around individual pitchers because pitch repertoires and sequencing tendencies vary substantially across MLB pitchers.

The current production model is a pitcher-specific Random Forest trained on historical Statcast data and evaluated chronologically against future games.

---

## Original Research Results

The original Kevin Gausman experiment used approximately 25,000 career pitches and compared several models and feature-engineering stages.

| Dataset | Model | Test Accuracy |
|---|---|---:|
| KG1 | Stratified Baseline | 42.65% |
| KG1 | Random Forest | 57.72% |
| KG2 | Random Forest | 58.52% |
| KG3 | Random Forest | 58.74% |
| KG4 | Random Forest | 58.00% |
| KG4 | Logistic Regression | 55.43% |
| KG4 | Gradient Boosting | 58.11% |
| KG4 | Linear SVM | 55.63% |

Random Forest provided the strongest and most consistent performance and was selected as the primary model for the expanded system.

The production pipeline continues to evaluate performance game-by-game against a stratified baseline and records:

- model accuracy
- baseline accuracy
- correctly predicted pitches
- relative improvement over baseline
- cumulative pitcher and season performance

---

## End-to-End Workflow

```text
MLB Schedule API
        │
        ▼
Identify Starting Pitchers
        │
        ▼
Baseball Savant / Statcast
        │
        ▼
Download Historical Pitch Data
        │
        ▼
Schema Validation
        │
        ▼
Cleaning + Feature Engineering
KG1 → KG2 → KG3 → KG4
        │
        ▼
Pitcher-Specific Random Forest
        │
        ├───────────────────────────────┐
        ▼                               ▼
Postgame Replay                  Live Prediction
(Savant, completed games)        (GUMBO feed, in progress)
        │                               │
        ├── Model Prediction            ├── Next-Pitch Prediction
        ├── Stratified Baseline         ├── Actual Pitch
        └── Actual Pitch                └── Timing Audit
        │                               │
        ▼                               ▼
Performance History              Live Prediction Log
        │
        ▼
Streamlit Dashboard
```

### 1. Starter Identification
The MLB schedule API identifies probable or confirmed starting pitchers for a given date.

### 2. Data Retrieval
Career pitch history is retrieved dynamically from Baseball Savant for each pitcher.

### 3. Validation and Cleaning
Incoming Statcast exports are checked against canonical schemas before being cleaned and chronologically ordered.

### 4. Feature Engineering
Raw pitch data is transformed through successive KG feature stages, including previous-pitch information, count context, handedness, pitch sequencing, score context, and recent pitch usage.

### 5. Model Training
A separate Random Forest model is trained for each pitcher using only information available before the prediction date.

### 6. Postgame Evaluation
Completed games are replayed pitch-by-pitch using a frozen pregame model. The target game is excluded from training to prevent temporal leakage.

### 7. Performance Tracking
Pitcher-game results are written to a persistent performance history and displayed through an interactive dashboard.

### 8. Live Prediction
During a game in progress, features are rebuilt from the MLB GUMBO feed and the
frozen pre-game model predicts each pitch before it is thrown. See
[Live Game Prediction](#live-game-prediction).

---

## Architecture and Project Evolution

### Phase 1 — Notebook Prototype

The original project focused on Kevin Gausman and was developed primarily in Jupyter notebooks.

This stage included:

- exploratory data analysis
- data cleaning
- feature engineering experiments
- multiple model comparisons
- feature ablation
- evaluation of Random Forest, Logistic Regression, Gradient Boosting, and SVM models

### Phase 2 — End-to-End Pipeline

The notebook logic was converted into a reusable Python package capable of processing any MLB starting pitcher.

Major improvements include:

- dynamic starting-pitcher discovery
- automated Baseball Savant data retrieval
- strict schema validation
- reusable cleaning and feature-engineering modules
- pitcher-specific model training
- chronological rather than random evaluation
- prevention of target-game leakage
- recency-weighted training data
- repertoire-aware training weights
- stratified baseline comparison
- postgame game replay
- persistent performance history
- automated testing
- Streamlit performance dashboard

The notebooks remain in the repository as the original research and experimentation layer, while production logic now lives in the `pitch_prediction/` package.

---

## Live Game Prediction

The system can predict each pitch **before it is thrown** during a game in
progress, using MLB's GUMBO live feed
(`statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live`) in place of the Savant
export, and the same frozen pre-game model the postgame replay uses.

### Feature parity

The production models are trained on Savant's KG4 frame, so live prediction
requires rebuilding that frame from a different source. Measured across
**67 pitcher-games and 5,685 pitches** with the same frozen models scoring both
feature sets:

| Feature source | Accuracy | Relative improvement over baseline |
|---|---:|---:|
| Savant (postgame) | 40.2% | +60.3% |
| GUMBO (live) | 39.8% | +58.9% |

The two agree on 94% of pitches, and live was better or equal in 41 of 67
pitcher-games. **35 of 66 comparable columns match Savant exactly.** The rest:

- `sz_top` / `sz_bot` / `strike_zone_height` are measured on the pitch itself,
  so they cannot be known before it is thrown. They are constant per batter
  within a game, so only each batter's first plate appearance needs an
  estimate, drawn from a league-wide batter zone reference.
- `bat_win_exp` varies pitch by pitch in Savant, while the only live source
  publishes one value per plate appearance. Supplying that approximation
  measured *worse* than leaving the column missing for the model's imputer, so
  it is opt-in via `--with-win-probability`.
- `if_fielding_alignment` and `of_fielding_alignment` are absent from the feed.
  Their combined Random Forest importance is under 0.5%, and they are left
  missing.
- `launch_speed_angle` is not published live but is a deterministic function of
  exit velocity and launch angle, so it is recovered from
  `config/launch_speed_angle_grid.csv` at 99.8% fidelity.

### Predicting a pitch ahead

MLB publishes a pitch about **19 seconds** after it is thrown (median, measured
live at 0.5s polling), while pitches arrive about 20 seconds apart. A predictor
that waits for the previous pitch therefore has roughly a second of headroom
and misses about half the time. Polling faster does not help: published times
land on a ~10 second grid. Baseball Savant's own game feed is a further 19
seconds behind, so there is no faster public source.

The fix is to stop needing the data. Models trained by
`scripts.train_live_models` withhold the previous pitch's continuous
measurements, which leaves only *enumerable* unknowns: inside an at-bat the
next count has at most three outcomes, and the previous pitch type comes from
the pitcher's repertoire. The engine scores every candidate state before the
current pitch is thrown and selects the matching one when the feed catches up,
so the prediction carries the earlier timestamp.

```bash
python -m scripts.run_daily_pipeline --date 2026-09-08   # before first pitch
python -m scripts.train_live_models  --date 2026-09-08
```

Measured on a live game, same engine:

| | predicted before the pitch | median lead |
|---|---:|---:|
| production model (waits for the feed) | 6/12 (50%) | +1.2s |
| ahead-capable model | 9/10 (90%) | +12.1s |
| &nbsp;&nbsp;of those, computed ahead | | **+24.6s** |
| &nbsp;&nbsp;of those, not computed ahead | | +1.6s |

A candidate was waiting for **181 of 187 within-at-bat pitches (96.8%)** across
three replayed starts.

Plate-appearance endings are deliberately not enumerated. Measured over 1,164
pitches, gaps inside an at-bat are 15.5s median and only 26% beat the feed,
while the first pitch of a new at-bat is 31.2s median and 99% beat it: the
batter walking up already supplies the headroom, so enumerating outs, bases and
score would add real complexity for nothing.

The cost is 2.4 points of relative improvement over baseline (+62.6% to
+60.2% across 46 pitcher-games), which is not statistically distinguishable
from zero (paired p=0.15).

### Timing

A pitch reaches the feed a few seconds after it is thrown, and the next pitch
follows within roughly 15 to 25 seconds, so the engine must predict inside a
narrow window. Two properties make it robust rather than dependent on winning
a race:

- **Continuous re-prediction.** Whenever the observable state for the pending
  pitch changes, it is re-predicted and the prior prediction is superseded. A
  slow feed costs freshness, not the prediction.
- **Self-auditing.** Every prediction records the instant it was made, which is
  compared against the pitch's feed `startTime` and marked `before_pitch` or
  `late`. Accuracy can then be reported over only those predictions that
  provably preceded their pitch, so a latency problem surfaces as an honest
  number instead of an inflated one.

Replaying a full start through the feed's archived snapshots, whose ~15 second
cadence is slower than live polling:

```text
pitches scored          95
predicted before pitch  95 / 95  (100%)
lead over pitch         median +23.1s   minimum +8.7s
```

### Replay a completed game through the live code path

`?timecode=` returns the feed exactly as it existed at that moment, so a
finished game can be replayed as the sequence of states a live client would
have observed. This is how live behaviour is tested without waiting for a game:

```bash
python -m scripts.run_live_prediction \
    --game-pk 824802 --pitcher-id 680694 --replay
```

### Follow a game in progress

```bash
python -m scripts.run_live_prediction --game-pk <GAME_PK> --pitcher-id <MLBAM_ID>
```

Requires a frozen pre-game model for that date, so run
`scripts.run_daily_pipeline` first. Resolved predictions are appended to
`Data/daily_pipeline/predictions/live/<game_pk>_<pitcher_id>.jsonl`.

### Dashboard

```bash
streamlit run dashboard/live.py
```

Every completed start is replayed pitch by pitch through the model that was
frozen before that game, so the record is complete: every pitch has a
prediction, an actual, and a baseline.

**Games** leads with the day's accuracy and relative improvement over baseline,
above a scoreboard of that date's games. Each card carries the final score and
how each starter was predicted. Opening a game gives a tab per starter with
four figures (pitches, accuracy, baseline, relative improvement), a strip
showing the whole outing pitch by pitch, the pitcher's repertoire for that
start, and the full predicted-against-actual log.

### Why the dashboard replays rather than predicts live

The engine in `pitch_prediction/live/` does predict a pitch before it is
thrown, and the feature and timing work behind it is documented below. It is
not what the dashboard shows, because its coverage is too low to build a
product on.

Measured against the replay's complete pitch count -- the only honest
denominator, since it counts pitches that got no live prediction at all --
live mode reached **15% of pitches** across 30 pitcher-games on 2026-09-11.
One pitcher-game cleared 70%; the median was 14%. MLB publishes a pitch about
19 seconds after it is thrown while pitches arrive about 20 seconds apart, and
enumerating candidate states ahead of time narrows that gap without closing it.

Replaying afterwards covers every pitch, so that is what the dashboard is
built on. The live engine, its tests, and
`scripts/run_live_prediction.py` remain for future work.

### Audit live feature fidelity

```bash
python -m scripts.verify_live_features --date 2026-08-20
```

Rebuilds features from the feed for every pitcher-game with a frozen model,
reports per-column match rates against Savant, and prints the accuracy cost of
going live. Run this after any change under `pitch_prediction/live/`.

---

## Modeling

The production model uses a scikit-learn pipeline with:

- numeric missing-value imputation
- categorical imputation
- one-hot encoding
- Random Forest classification

Current Random Forest configuration:

```python
RandomForestClassifier(
    n_estimators=800,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=5,
    max_features="log2",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
```

### Chronological Evaluation

Games are ordered by date and split chronologically:

```text
Older Games → Training
Newest Games → Testing
```

This more closely represents the real prediction problem than randomly splitting individual pitches.

### Recency and Repertoire Weighting

Training observations are weighted so that:

- recent seasons have greater influence than older seasons
- pitches declining from a pitcher's repertoire receive less historical influence
- pitches becoming more prominent receive greater recent influence

Final sample weights combine both components:

```text
sample_weight = recency_weight × repertoire_weight
```

### Baseline

The model is compared against:

```python
DummyClassifier(
    strategy="stratified",
    random_state=42,
)
```

The baseline predicts according to the pitcher's historical pitch distribution without using game context.

---

## Repository Structure

```text
Predicting-Baseball-Pitches/
│
├── config/                    # Canonical Statcast schemas
├── dashboard/                 # Streamlit dashboards
│   ├── app.py                 # Original performance dashboard (superseded)
│   ├── components.py          # Dashboard markup
│   └── live.py                # Replay dashboard
├── Data/                      # Pipeline outputs and performance history
├── Notebooks/                 # Original research notebooks
├── pitch_prediction/          # Core production package
│   ├── clients.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── live/                  # Live-game prediction from the MLB GUMBO feed
│   │   ├── batter_zones.py
│   │   ├── context.py
│   │   ├── engine.py
│   │   ├── features.py
│   │   ├── gumbo.py
│   │   ├── mapping.py
│   │   └── verification.py
│   ├── model.py
│   ├── performance_history.py
│   ├── pipeline.py
│   ├── postgame_replay.py
│   ├── repertoire.py
│   └── schema.py
├── scripts/                   # Pipeline and experiment entry points
├── tests/                     # Automated tests
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11 recommended
- Internet connection for MLB and Baseball Savant data retrieval

Install dependencies from:

```bash
pip install -r requirements.txt
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/jonnylee777/Predicting-Baseball-Pitches.git
cd Predicting-Baseball-Pitches
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running it daily

```bash
./scripts/daily.sh
```

1. Replay yesterday's completed games, writing the pitch-by-pitch logs and the
   performance history the dashboard reads
2. Rebuild the README's results section
3. Prune models older than the retention window

**Nothing needs preparing beforehand.** The replayer downloads its own Statcast
data, engineers the features, and trains its own pre-game model for the date
when none is frozen -- on pitches from before that game only, so it stays
leakage-free. Run it any time after the previous day's games have finished; a
full slate takes about three minutes. Logs land in `Data/daily_pipeline/logs/`.

Schedule it for the morning with cron -- `crontab -e`, then:

```cron
37 8 * * * cd /path/to/repo && PYTHON=/path/to/venv/bin/python ./scripts/daily.sh >> /path/to/repo/Data/daily_pipeline/logs/cron.log 2>&1
```

Morning is after even the latest west-coast finish. Set `PYTHON` explicitly --
cron's `PATH` will not find a virtualenv -- and `KEEP_DAYS` to retain models
longer than the default three days. On macOS, `cron` may need adding under
System Settings -> Privacy & Security -> Full Disk Access; an empty `cron.log`
the next morning is the symptom.

### What is kept, and what is reclaimed

Pitch-by-pitch game logs are **kept indefinitely** -- 318 games occupy 6.8 MB,
roughly 100 MB across a season -- so the full history stays queryable. Frozen
models are the only large artefact, at about 18 MB each compressed, and are
only needed until their date has been evaluated, so `scripts.prune_models`
retains a rolling window:

```bash
python -m scripts.prune_models --keep-days 3 --dry-run
```

It refuses to delete a date with no evaluated results unless given `--force`.
Note that `scripts.verify_live_features` and the ablations score against the
frozen models for the dates they evaluate, so pruning a date removes it as a
test fixture; re-running the pipeline for that date rebuilds them.

---

## How to Run

### Run the starting-pitcher data pipeline

```bash
python -m scripts.run_daily_pipeline --date 2026-08-20
```

### Evaluate all eligible starters from a completed game date

```bash
python -m scripts.run_daily_postgame_replay --date 2026-08-20
```

If no date is provided, the postgame pipeline defaults to the previous day.

### Evaluate a single pitcher

```bash
python -m scripts.run_postgame_replay \
    --date 2026-08-20 \
    --pitcher-id <MLBAM_ID>
```

### Launch the dashboard

```bash
streamlit run dashboard/live.py
```

`dashboard/app.py` is the original performance-history dashboard. It still
works and is kept for reference, but `live.py` supersedes it.

### Regenerate the README results section

```bash
python -m scripts.build_readme_results
```

Rebuilds the graphic and the table in **Results** from the current performance
history. Use `--window-days` to change the reporting window.

### Follow or replay a game from the command line

```bash
python -m scripts.run_live_prediction --game-pk <GAME_PK> --pitcher-id <MLBAM_ID>
python -m scripts.run_live_prediction --game-pk 824802 --pitcher-id 680694 --replay
```

### Audit the live feature path

```bash
python -m scripts.verify_live_features --date 2026-08-20
```

### Run the test suite

```bash
python -m pytest
```

---

## Performance History

Game-level evaluation results are stored in:

```text
Data/daily_pipeline/performance_history.csv
```

Each row represents one pitcher-game.

The unique key is:

```text
(game_pk, pitcher_id)
```

Re-running an evaluation replaces the existing record rather than creating a duplicate.

Detailed postgame prediction logs are also saved for pitch-level analysis.

---

## Testing and Reliability

The automated test suite covers key production behavior including:

- Statcast schema validation
- data cleaning
- feature engineering
- chronological splitting
- prediction-feature exclusion
- model persistence
- recency weighting
- repertoire weighting
- postgame replay
- target-game leakage prevention

The pipeline is designed to fail explicitly when upstream data schemas change rather than silently training on incompatible data.

---

## Current Status

The project currently supports automated data collection, pitcher-specific model training, historical postgame evaluation, persistent performance tracking, dashboard reporting, and live in-game prediction from the MLB GUMBO feed.

Both evaluation paths are available: postgame replay of completed games against
Savant data, and live prediction of each pitch before it is thrown. The live
path costs about 1.4 points of relative improvement against the postgame path
and is verified offline by replaying completed games through the live code.

Live prediction currently covers **starting pitchers only**, because models are
trained per starter. Once a starter is relieved there is no model for the
pitcher on the mound.

---

## Future Work

Planned improvements include:
- live prediction for relief pitchers
- model and feature version tracking
- larger historical backtesting
- season-over-season evaluation
- additional tree-based models such as XGBoost and CatBoost
- probability calibration and model confidence analysis