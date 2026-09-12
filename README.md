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
    <img alt="Relative improvement over baseline, last 30 days: +60.5% overall across 28,843 pitches, shown as one column per evaluated game date against the period average" src="Docs/assets/recent_performance_light.png" width="900">
  </picture>
</p>

**Trailing 30 days** · 13 evaluated game dates (August 18 – September 11, 2026) · 337 pitcher-games · 160 pitchers · 28,843 pitches

| Game date | Pitcher-games | Pitches | Relative improvement over baseline |
|---|---:|---:|---:|
| Aug 18 | 22 | 1,774 | +55.3% |
| Aug 19 | 28 | 2,450 | +59.7% |
| Aug 20 | 18 | 1,641 | +66.8% |
| Aug 22 | 29 | 2,632 | +63.9% |
| Aug 23 | 29 | 2,290 | +58.2% |
| Aug 24 | 20 | 1,841 | +58.2% |
| Aug 25 | 28 | 2,421 | +49.5% |
| Aug 27 | 14 | 1,206 | +61.0% |
| Aug 28 | 30 | 2,691 | +59.1% |
| Aug 29 | 32 | 2,415 | +57.2% |
| Aug 30 | 27 | 2,455 | +54.2% |
| Sep 8 | 30 | 2,519 | +68.7% |
| Sep 11 | 30 | 2,508 | +75.0% |
| **30-day total** | **337** | **28,843** | **+60.5%** |

The model finished ahead of the baseline in 329 of 337 pitcher-games (98%).

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
        ▼
Postgame Replay
        │
        ├── Model Prediction
        ├── Stratified Baseline
        └── Actual Pitch
        │
        ▼
Performance History
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

## Dashboard

```bash
streamlit run dashboard/replay.py
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

An engine that predicted pitches live, during a game, was built and measured.
It is not part of the project, because its coverage was too low to build on.

Measured against the replay's complete pitch count — the only honest
denominator, since it counts pitches that got no live prediction at all --
live mode reached **15% of pitches** across 30 pitcher-games on 2026-09-11.
One pitcher-game cleared 70%; the median was 14%. MLB publishes a pitch about
19 seconds after it is thrown while pitches arrive about 20 seconds apart, and
enumerating candidate states ahead of time narrows that gap without closing it.

Replaying afterwards covers every pitch, so that is what the dashboard is
built on. The live code was removed; this note records why, so the question
does not get reopened without new evidence.

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
├── .github/workflows/         # Daily postgame replay, scheduled
├── config/                    # Canonical Statcast schemas
├── dashboard/                 # Streamlit replay dashboard
│   ├── components.py          # Markup
│   └── replay.py              # The app
├── Data/                      # Pipeline outputs and performance history
├── Notebooks/                 # Original research notebooks
├── pitch_prediction/          # Core production package
│   ├── clients.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── performance_history.py
│   ├── pipeline.py
│   ├── postgame_replay.py
│   ├── repertoire.py
│   └── schema.py
├── scripts/                   # Entry points and the daily driver
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

This runs itself. `.github/workflows/daily-postgame-replay.yml` fires at 15:00
UTC each day from March to November, which gives Statcast time to publish the
previous day's data. It installs dependencies, **runs the test suite**, replays
every completed start from yesterday, and commits the updated
`performance_history.csv` and pitch-by-pitch logs back to the repository. So
the dashboard's data arrives through git, and a fresh clone has the full
history.

Gating on tests means a regression stops the day's results being written rather
than quietly corrupting them.

To backfill or re-run a specific date, trigger the workflow manually from the
Actions tab with a `game_date` input.

### Running it by hand

`scripts/daily.sh` does the same work locally, for when you want a date now
rather than at the next scheduled run:

```bash
./scripts/daily.sh                    # yesterday
KEEP_DAYS=7 ./scripts/daily.sh        # keep models a week instead of three days
```

1. Replay yesterday's completed games
2. Rebuild the README's results section
3. Prune models older than the retention window

**Nothing needs preparing beforehand.** The replayer downloads its own Statcast
data, engineers the features, and trains its own pre-game model for the date
when none is frozen — on pitches from before that game only, so it stays
leakage-free. A full slate takes about three minutes. Logs land in
`Data/daily_pipeline/logs/`.

For a single date without the surrounding steps:

```bash
python -m scripts.run_daily_postgame_replay --date 2026-09-12
```

### What is kept, and what is reclaimed

Pitch-by-pitch game logs are **kept indefinitely** — 318 games occupy 6.8 MB,
roughly 100 MB across a season — so the full history stays queryable. Frozen
models are the only large artefact, at about 18 MB each compressed, and are
only needed until their date has been evaluated, so `scripts.prune_models`
retains a rolling window:

```bash
python -m scripts.prune_models --keep-days 3 --dry-run
```

It refuses to delete a date with no evaluated results unless given `--force`.

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
streamlit run dashboard/replay.py
```



### Regenerate the README results section

```bash
python -m scripts.build_readme_results
```

Rebuilds the graphic and the table in **Results** from the current performance
history. Use `--window-days` to change the reporting window.

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

The project automates the full loop: discovering each day's starters,
retrieving their Statcast history, validating and engineering it, training a
model per pitcher, replaying every completed start pitch by pitch, and
publishing the results to a dashboard. A GitHub Action runs it daily and
commits the results, so the record grows without intervention.

Evaluation is **postgame replay**: each pitcher-game is scored with a model
frozen on data from before that game, which is what makes the accuracy figures
honest rather than hindsight. Coverage is **starting pitchers only**, because
models are trained per pitcher and relievers throw too few pitches to support
one.

---

## Future Work

- relief pitchers, most likely via a pooled model rather than one each
- model and feature version tracking
- larger historical backtesting and season-over-season evaluation
- additional tree-based models such as XGBoost and CatBoost
- probability calibration and model confidence analysis
