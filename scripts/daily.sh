#!/bin/zsh
# Everything the project needs each day, in dependency order.
#
# Run this in the morning, before the day's first pitch. Two ordering
# constraints matter:
#
#   * The pipeline and live-model training must finish BEFORE first pitch.
#     Both train on data up to yesterday, so running them late would not leak
#     future data, but a game already underway cannot be followed from its
#     first pitch.
#   * The postgame replay for yesterday runs first, because pruning decides
#     what to delete from whether a date has evaluated results.
#
# Schedule with cron (see the README) or launchd. Logs land in
# Data/daily_pipeline/logs/.

set -u
cd "$(dirname "$0")/.." || exit 1

TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d)
LOG_DIR="Data/daily_pipeline/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily_${TODAY}.log"

PYTHON=${PYTHON:-python}
# Days of frozen models to retain. Raise it if you need older dates as
# fixtures for scripts.verify_live_features or the ablations, which score
# against the models for the dates they evaluate.
KEEP_DAYS=${KEEP_DAYS:-3}

step() {
  echo "" | tee -a "$LOG"
  echo "=== $1 === $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
  shift
  if "$@" >>"$LOG" 2>&1; then
    echo "    ok" | tee -a "$LOG"
  else
    echo "    FAILED (exit $?) — continuing" | tee -a "$LOG"
  fi
}

echo "daily run $TODAY" | tee "$LOG"

# 1. Evaluate yesterday's completed games. This writes the pitch-by-pitch
#    logs and performance history the dashboard's history and leaderboard
#    read, so it comes before pruning.
step "postgame replay for $YESTERDAY" \
  $PYTHON -m scripts.run_daily_postgame_replay --date "$YESTERDAY"

# 2. Today's starters: download history, engineer features, train the
#    production models.
step "daily pipeline for $TODAY" \
  $PYTHON -m scripts.run_daily_pipeline --date "$TODAY"

# 3. The live models, which withhold the columns that arrive too late to
#    predict ahead of a pitch.
step "live models for $TODAY" \
  $PYTHON -m scripts.train_live_models --date "$TODAY"

# 4. Refresh the README's results section from the new history.
step "rebuild README results" \
  $PYTHON -m scripts.build_readme_results

# 5. Reclaim disk. Models are the only large artefact and are only needed
#    until their date has been evaluated; pitch-by-pitch logs are tiny and
#    are kept forever.
step "prune old models (keeping $KEEP_DAYS days)" \
  $PYTHON -m scripts.prune_models --keep-days "$KEEP_DAYS"

echo "" | tee -a "$LOG"
echo "done $(date -u +%H:%M:%SZ) — log: $LOG" | tee -a "$LOG"
