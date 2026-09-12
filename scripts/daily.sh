#!/bin/zsh
# Everything the project needs each day.
#
# Evaluates yesterday's completed games and puts them in the dashboard. The
# replayer downloads its own data and trains its own pre-game model for the
# date when none is frozen, using only pitches from before that game, so this
# is the only step required -- there is nothing to prepare beforehand.
#
# Run it any time after the previous day's games have finished. Morning is the
# natural slot. Logs land in Data/daily_pipeline/logs/.

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
#    logs and the performance history the dashboard reads, so it comes before
#    pruning, which decides what to delete from whether a date was evaluated.
step "postgame replay for $YESTERDAY" \
  $PYTHON -m scripts.run_daily_postgame_replay --date "$YESTERDAY"

# 2. Refresh the README's results section from the new history.
step "rebuild README results" \
  $PYTHON -m scripts.build_readme_results

# 3. Reclaim disk. Models are the only large artefact and are only needed
#    until their date has been evaluated. Pitch-by-pitch logs are about half a
#    megabyte a day and are kept indefinitely.
step "prune old models (keeping $KEEP_DAYS days)" \
  $PYTHON -m scripts.prune_models --keep-days "$KEEP_DAYS"

echo "" | tee -a "$LOG"
echo "done $(date -u +%H:%M:%SZ) — log: $LOG" | tee -a "$LOG"
