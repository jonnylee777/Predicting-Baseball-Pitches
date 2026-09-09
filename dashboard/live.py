"""MLB pitch prediction dashboard.

The date chosen in the sidebar decides what the page shows, because only one
thing is meaningful for any given date:

* **Today** -- a scoreboard of today's games. A game that is underway can be
  followed, predicting each pitch before it is thrown.
* **A past date** -- the completed record: every evaluated pitcher-game for
  that date, and for each one a pitch-by-pitch log of predicted against actual
  pitch, the pitcher's repertoire for that start, accuracy, and relative
  improvement over baseline.

Past dates are read straight from the postgame evaluation logs the pipeline
already writes, so they load instantly and involve no live feed at all.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

# ``streamlit run`` puts this file's directory on sys.path, not the repository
# root, so the project packages are not importable by default. This must run
# before the first project import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components import (
    THEME_CSS,
    GameView,
    game_card_html,
    hero_metric_html,
    pitch_table_html,
    prediction_html,
    repertoire_table_html,
    scoreboard_html,
    tile_html,
)
from pitch_prediction.live.batter_zones import load_or_build
from pitch_prediction.live.engine import (
    LivePredictionEngine,
    PitchPrediction,
    build_context,
    load_pregame_model,
)
from pitch_prediction.live.feature_sets import PREDICT_AHEAD_EXCLUSIONS
from pitch_prediction.live.gumbo import GumboClient
from pitch_prediction.model import PitchModelTrainer

DATA_ROOT = PROJECT_ROOT / "Data" / "daily_pipeline"
PERFORMANCE_HISTORY = DATA_ROOT / "performance_history.csv"
POSTGAME_ROOT = DATA_ROOT / "predictions" / "postgame"

st.set_page_config(page_title="MLB Pitch Prediction", page_icon="⚾", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    """<style>
    .block-container { padding-top: 2.2rem; max-width: 1500px; }
    [data-testid="stSidebar"] { min-width: 260px; max-width: 260px; }
    </style>""",
    unsafe_allow_html=True,
)


def theme_attribute() -> str:
    """Stamp the viewer's theme so it beats the OS setting in both directions."""

    try:
        base = st.get_option("theme.base")
    except Exception:  # noqa: BLE001 - option absent on some versions
        base = "light"
    return ' data-theme="dark"' if base == "dark" else ' data-theme="light"'


THEME = theme_attribute()


def block(html: str) -> None:
    st.markdown(f'<div class="viz"{THEME}>{html}</div>', unsafe_allow_html=True)


def rerun() -> None:
    (getattr(st, "rerun", None) or st.experimental_rerun)()


def row_of(items: list[str]) -> None:
    for column, html in zip(st.columns(len(items)), items):
        with column:
            block(html)


# ============================================================
# COMPLETED RESULTS
# ============================================================


@st.cache_data(show_spinner=False)
def load_performance_history() -> pd.DataFrame:
    if not PERFORMANCE_HISTORY.exists():
        return pd.DataFrame()
    frame = pd.read_csv(PERFORMANCE_HISTORY)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame


def evaluated_dates() -> list[dt.date]:
    history = load_performance_history()
    if history.empty:
        return []
    return sorted({d.date() for d in history["game_date"].dropna()}, reverse=True)


@st.cache_data(show_spinner=False)
def load_pitch_log(predictions_path: str) -> pd.DataFrame:
    path = PROJECT_ROOT / predictions_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def repertoire_rows(log: pd.DataFrame) -> list[dict]:
    """Usage share per pitch type, with the model's recall on each."""

    if log.empty:
        return []
    total = len(log)
    rows = []
    for pitch, group in log.groupby("actual_pitch"):
        rows.append(
            {
                "pitch": str(pitch),
                "count": int(len(group)),
                "thrown": len(group) / total,
                "recall": float(group["model_correct"].mean()),
            }
        )
    return rows


def render_pitcher_game(record: pd.Series) -> None:
    log = load_pitch_log(record["predictions_path"])

    relative = record.get("relative_improvement")
    row_of(
        [
            tile_html("Pitches", f"{int(record['pitch_count'])}",
                      f"{int(record['model_correct'])} predicted correctly"),
            tile_html("Accuracy", f"{record['model_accuracy']:.1%}",
                      "predicted type matched"),
            tile_html("Baseline", f"{record['baseline_accuracy']:.1%}",
                      "stratified from pre-game mix"),
            tile_html(
                "Relative improvement",
                f"{relative:+.1%}" if pd.notna(relative) else "--",
                "over that baseline",
            ),
        ]
    )
    st.write("")

    if log.empty:
        st.warning(
            "The pitch-by-pitch log for this start is not on disk at "
            f"`{record['predictions_path']}`."
        )
        return

    # "count" collides with the namedtuple method of the same name, so the
    # column is renamed before iterating.
    recent = log.iloc[::-1].rename(columns={"count": "count_text"})

    left, right = st.columns([1, 1.25])
    with left:
        block(repertoire_table_html(repertoire_rows(log)))
    with right:
        block(
            pitch_table_html(
                [
                    {
                        "inning": f"{r.inning_topbot} {int(r.inning)}",
                        "count": r.count_text,
                        "predicted": r.model_prediction,
                        "actual": r.actual_pitch,
                        "correct": bool(r.model_correct),
                    }
                    for r in recent.itertuples()
                ],
                limit=10,
            )
        )

    st.markdown("###### Full pitch log")
    table = pd.DataFrame(
        {
            "Pitch": log["pitch_number_of_game"],
            "Inn": log["inning_topbot"].str.cat(log["inning"].astype(str), sep=" "),
            "Count": log["count"],
            "Predicted": log["model_prediction"],
            "Actual": log["actual_pitch"],
            "Result": log["model_correct"].map({True: "correct", False: "missed"}),
            "Confidence": log["model_confidence"],
            "Baseline": log["baseline_prediction"],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=380,
        column_config={
            "Confidence": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )


def render_results(game_date: dt.date) -> None:
    history = load_performance_history()
    day = history[history["game_date"].dt.date == game_date].copy()

    if day.empty:
        st.info(
            f"No evaluated results for {game_date:%B %-d, %Y}. Generate them with "
            f"`python -m scripts.run_daily_postgame_replay --date {game_date}`."
        )
        return

    pitches = int(day["pitch_count"].sum())
    accuracy = day["model_correct"].sum() / pitches
    baseline = day["baseline_correct"].sum() / pitches
    st.markdown(
        f"#### {game_date:%A, %B %-d, %Y} &middot; {len(day)} pitcher-games "
        f"&middot; {pitches:,} pitches"
    )
    row_of(
        [
            hero_metric_html("Accuracy", f"{accuracy:.1%}",
                             f"baseline {baseline:.1%}"),
            hero_metric_html(
                "Relative improvement over baseline",
                f"{(accuracy - baseline) / baseline:+.1%}",
                f"{len(day)} starts evaluated",
            ),
        ]
    )
    st.write("")

    day = day.sort_values("relative_improvement", ascending=False)
    summary = pd.DataFrame(
        {
            "Pitcher": day["pitcher_name"],
            "Opponent": day["opponent"],
            "Pitches": day["pitch_count"],
            "Accuracy": day["model_accuracy"],
            "Baseline": day["baseline_accuracy"],
            "Relative improvement": day["relative_improvement"],
        }
    )
    st.markdown("###### Every start evaluated this date")
    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
            "Baseline": st.column_config.NumberColumn(format="%.1f%%"),
            "Relative improvement": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

    st.markdown("---")
    labels = {
        f"{row.pitcher_name} — vs {row.opponent} ({int(row.pitch_count)} pitches)":
            index
        for index, row in day.iterrows()
    }
    choice = st.selectbox("Pitcher game log", list(labels))
    st.markdown(f"#### {choice.split(' — ')[0]}")
    render_pitcher_game(day.loc[labels[choice]])


# ============================================================
# TODAY
# ============================================================


@st.cache_resource(show_spinner=False)
def get_client() -> GumboClient:
    return GumboClient()


@st.cache_data(show_spinner=False, ttl=30)
def load_schedule(game_date: str) -> list[dict]:
    return get_client().schedule(game_date)


@st.cache_data(show_spinner=False, ttl=15)
def load_payload(game_pk: int) -> dict:
    return get_client().feed_live(game_pk).payload


@st.cache_data(show_spinner="Loading batter strike-zone reference...")
def load_zone_reference(game_date: str) -> dict[int, tuple[float, float]]:
    return load_or_build(
        DATA_ROOT / "features" / "kg4" / "pitchers",
        through_date=dt.date.fromisoformat(game_date),
        cache_dir=DATA_ROOT / "reference",
    )


@st.cache_resource(show_spinner="Loading pre-game model...")
def load_engine_parts(pitcher_id: int, game_date: str):
    model, is_live_model = load_pregame_model(DATA_ROOT, game_date, pitcher_id)
    context = build_context(
        DATA_ROOT,
        pitcher_id,
        dt.date.fromisoformat(game_date),
        batter_zone_reference=load_zone_reference(game_date),
    )
    return model, context, is_live_model


def has_model(pitcher_id: int, game_date: str) -> bool:
    return (
        DATA_ROOT / "models" / game_date / "pitchers" / f"{pitcher_id}.joblib"
    ).exists()


def game_status(game: dict) -> tuple[str, bool, str]:
    state = game["status"]["abstractGameState"]
    linescore = game.get("linescore") or {}
    if state == "Live":
        half = linescore.get("inningHalf", "")
        inning = linescore.get("currentInning", "")
        return (
            f"{half} {inning}".strip(),
            True,
            f"{linescore.get('balls', 0)}-{linescore.get('strikes', 0)}, "
            f"{linescore.get('outs', 0)} out",
        )
    if state == "Final":
        return "Final", False, ""
    start = pd.to_datetime(game["gameDate"]).tz_convert("America/New_York")
    return game["status"]["detailedState"], False, f"{start:%-I:%M %p} ET"


def _open_game(game_pk: int, date_text: str) -> None:
    st.session_state["game_pk"] = game_pk
    st.session_state["game_date"] = date_text


def _close_game() -> None:
    st.session_state.pop("game_pk", None)


def render_today(game_date: dt.date) -> None:
    date_text = game_date.isoformat()
    games = load_schedule(date_text)
    st.markdown(f"#### Games &middot; {game_date:%A, %B %-d, %Y}")
    if not games:
        st.info("No games scheduled on this date.")
        return

    per_row = 3
    for start in range(0, len(games), per_row):
        columns = st.columns(per_row)
        for column, game in zip(columns, games[start : start + per_row]):
            game_pk = int(game["gamePk"])
            status, is_live, detail = game_status(game)
            starters = {
                game["teams"][side]["probablePitcher"]["fullName"]:
                    int(game["teams"][side]["probablePitcher"]["id"])
                for side in ("away", "home")
                if game["teams"][side].get("probablePitcher")
            }
            eligible = {n: p for n, p in starters.items() if has_model(p, date_text)}
            note = None if eligible else "No pre-game model — run the daily pipeline"

            with column:
                block(
                    game_card_html(
                        away=game["teams"]["away"]["team"]["abbreviation"],
                        home=game["teams"]["home"]["team"]["abbreviation"],
                        away_score=game["teams"]["away"].get("score"),
                        home_score=game["teams"]["home"].get("score"),
                        status=status,
                        is_live=is_live,
                        detail=detail,
                        pitchers=list(starters),
                        note=note,
                    )
                )
                st.button(
                    "Follow live" if is_live else "Open",
                    key=f"open_{game_pk}",
                    disabled=not eligible,
                    use_container_width=True,
                    on_click=_open_game,
                    args=(game_pk, date_text),
                )
        st.write("")


def read_game_view(snapshot, pending: PitchPrediction | None) -> GameView:
    home, away = snapshot.team_abbreviations
    linescore = snapshot.payload.get("liveData", {}).get("linescore", {})
    teams = linescore.get("teams", {})
    offense = linescore.get("offense", {})
    matchup = (snapshot.current_play or {}).get("matchup", {})
    return GameView(
        away=away or "AWY",
        home=home or "HOM",
        away_score=int(teams.get("away", {}).get("runs") or 0),
        home_score=int(teams.get("home", {}).get("runs") or 0),
        inning=int(linescore.get("currentInning") or 0),
        topbot="Top" if linescore.get("isTopInning") else "Bot",
        balls=pending.balls if pending else int(linescore.get("balls") or 0),
        strikes=pending.strikes if pending else int(linescore.get("strikes") or 0),
        outs=pending.outs if pending else int(linescore.get("outs") or 0),
        status=snapshot.detailed_state,
        on_1b=bool(offense.get("first") or matchup.get("postOnFirst")),
        on_2b=bool(offense.get("second") or matchup.get("postOnSecond")),
        on_3b=bool(offense.get("third") or matchup.get("postOnThird")),
    )


def engine_for(pitcher_id: int, name: str, game_pk: int, date_text: str):
    key = f"engine_{game_pk}_{pitcher_id}"
    if key not in st.session_state:
        model, context, is_live_model = load_engine_parts(pitcher_id, date_text)
        st.session_state[key] = LivePredictionEngine(
            model=model,
            context=context,
            pitcher_name=name,
            trainer=PitchModelTrainer(
                exclude_features=(
                    PREDICT_AHEAD_EXCLUSIONS if is_live_model else ()
                )
            ),
            log_path=DATA_ROOT / "predictions" / "live"
            / f"{game_pk}_{pitcher_id}.jsonl",
        )
    return st.session_state[key]


def live_pitcher_panel(engine: LivePredictionEngine) -> None:
    pending = engine.pending
    left, right = st.columns([1.1, 1])
    with left:
        block(
            prediction_html(
                predicted=pending.predicted_pitch_type if pending else None,
                confidence=pending.confidence if pending else None,
                probabilities=pending.probabilities if pending else None,
                batter=pending.batter_name if pending else None,
                balls=pending.balls if pending else None,
                strikes=pending.strikes if pending else None,
                pitch_number=pending.pitch_number_of_game if pending else None,
            )
        )
    with right:
        block(
            pitch_table_html(
                [
                    {
                        "inning": f"{p.inning_topbot} {p.inning}",
                        "count": f"{p.balls}-{p.strikes}",
                        "predicted": p.predicted_pitch_type,
                        "actual": p.actual_pitch_type,
                        "correct": p.correct,
                    }
                    for p in reversed(engine.history)
                ]
            )
        )

    stats = engine.stats
    actual = [p.actual_pitch_type for p in engine.history if p.actual_pitch_type]
    baseline = engine.context.expected_baseline_accuracy(actual)
    relative = (
        (stats.accuracy - baseline) / baseline
        if baseline and stats.accuracy is not None and baseline > 0
        else None
    )
    row_of(
        [
            tile_html("Pitches scored", str(stats.resolved), f"{stats.correct} correct"),
            tile_html(
                "Accuracy",
                f"{stats.accuracy:.1%}" if stats.accuracy is not None else "--",
                "predicted type matched",
            ),
            tile_html(
                "Relative improvement",
                f"{relative:+.1%}" if relative is not None else "--",
                f"baseline {baseline:.1%}" if baseline else "over stratified baseline",
            ),
            tile_html(
                "Predicted before pitch",
                f"{stats.before_pitch}/{stats.resolved}" if stats.resolved else "--",
                f"{stats.late} late",
            ),
        ]
    )


def render_live_game(game_pk: int, date_text: str) -> None:
    payload = load_payload(game_pk)
    teams = payload.get("gameData", {}).get("teams", {})
    state = payload.get("gameData", {}).get("status", {}).get("abstractGameState")
    st.button("← All games", on_click=_close_game)
    st.markdown(
        f"### {teams.get('away', {}).get('abbreviation', '')} @ "
        f"{teams.get('home', {}).get('abbreviation', '')}"
    )

    probables = payload.get("gameData", {}).get("probablePitchers", {})
    starters = {
        probables[side]["fullName"]: int(probables[side]["id"])
        for side in ("away", "home")
        if probables.get(side)
    }
    eligible = {n: p for n, p in starters.items() if has_model(p, date_text)}
    if not eligible:
        st.error(
            "No frozen pre-game model for either starter. Run "
            f"`python -m scripts.run_daily_pipeline --date {date_text}` before "
            "the game starts."
        )
        return

    if state == "Preview":
        st.info(
            "This game has not started. Predictions begin with the first pitch."
        )
        return
    if state == "Final":
        st.info(
            "This game is over. Its pitch-by-pitch results appear under its date "
            "once the postgame replay has run:\n\n"
            f"`python -m scripts.run_daily_postgame_replay --date {date_text}`"
        )
        return

    interval = st.sidebar.slider("Refresh (seconds)", 2, 15, 3)
    snapshot = get_client().feed_live(game_pk)
    engines = {
        name: engine_for(pid, name, game_pk, date_text)
        for name, pid in eligible.items()
    }
    for engine in engines.values():
        engine.observe(snapshot)

    first = next(iter(engines.values()))
    block(scoreboard_html(read_game_view(snapshot, first.pending)))
    st.write("")

    names = list(engines)
    if len(names) > 1:
        for tab, name in zip(st.tabs(names), names):
            with tab:
                live_pitcher_panel(engines[name])
    else:
        st.markdown(f"**{names[0]}**")
        live_pitcher_panel(engines[names[0]])

    st.caption(
        "Each prediction is timestamped and compared against its pitch's feed "
        "start time, so a prediction that arrives late is recorded as late "
        "rather than counted."
    )
    time.sleep(interval)
    rerun()


# ============================================================
# APP
# ============================================================

TODAY = dt.date.today()

st.sidebar.title("MLB pitch prediction")
options = {f"Today — {TODAY:%B %-d, %Y}": TODAY}
for day in evaluated_dates():
    if day != TODAY:
        options[f"{day:%B %-d, %Y}"] = day
selection = st.sidebar.selectbox("Date", list(options))
selected_date = options[selection]

st.sidebar.caption(
    "Today shows the live scoreboard; games underway can be followed pitch by "
    "pitch. Past dates show the completed pitch-by-pitch record."
)

st.title("MLB pitch prediction")
st.caption(
    "Predicting the next pitch type for MLB starting pitchers, before it is thrown."
)

history = load_performance_history()
if history.empty:
    row_of([hero_metric_html("Pitch prediction accuracy", "--", "no history yet"),
            hero_metric_html("Relative improvement over baseline", "--", "")])
else:
    pitches = int(history["pitch_count"].sum())
    accuracy = history["model_correct"].sum() / pitches
    baseline = history["baseline_correct"].sum() / pitches
    row_of(
        [
            hero_metric_html(
                "Pitch prediction accuracy",
                f"{accuracy:.1%}",
                f"{pitches:,} pitches across {len(history)} evaluated starts",
            ),
            hero_metric_html(
                "Relative improvement over baseline",
                f"{(accuracy - baseline) / baseline:+.1%}",
                f"baseline {baseline:.1%} &middot; "
                f"{history['game_date'].min():%b %-d} to "
                f"{history['game_date'].max():%b %-d, %Y}",
            ),
        ]
    )
st.write("")

if selected_date >= TODAY:
    if "game_pk" in st.session_state:
        render_live_game(int(st.session_state["game_pk"]),
                         st.session_state["game_date"])
    else:
        render_today(selected_date)
else:
    st.session_state.pop("game_pk", None)
    render_results(selected_date)
