"""Live pitch prediction dashboard.

Two views:

* **Scores** -- the season's headline accuracy and relative improvement over
  baseline, above a scoreboard of a date's games. Selecting a game opens it.
* **Game** -- a Gameday-style view of one game. For each starting pitcher with
  a frozen pre-game model, the pitch about to be thrown is predicted and shown
  beside a table of the pitches actually thrown, with a running log.

The prediction work belongs to ``pitch_prediction.live.engine``; this module
drives it and renders the result. Every resolved prediction is also appended to
a JSONL log by the engine, so closing the browser loses the view, not the
record.

Refreshing reruns the whole script on a timer rather than using fragments,
which do not exist before Streamlit 1.33. The engines live in session state, so
no prediction history is lost across reruns.
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
    scoreboard_html,
    tile_html,
)
from pitch_prediction.live.batter_zones import load_or_build
from pitch_prediction.live.engine import (
    LivePredictionEngine,
    PitchPrediction,
    build_context,
    load_pregame_model,
    parse_timecode,
)
from pitch_prediction.live.gumbo import GumboClient

DATA_ROOT = PROJECT_ROOT / "Data" / "daily_pipeline"
PERFORMANCE_HISTORY = DATA_ROOT / "performance_history.csv"

st.set_page_config(
    page_title="Live Pitch Prediction",
    page_icon="⚾",
    layout="wide",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    """<style>
    .block-container { padding-top: 2.2rem; max-width: 1500px; }
    [data-testid="stSidebar"] { min-width: 250px; max-width: 250px; }
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


# ============================================================
# DATA
# ============================================================


@st.cache_resource(show_spinner=False)
def get_client() -> GumboClient:
    return GumboClient()


@st.cache_data(show_spinner=False, ttl=30)
def load_schedule(game_date: str) -> list[dict]:
    return get_client().schedule(game_date)


@st.cache_data(show_spinner=False, ttl=20)
def load_snapshot_payload(game_pk: int, timecode: str | None = None) -> dict:
    return get_client().feed_live(game_pk, timecode=timecode).payload


@st.cache_data(show_spinner=False)
def load_timestamps(game_pk: int) -> list[str]:
    return get_client().timestamps(game_pk)


@st.cache_data(show_spinner="Loading batter strike-zone reference...")
def load_zone_reference(game_date: str) -> dict[int, tuple[float, float]]:
    return load_or_build(
        DATA_ROOT / "features" / "kg4" / "pitchers",
        through_date=dt.date.fromisoformat(game_date),
        cache_dir=DATA_ROOT / "reference",
    )


@st.cache_data(show_spinner=False)
def load_performance_history() -> pd.DataFrame:
    if not PERFORMANCE_HISTORY.exists():
        return pd.DataFrame()
    frame = pd.read_csv(PERFORMANCE_HISTORY)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame


@st.cache_resource(show_spinner="Loading pre-game model...")
def load_engine_parts(pitcher_id: int, game_date: str):
    model = load_pregame_model(DATA_ROOT, game_date, pitcher_id)
    context = build_context(
        DATA_ROOT,
        pitcher_id,
        dt.date.fromisoformat(game_date),
        batter_zone_reference=load_zone_reference(game_date),
    )
    return model, context


def has_model(pitcher_id: int, game_date: str) -> bool:
    return (
        DATA_ROOT / "models" / game_date / "pitchers" / f"{pitcher_id}.joblib"
    ).exists()


def dates_with_models() -> list[dt.date]:
    root = DATA_ROOT / "models"
    if not root.exists():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.glob("pitchers/*.joblib")):
            try:
                found.append(dt.date.fromisoformat(child.name))
            except ValueError:
                continue
    return found


def starters_for(payload: dict) -> dict[str, int]:
    probables = payload.get("gameData", {}).get("probablePitchers", {})
    starters: dict[str, int] = {}
    for side in ("away", "home"):
        pitcher = probables.get(side)
        if pitcher:
            starters[pitcher["fullName"]] = int(pitcher["id"])
    return starters


# ============================================================
# SCORES VIEW
# ============================================================


def headline_metrics() -> None:
    history = load_performance_history()
    left, right = st.columns(2)
    if history.empty:
        with left:
            block(hero_metric_html("Pitch prediction accuracy", "--", "no history yet"))
        with right:
            block(hero_metric_html("Relative improvement", "--", "no history yet"))
        return

    pitches = int(history["pitch_count"].sum())
    accuracy = history["model_correct"].sum() / pitches
    baseline = history["baseline_correct"].sum() / pitches
    relative = (accuracy - baseline) / baseline
    games = len(history)
    span = (
        f"{history['game_date'].min():%b %-d} to {history['game_date'].max():%b %-d, %Y}"
    )

    with left:
        block(
            hero_metric_html(
                "Pitch prediction accuracy",
                f"{accuracy:.1%}",
                f"{pitches:,} pitches across {games} pitcher-games",
            )
        )
    with right:
        block(
            hero_metric_html(
                "Relative improvement over baseline",
                f"{relative:+.1%}",
                f"baseline {baseline:.1%} &middot; {span}",
            )
        )


def game_status(game: dict) -> tuple[str, bool, str]:
    """Return the card's status line, whether it is live, and its detail line."""

    state = game["status"]["abstractGameState"]
    detailed = game["status"]["detailedState"]
    linescore = game.get("linescore") or {}

    if state == "Live":
        half = linescore.get("inningHalf", "")
        inning = linescore.get("currentInning", "")
        outs = linescore.get("outs", 0)
        return (
            f"{half} {inning}".strip(),
            True,
            f"{linescore.get('balls', 0)}-{linescore.get('strikes', 0)}, {outs} out",
        )
    if state == "Final":
        innings = linescore.get("currentInning")
        extra = f" / {innings}" if innings and innings != 9 else ""
        return f"Final{extra}", False, ""
    start = pd.to_datetime(game["gameDate"]).tz_convert("America/New_York")
    return detailed, False, f"{start:%-I:%M %p} ET"


def render_scores(game_date: dt.date) -> None:
    date_text = game_date.isoformat()
    games = load_schedule(date_text)
    st.markdown(f"#### Games &middot; {game_date:%A, %B %-d, %Y}")
    if not games:
        st.info("No games scheduled on this date.")
        return

    columns_per_row = 3
    for start in range(0, len(games), columns_per_row):
        row = games[start : start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column, game in zip(columns, row):
            game_pk = int(game["gamePk"])
            away = game["teams"]["away"]["team"]["abbreviation"]
            home = game["teams"]["home"]["team"]["abbreviation"]
            status, is_live, detail = game_status(game)
            starters = starters_for(
                {"gameData": {"probablePitchers": {
                    side: game["teams"][side].get("probablePitcher")
                    for side in ("away", "home")
                }}}
            )
            eligible = {n: p for n, p in starters.items() if has_model(p, date_text)}
            labels = [
                f"{'&#9654; ' if name in eligible else ''}{name}"
                for name in starters
            ]
            note = None if eligible else "No pre-game model for this date"

            with column:
                block(
                    game_card_html(
                        away=away,
                        home=home,
                        away_score=game["teams"]["away"].get("score"),
                        home_score=game["teams"]["home"].get("score"),
                        status=status,
                        is_live=is_live,
                        detail=detail,
                        pitchers=labels,
                        note=note,
                    )
                )
                st.button(
                    "Open game" if eligible else "Open game (no model)",
                    key=f"open_{game_pk}",
                    disabled=not eligible,
                    use_container_width=True,
                    on_click=_open_game,
                    args=(game_pk, date_text),
                )
        st.write("")


def _open_game(game_pk: int, date_text: str) -> None:
    st.session_state["view"] = "game"
    st.session_state["game_pk"] = game_pk
    st.session_state["game_date"] = date_text


def _back_to_scores() -> None:
    st.session_state["view"] = "scores"


# ============================================================
# GAME VIEW
# ============================================================


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


def engine_for(pitcher_id: int, pitcher_name: str, game_pk: int, date_text: str,
               replay: bool, clock=None) -> LivePredictionEngine:
    key = f"engine_{game_pk}_{pitcher_id}_{int(replay)}"
    if key not in st.session_state:
        model, context = load_engine_parts(pitcher_id, date_text)
        log_path = (
            DATA_ROOT / "predictions" / "live" / f"{game_pk}_{pitcher_id}.jsonl"
        )
        st.session_state[key] = LivePredictionEngine(
            model=model,
            context=context,
            pitcher_name=pitcher_name,
            log_path=log_path,
            **({"clock": clock} if clock else {}),
        )
    return st.session_state[key]


def pitcher_panel(engine: LivePredictionEngine) -> None:
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
    tiles = [
        ("Pitches scored", str(stats.resolved), f"{stats.correct} correct"),
        (
            "Accuracy",
            f"{stats.accuracy:.1%}" if stats.accuracy is not None else "--",
            "predicted type matched",
        ),
        (
            "Relative improvement",
            f"{relative:+.1%}" if relative is not None else "--",
            f"baseline {baseline:.1%}" if baseline else "over stratified baseline",
        ),
        (
            "Predicted before pitch",
            f"{stats.before_pitch}/{stats.resolved}" if stats.resolved else "--",
            f"{stats.late} late",
        ),
    ]
    for column, (label, value, sub) in zip(st.columns(4), tiles):
        with column:
            block(tile_html(label, value, sub))

    with st.expander(f"Full pitch log — {engine.pitcher_name}"):
        if not engine.history:
            st.caption("Predictions appear here as pitches are thrown.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Inn": f"{p.inning_topbot} {p.inning}",
                            "Count": f"{p.balls}-{p.strikes}",
                            "Batter": p.batter_name,
                            "Predicted": p.predicted_pitch_type,
                            "Actual": p.actual_pitch_type or "",
                            "Result": "correct" if p.correct else "missed",
                            "Confidence": p.confidence,
                            "Lead (s)": p.prediction_lead_seconds,
                            "Zone source": p.strike_zone_source or "",
                        }
                        for p in reversed(engine.history)
                    ]
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.NumberColumn(format="%.0f%%"),
                    "Lead (s)": st.column_config.NumberColumn(format="%.1f"),
                },
            )


def render_game(
    game_pk: int,
    date_text: str,
    payload: dict,
    replay: bool,
    following: bool,
) -> None:
    starters = starters_for(payload)
    eligible = {n: p for n, p in starters.items() if has_model(p, date_text)}

    teams = payload.get("gameData", {}).get("teams", {})
    away = teams.get("away", {}).get("abbreviation", "AWY")
    home = teams.get("home", {}).get("abbreviation", "HOM")
    st.markdown(f"### {away} @ {home}")

    if not eligible:
        st.error(
            "No frozen pre-game model for either starter on this date. Run "
            f"`python -m scripts.run_daily_pipeline --date {date_text}` before "
            "the game starts."
        )
        return

    clock = None
    timecodes: list[str] = []
    if replay:
        timecodes = load_timestamps(game_pk)
        position_key = f"tcpos_{game_pk}"
        st.session_state.setdefault(position_key, 0)
        clock = lambda: parse_timecode(  # noqa: E731
            timecodes[max(0, min(st.session_state[position_key], len(timecodes)) - 1)]
        )
        position = min(st.session_state[position_key], len(timecodes) - 1)
        snapshot = get_client().feed_live(game_pk, timecode=timecodes[position])
        if following and st.session_state[position_key] < len(timecodes) - 1:
            st.session_state[position_key] += 1
    else:
        snapshot = get_client().feed_live(game_pk)

    engines = {
        name: engine_for(pid, name, game_pk, date_text, replay, clock)
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
                pitcher_panel(engines[name])
    else:
        st.markdown(f"**{names[0]}**")
        pitcher_panel(engines[names[0]])

    if replay:
        st.caption(
            f"Replay snapshot {min(st.session_state.get(f'tcpos_{game_pk}', 0) + 1, len(timecodes))}"
            f" of {len(timecodes)}"
        )


# ============================================================
# APP
# ============================================================

st.session_state.setdefault("view", "scores")

modelled = dates_with_models()
default_date = modelled[-1] if modelled else dt.date.today()

st.sidebar.title("Live pitch prediction")

if st.session_state["view"] == "game":
    st.sidebar.button("← All games", on_click=_back_to_scores,
                      use_container_width=True)
    game_pk = int(st.session_state["game_pk"])
    date_text = st.session_state["game_date"]

    # A completed game has to be replayed snapshot by snapshot: reading its
    # final feed would deliver every pitch at once, with no pitch ever having
    # been predicted beforehand. The toggle's default must therefore be seeded
    # before the widget is created, because a keyed toggle with no explicit
    # value defaults to False and would override it.
    game_payload = load_snapshot_payload(game_pk)
    is_final = game_payload.get("gameData", {}).get("status", {}).get(
        "abstractGameState"
    ) == "Final"
    st.session_state.setdefault("replay", is_final)

    st.sidebar.toggle(
        "Replay mode",
        key="replay",
        help="Drive the engine through this game's archived feed snapshots, so "
        "the view can be shown when no game is live. Required for a completed "
        "game.",
    )
    following = st.sidebar.toggle("Follow game", value=False)
    interval = st.sidebar.slider("Refresh (seconds)", 2, 15, 3)
    st.sidebar.caption(
        "Each prediction is timestamped and compared against its pitch's feed "
        "start time, so a late prediction is recorded as late rather than "
        "counted."
    )
    st.title("Gameday")
    render_game(
        game_pk, date_text, game_payload, bool(st.session_state["replay"]), following
    )
    if following:
        time.sleep(interval)
        rerun()
    else:
        st.info("Turn on **Follow game** in the sidebar to start predicting.")
else:
    selected = st.sidebar.date_input("Date", value=default_date)
    if modelled:
        st.sidebar.caption(
            "Dates with frozen pre-game models: "
            + ", ".join(d.isoformat() for d in modelled)
        )
    st.title("Live pitch prediction")
    st.caption(
        "Predicting the next pitch type for MLB starting pitchers, before it is "
        "thrown."
    )
    headline_metrics()
    st.write("")
    render_scores(selected)
