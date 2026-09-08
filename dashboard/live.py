"""Live Gameday view: predicted pitch versus actual, with a running accuracy log.

Run with:

    streamlit run dashboard/live.py

The prediction work happens in ``pitch_prediction.live.engine``; this module
only drives it and renders the result. Every resolved prediction is also
appended to a JSONL log by the engine, so closing the browser loses the view,
not the record.

Replay mode drives the same engine through a completed game's archived feed
snapshots, which is how the view can be demonstrated when no game is live.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.components import (
    THEME_CSS,
    GameView,
    last_pitch_html,
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data" / "daily_pipeline"

st.set_page_config(
    page_title="Live Pitch Prediction",
    page_icon="⚾",
    layout="wide",
)

# ============================================================
# DESIGN TOKENS
# ============================================================

st.markdown(THEME_CSS, unsafe_allow_html=True)


def theme_attribute() -> str:
    """Stamp the viewer's theme so it beats the OS setting in both directions.

    Stamping only the dark case would leave a light Streamlit theme showing
    dark cards whenever the operating system is set to dark, so light is
    stamped explicitly too.
    """

    try:
        base = st.get_option("theme.base")
    except Exception:  # noqa: BLE001 - option absent on some versions
        base = "light"
    return ' data-theme="dark"' if base == "dark" else ' data-theme="light"' 


THEME = theme_attribute()


def block(html: str) -> None:
    st.markdown(f'<div class="viz"{THEME}>{html}</div>', unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================


@st.cache_resource(show_spinner=False)
def get_client() -> GumboClient:
    return GumboClient()


@st.cache_data(show_spinner=False, ttl=60)
def load_schedule(game_date: str) -> list[dict]:
    return get_client().schedule(game_date)


@st.cache_data(show_spinner="Loading batter strike-zone reference...")
def load_zone_reference(game_date: str) -> dict[int, tuple[float, float]]:
    return load_or_build(
        DATA_ROOT / "features" / "kg4" / "pitchers",
        through_date=dt.date.fromisoformat(game_date),
        cache_dir=DATA_ROOT / "reference",
    )


@st.cache_resource(show_spinner="Loading pre-game model...")
def load_engine(game_pk: int, pitcher_id: int, game_date: str, pitcher_name: str):
    model = load_pregame_model(DATA_ROOT, game_date, pitcher_id)
    context = build_context(
        DATA_ROOT,
        pitcher_id,
        dt.date.fromisoformat(game_date),
        batter_zone_reference=load_zone_reference(game_date),
    )
    log_path = DATA_ROOT / "predictions" / "live" / f"{game_pk}_{pitcher_id}.jsonl"
    return model, context, log_path


def has_model(pitcher_id: int, game_date: str) -> bool:
    return (
        DATA_ROOT / "models" / game_date / "pitchers" / f"{pitcher_id}.joblib"
    ).exists()


# ============================================================
# RENDERING
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


def render_prediction(pending: PitchPrediction | None) -> None:
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


def render_last_pitch(history: list[PitchPrediction]) -> None:
    last = history[-1] if history else None
    block(
        last_pitch_html(
            batter=last.batter_name if last else None,
            predicted=last.predicted_pitch_type if last else None,
            actual=last.actual_pitch_type if last else None,
            correct=last.correct if last else None,
            timing=last.timing if last else None,
            lead_seconds=last.prediction_lead_seconds if last else None,
        )
    )


def render_stats(engine: LivePredictionEngine) -> None:
    stats = engine.stats
    honest = stats.honest_accuracy(engine.history)
    tiles = [
        ("Pitches scored", f"{stats.resolved}", f"{stats.correct} correct"),
        (
            "Accuracy",
            f"{stats.accuracy:.1%}" if stats.accuracy is not None else "--",
            "predicted type matched",
        ),
        (
            "Accuracy, in time only",
            f"{honest:.1%}" if honest is not None else "--",
            "excludes any late prediction",
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


def render_log(history: list[PitchPrediction]) -> None:
    st.markdown("#### Pitch-by-pitch log")
    if not history:
        st.caption("Predictions appear here as pitches are thrown.")
        return

    frame = pd.DataFrame(
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
                "Timing": p.timing or "",
                "Zone source": p.strike_zone_source or "",
            }
            for p in reversed(history)
        ]
    )
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Confidence": st.column_config.NumberColumn(format="%.0f%%"),
            "Lead (s)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Live prediction")


def latest_modelled_date() -> dt.date:
    """Default to the most recent date with frozen pre-game models.

    Following a game requires a model that was frozen before it started, so
    defaulting to today would usually open on an error.
    """

    model_root = DATA_ROOT / "models"
    dates = []
    if model_root.exists():
        for child in model_root.iterdir():
            if not child.is_dir() or not any(child.glob("pitchers/*.joblib")):
                continue
            try:
                dates.append(dt.date.fromisoformat(child.name))
            except ValueError:
                continue
    return max(dates) if dates else dt.date.today()


selected_date = st.sidebar.date_input("Game date", value=latest_modelled_date())
date_text = selected_date.isoformat()

games = load_schedule(date_text)
if not games:
    st.sidebar.warning("No games scheduled on this date.")
    st.stop()

labels = {
    (
        f"{game['teams']['away']['team']['abbreviation']} @ "
        f"{game['teams']['home']['team']['abbreviation']}  "
        f"({game['status']['detailedState']})"
    ): int(game["gamePk"])
    for game in games
}
game_label = st.sidebar.selectbox("Game", list(labels))
game_pk = labels[game_label]

client = get_client()
snapshot = client.feed_live(game_pk)

# Only starters have models, so only starters can be followed.
starters = {}
for side in ("away", "home"):
    probable = (
        snapshot.payload.get("gameData", {})
        .get("probablePitchers", {})
        .get(side)
    )
    if probable:
        starters[probable["fullName"]] = int(probable["id"])

eligible = {
    name: pid for name, pid in starters.items() if has_model(pid, date_text)
}
if not eligible:
    st.sidebar.error(
        "No frozen pre-game model for either starter on this date.\n\n"
        "Run `python -m scripts.run_daily_pipeline --date "
        f"{date_text}` first."
    )
    missing = [f"{n} ({p})" for n, p in starters.items()]
    st.sidebar.caption("Starters: " + (", ".join(missing) or "unknown"))
    st.stop()

pitcher_name = st.sidebar.selectbox("Pitcher", list(eligible))
pitcher_id = eligible[pitcher_name]

replay = st.sidebar.toggle(
    "Replay mode",
    value=snapshot.is_final,
    help="Drive the engine through this game's archived feed snapshots. Use "
    "this to demonstrate the view when no game is live.",
)
interval = st.sidebar.slider("Refresh (seconds)", 2, 15, 3)
following = st.sidebar.toggle("Follow game", value=False)

st.sidebar.caption(
    "Predictions are timestamped and compared against each pitch's feed start "
    "time, so a late prediction is recorded as late rather than counted."
)

# ============================================================
# MAIN
# ============================================================

st.title(f"{pitcher_name}")
st.caption(f"{game_label} · game {game_pk}")

model, context, log_path = load_engine(game_pk, pitcher_id, date_text, pitcher_name)

state_key = f"engine_{game_pk}_{pitcher_id}_{int(replay)}"
if state_key not in st.session_state:
    clock = None
    if replay:
        timecodes = client.timestamps(game_pk)
        st.session_state[f"tc_{state_key}"] = timecodes
        st.session_state[f"tcpos_{state_key}"] = 0
        clock = lambda: parse_timecode(  # noqa: E731
            st.session_state[f"tc_{state_key}"][
                max(0, st.session_state[f"tcpos_{state_key}"] - 1)
            ]
        )
    st.session_state[state_key] = LivePredictionEngine(
        model=model,
        context=context,
        pitcher_name=pitcher_name,
        log_path=log_path,
        **({"clock": clock} if clock else {}),
    )

engine: LivePredictionEngine = st.session_state[state_key]

fragment = getattr(st, "fragment", None) or st.experimental_fragment


@fragment(run_every=interval if following else None)
def live_panel() -> None:
    current = snapshot
    if replay:
        timecodes = st.session_state[f"tc_{state_key}"]
        position = st.session_state[f"tcpos_{state_key}"]
        if position < len(timecodes):
            current = client.feed_live(game_pk, timecode=timecodes[position])
            st.session_state[f"tcpos_{state_key}"] = position + 1
        else:
            st.info("Replay complete.")
            current = client.feed_live(game_pk, timecode=timecodes[-1])
    else:
        current = client.feed_live(game_pk)

    engine.observe(current)

    block(scoreboard_html(read_game_view(current, engine.pending)))
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        render_prediction(engine.pending)
    with right:
        render_last_pitch(engine.history)
    st.write("")
    render_stats(engine)
    st.write("")
    render_log(engine.history)

    if replay:
        position = st.session_state[f"tcpos_{state_key}"]
        total = len(st.session_state[f"tc_{state_key}"])
        st.caption(f"Replay snapshot {min(position, total)} of {total}")


live_panel()

if not following:
    st.info(
        "Turn on **Follow game** in the sidebar to start predicting. "
        "Each refresh reads the feed, predicts the next pitch, and scores the "
        "previous one."
    )
