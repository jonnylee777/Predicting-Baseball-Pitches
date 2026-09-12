"""MLB pitch prediction — replay dashboard.

Every completed game is replayed pitch by pitch through the frozen pre-game
model for that date, so the record is complete: every pitch a starter threw
has a prediction, an actual, and a baseline to compare against.

Three views:

* **Overview** -- a date's headline accuracy and relative improvement over
  baseline, above a scoreboard of that day's games.
* **Game** -- one game: the score, then a tab per starting pitcher with their
  repertoire, an at-a-glance strip of the whole outing, and the full
  pitch-by-pitch log of predicted against actual.
* **Leaderboard** -- every pitcher evaluated to date, ranked by how far the
  model beats that pitcher's own baseline.

Live in-game prediction was measured at 15% pitch coverage and is not exposed
here; the engine remains in ``pitch_prediction.live`` for future work.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components import (
    THEME_CSS,
    accuracy_strip_html,
    game_summary_card_html,
    hero_metric_html,
    pitch_table_html,
    repertoire_table_html,
    tile_html,
)
from pitch_prediction.live.gumbo import GumboClient

DATA_ROOT = PROJECT_ROOT / "Data" / "daily_pipeline"
PERFORMANCE_HISTORY = DATA_ROOT / "performance_history.csv"

st.set_page_config(page_title="MLB Pitch Prediction", page_icon="⚾", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    """<style>
    .block-container { padding-top: 2.0rem; max-width: 1480px; }
    [data-testid="stSidebar"] { min-width: 250px; max-width: 250px; }
    </style>""",
    unsafe_allow_html=True,
)


def theme_attribute() -> str:
    try:
        base = st.get_option("theme.base")
    except Exception:  # noqa: BLE001 - option absent on some versions
        base = "light"
    return ' data-theme="dark"' if base == "dark" else ' data-theme="light"'


THEME = theme_attribute()


def block(html: str) -> None:
    st.markdown(f'<div class="viz"{THEME}>{html}</div>', unsafe_allow_html=True)


def row_of(items: list[str]) -> None:
    for column, html in zip(st.columns(len(items)), items):
        with column:
            block(html)


# ============================================================
# DATA
# ============================================================


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    if not PERFORMANCE_HISTORY.exists():
        return pd.DataFrame()
    frame = pd.read_csv(PERFORMANCE_HISTORY)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame


def evaluated_dates() -> list[dt.date]:
    history = load_history()
    if history.empty:
        return []
    return sorted({d.date() for d in history["game_date"].dropna()}, reverse=True)


@st.cache_data(show_spinner=False, ttl=900)
def load_schedule(game_date: str) -> dict[int, dict]:
    """Teams and final score per game, for the scoreboard cards."""

    try:
        games = GumboClient().schedule(game_date)
    except Exception:  # noqa: BLE001 - the dashboard still works without it
        return {}
    out = {}
    for game in games:
        out[int(game["gamePk"])] = {
            "away": game["teams"]["away"]["team"].get("abbreviation", "AWY"),
            "home": game["teams"]["home"]["team"].get("abbreviation", "HOM"),
            "away_score": game["teams"]["away"].get("score"),
            "home_score": game["teams"]["home"].get("score"),
            "status": game["status"]["detailedState"],
        }
    return out


@st.cache_data(show_spinner=False)
def load_pitch_log(predictions_path: str) -> pd.DataFrame:
    path = PROJECT_ROOT / predictions_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def repertoire_rows(log: pd.DataFrame) -> list[dict]:
    if log.empty:
        return []
    total = len(log)
    return [
        {
            "pitch": str(pitch),
            "count": int(len(group)),
            "thrown": len(group) / total,
            "recall": float(group["model_correct"].mean()),
        }
        for pitch, group in log.groupby("actual_pitch")
    ]


def _open_game(game_pk: int, game_date: str) -> None:
    st.session_state["game_pk"] = game_pk
    st.session_state["game_date"] = game_date


def _close_game() -> None:
    st.session_state.pop("game_pk", None)


# ============================================================
# OVERVIEW
# ============================================================


def render_overview(game_date: dt.date) -> None:
    history = load_history()
    day = history[history["game_date"].dt.date == game_date]
    if day.empty:
        st.info(
            f"No evaluated games for {game_date:%B %-d, %Y}. Generate them with "
            f"`python -m scripts.run_daily_postgame_replay --date {game_date}`."
        )
        return

    pitches = int(day["pitch_count"].sum())
    accuracy = day["model_correct"].sum() / pitches
    baseline = day["baseline_correct"].sum() / pitches
    row_of(
        [
            hero_metric_html(
                "Pitch prediction accuracy",
                f"{accuracy:.1%}",
                f"{pitches:,} pitches &middot; {len(day)} starts &middot; "
                f"{day['game_pk'].nunique()} games",
            ),
            hero_metric_html(
                "Relative improvement over baseline",
                f"{(accuracy - baseline) / baseline:+.1%}",
                f"baseline {baseline:.1%} &middot; every pitch replayed",
            ),
        ]
    )
    st.write("")
    st.markdown(f"#### {game_date:%A, %B %-d, %Y}")

    schedule = load_schedule(game_date.isoformat())
    games = sorted(day["game_pk"].unique())
    per_row = 3
    for start in range(0, len(games), per_row):
        columns = st.columns(per_row)
        for column, game_pk in zip(columns, games[start : start + per_row]):
            info = schedule.get(int(game_pk), {})
            starters = day[day["game_pk"] == game_pk]
            with column:
                block(
                    game_summary_card_html(
                        away=info.get("away", "AWY"),
                        home=info.get("home", "HOM"),
                        away_score=info.get("away_score"),
                        home_score=info.get("home_score"),
                        status=info.get("status", "Final"),
                        pitchers=[
                            {
                                "name": r.pitcher_name,
                                "pitches": int(r.pitch_count),
                                "accuracy": float(r.model_accuracy),
                                "relative": (
                                    float(r.relative_improvement)
                                    if pd.notna(r.relative_improvement)
                                    else None
                                ),
                            }
                            for r in starters.itertuples()
                        ],
                    )
                )
                st.button(
                    "Pitch by pitch",
                    key=f"open_{game_pk}",
                    use_container_width=True,
                    on_click=_open_game,
                    args=(int(game_pk), game_date.isoformat()),
                )
        st.write("")


# ============================================================
# GAME
# ============================================================


def render_pitcher(record: pd.Series) -> None:
    log = load_pitch_log(record["predictions_path"])
    relative = record.get("relative_improvement")
    row_of(
        [
            tile_html("Pitches", f"{int(record['pitch_count'])}",
                      f"{int(record['model_correct'])} predicted correctly"),
            tile_html("Accuracy", f"{record['model_accuracy']:.1%}",
                      "every pitch of the outing"),
            tile_html("Baseline", f"{record['baseline_accuracy']:.1%}",
                      "stratified from the pre-game mix"),
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
            f"The pitch-by-pitch log is missing at `{record['predictions_path']}`."
        )
        return

    block(
        accuracy_strip_html(
            [bool(x) for x in log["model_correct"]],
            f"The outing, pitch by pitch — {record['pitcher_name']}",
        )
    )
    st.write("")

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
        table, hide_index=True, use_container_width=True, height=420,
        column_config={
            "Confidence": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )


def render_game(game_pk: int, game_date: str) -> None:
    history = load_history()
    starters = history[history["game_pk"] == game_pk]
    st.button("← All games", on_click=_close_game)
    if starters.empty:
        st.info("No evaluated starters for this game.")
        return

    info = load_schedule(game_date).get(game_pk, {})
    away, home = info.get("away", "AWY"), info.get("home", "HOM")
    away_score, home_score = info.get("away_score"), info.get("home_score")
    score = (
        f" &nbsp;{away_score}&ndash;{home_score}"
        if away_score is not None and home_score is not None
        else ""
    )
    st.markdown(f"### {away} @ {home}{score}")
    st.caption(
        f"{dt.date.fromisoformat(game_date):%A, %B %-d, %Y} &middot; "
        "every pitch replayed through the frozen pre-game model"
    )

    names = list(starters["pitcher_name"])
    if len(names) > 1:
        for tab, (_, record) in zip(st.tabs(names), starters.iterrows()):
            with tab:
                render_pitcher(record)
    else:
        render_pitcher(starters.iloc[0])


# ============================================================
# LEADERBOARD
# ============================================================


@st.cache_data(show_spinner=False)
def leaderboard(minimum_starts: int) -> pd.DataFrame:
    history = load_history()
    if history.empty:
        return pd.DataFrame()
    grouped = history.groupby("pitcher_name").agg(
        starts=("pitch_count", "size"),
        pitches=("pitch_count", "sum"),
        correct=("model_correct", "sum"),
        baseline_correct=("baseline_correct", "sum"),
        last_start=("game_date", "max"),
    )
    grouped["accuracy"] = grouped["correct"] / grouped["pitches"]
    grouped["baseline"] = grouped["baseline_correct"] / grouped["pitches"]
    grouped["relative"] = (
        (grouped["accuracy"] - grouped["baseline"]) / grouped["baseline"]
    )
    return (
        grouped[grouped["starts"] >= minimum_starts]
        .reset_index()
        .sort_values("relative", ascending=False)
    )


def render_leaderboard() -> None:
    minimum = st.sidebar.slider("Minimum evaluated starts", 1, 8, 2)
    board = leaderboard(minimum)
    history = load_history()
    if board.empty or history.empty:
        st.info("No evaluated history yet.")
        return

    pitches = int(history["pitch_count"].sum())
    accuracy = history["model_correct"].sum() / pitches
    baseline = history["baseline_correct"].sum() / pitches
    row_of(
        [
            hero_metric_html(
                "Accuracy, all evaluated games",
                f"{accuracy:.1%}",
                f"{pitches:,} pitches &middot; {len(history)} starts &middot; "
                f"{history['pitcher_name'].nunique()} pitchers",
            ),
            hero_metric_html(
                "Relative improvement over baseline",
                f"{(accuracy - baseline) / baseline:+.1%}",
                f"baseline {baseline:.1%}",
            ),
        ]
    )
    st.write("")
    st.markdown(f"#### {len(board)} pitchers with {minimum}+ evaluated starts")
    st.caption(
        "Ranked by relative improvement over each pitcher's own baseline, which "
        "is the fair comparison: a pitcher who throws one pitch 80% of the time "
        "is easy to predict but leaves little room to beat a baseline that "
        "already knows their mix."
    )
    table = pd.DataFrame(
        {
            "Pitcher": board["pitcher_name"],
            "Starts": board["starts"],
            "Pitches": board["pitches"],
            "Accuracy": board["accuracy"],
            "Baseline": board["baseline"],
            "Relative improvement": board["relative"],
            "Last start": board["last_start"].dt.date,
        }
    )
    config = {
        "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
        "Baseline": st.column_config.NumberColumn(format="%.1f%%"),
        "Relative improvement": st.column_config.NumberColumn(format="%+.1f%%"),
    }
    st.dataframe(table, hide_index=True, use_container_width=True,
                 height=420, column_config=config)
    best, worst = st.columns(2)
    with best:
        st.markdown("###### Most predictable")
        st.dataframe(table.head(10), hide_index=True,
                     use_container_width=True, column_config=config)
    with worst:
        st.markdown("###### Least predictable")
        st.dataframe(table.tail(10).iloc[::-1], hide_index=True,
                     use_container_width=True, column_config=config)


# ============================================================
# APP
# ============================================================

st.sidebar.title("MLB pitch prediction")
view = st.sidebar.radio("View", ["Games", "Leaderboard"], key="nav")

dates = evaluated_dates()
st.title("MLB pitch prediction")
st.caption(
    "Every pitch of every evaluated start, predicted by the model that was "
    "frozen before the game and compared against what was actually thrown."
)

if view == "Leaderboard":
    st.session_state.pop("game_pk", None)
    st.sidebar.caption(
        "Every pitcher evaluated to date, ranked against their own baseline."
    )
    render_leaderboard()
elif not dates:
    st.info("No evaluated games yet. Run `./scripts/daily.sh` to build the history.")
elif "game_pk" in st.session_state:
    render_game(int(st.session_state["game_pk"]), st.session_state["game_date"])
else:
    options = {f"{day:%B %-d, %Y}": day for day in dates}
    chosen = st.sidebar.selectbox("Date", list(options))
    st.sidebar.caption(
        "Pick a date to see that day's accuracy and games. Open a game for the "
        "pitch-by-pitch record."
    )
    render_overview(options[chosen])
