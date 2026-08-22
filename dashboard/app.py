"""Streamlit dashboard for MLB pitch prediction performance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "Data"
    / "daily_pipeline"
)

PERFORMANCE_HISTORY_PATH = (
    DATA_ROOT
    / "performance_history.csv"
)

POSTGAME_ROOT = (
    DATA_ROOT
    / "predictions"
    / "postgame"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MLB Pitch Prediction",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Narrow sidebar */
    [data-testid="stSidebar"] {
        min-width: 220px;
        max-width: 220px;
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 220px;
    }

    /* Cleaner sidebar spacing */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.3rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }

    /* Main page width / spacing */
    .block-container {
        padding-top: 1.8rem;
        max-width: 1450px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        padding: 0.2rem 0;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    /* Headings */
    h1, h2, h3 {
        margin-bottom: 0.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert a pandas Series to numeric."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def query_value(
    name: str,
    default=None,
):
    """Read one URL query parameter safely."""

    value = st.query_params.get(
        name,
        default,
    )

    if isinstance(
        value,
        list,
    ):

        if not value:
            return default

        return value[0]

    return value


def relative_improvement(
    model_accuracy: float | None,
    baseline_accuracy: float | None,
) -> float | None:

    if (
        model_accuracy is None
        or baseline_accuracy is None
        or baseline_accuracy <= 0
    ):

        return None

    return (
        model_accuracy
        - baseline_accuracy
    ) / baseline_accuracy


def format_accuracy(
    value: float | None,
) -> str:

    if (
        value is None
        or pd.isna(value)
    ):
        return "—"

    return f"{value:.1%}"


def format_relative(
    value: float | None,
) -> str:

    if (
        value is None
        or pd.isna(value)
    ):
        return "—"

    return f"{value:+.1%}"


def format_percent_text(
    value: float | None,
) -> str:
    """
    Convert decimal accuracy to a clean text percentage.

    Example:
        0.358 -> "35.8%"

    Using text avoids the warning icons Streamlit was showing.
    """

    if (
        value is None
        or pd.isna(value)
    ):
        return "—"

    return (
        f"{value * 100:.1f}%"
    )


# ============================================================
# PERFORMANCE DATA
# ============================================================


def normalize_performance_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize cumulative history and daily summary files into
    one consistent dashboard dataset.
    """

    if data.empty:
        return data

    data = data.copy()

    # --------------------------------------------------------
    # KEEP SUCCESSFUL DAILY REPLAYS
    # --------------------------------------------------------

    if (
        "replay_status"
        in data.columns
    ):

        replay_status = (
            data[
                "replay_status"
            ]
            .astype(str)
            .str.lower()
        )

        # Cumulative history rows may have no replay_status.
        #
        # Keep:
        #   success
        #   missing/NaN replay status
        #
        # Remove:
        #   failed
        #   statcast_not_ready
        #   game_not_final
        valid = (
            replay_status.eq(
                "success"
            )
            |
            data[
                "replay_status"
            ].isna()
        )

        data = data[
            valid
        ].copy()

    # --------------------------------------------------------
    # GAME DATE
    # --------------------------------------------------------

    if (
        "game_date"
        not in data.columns
    ):

        return pd.DataFrame()

    data[
        "game_date"
    ] = pd.to_datetime(
        data[
            "game_date"
        ],
        errors="coerce",
    )

    data = data[
        data[
            "game_date"
        ].notna()
    ].copy()

    # --------------------------------------------------------
    # SEASON
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Some daily_summary.csv files do not contain a season
    # column, while performance_history.csv does.
    #
    # When pandas combines the two, season exists but may be NaN
    # for the daily rows.
    #
    # Therefore we fill ANY missing season using game_date.year.
    # --------------------------------------------------------

    derived_season = (
        data[
            "game_date"
        ]
        .dt.year
    )

    if (
        "season"
        not in data.columns
    ):

        data[
            "season"
        ] = derived_season

    else:

        data[
            "season"
        ] = safe_numeric(
            data[
                "season"
            ]
        )

        data[
            "season"
        ] = (
            data[
                "season"
            ]
            .fillna(
                derived_season
            )
        )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "game_pk",
        "pitcher_id",
        "pitch_count",
        "model_correct",
        "baseline_correct",
        "model_accuracy",
        "baseline_accuracy",
        "accuracy_over_baseline",
        "accuracy_over_baseline_pp",
        "relative_improvement",
        "relative_improvement_percent",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[
                column
            ] = safe_numeric(
                data[
                    column
                ]
            )

    # --------------------------------------------------------
    # RELATIVE IMPROVEMENT
    # --------------------------------------------------------

    if (
        "relative_improvement"
        not in data.columns
    ):

        data[
            "relative_improvement"
        ] = pd.NA

    if {
        "model_accuracy",
        "baseline_accuracy",
    }.issubset(
        data.columns
    ):

        missing_relative = (
            data[
                "relative_improvement"
            ].isna()
        )

        valid_baseline = (
            data[
                "baseline_accuracy"
            ]
            > 0
        )

        mask = (
            missing_relative
            & valid_baseline
        )

        data.loc[
            mask,
            "relative_improvement",
        ] = (
            (
                data.loc[
                    mask,
                    "model_accuracy",
                ]
                -
                data.loc[
                    mask,
                    "baseline_accuracy",
                ]
            )
            /
            data.loc[
                mask,
                "baseline_accuracy",
            ]
        )

    # --------------------------------------------------------
    # BASELINE STRATEGY
    # --------------------------------------------------------
    #
    # If stratified rows exist, prefer them.
    #
    # Older most-frequent results should not be mixed into
    # current stratified-baseline dashboard statistics.
    # --------------------------------------------------------

    if (
        "baseline_strategy"
        in data.columns
    ):

        baseline_text = (
            data[
                "baseline_strategy"
            ]
            .astype(str)
            .str.lower()
        )

        has_stratified = (
            baseline_text
            .eq(
                "stratified"
            )
            .any()
        )

        if has_stratified:

            data = data[
                baseline_text.eq(
                    "stratified"
                )
                |
                data[
                    "baseline_strategy"
                ].isna()
            ].copy()

    return data


@st.cache_data
def load_performance_data() -> pd.DataFrame:
    """
    Load all existing completed evaluation results.

    Daily summary files are loaded after cumulative history so a
    newly rerun pitcher-game replaces an older saved copy.
    """

    frames = []

    # --------------------------------------------------------
    # CUMULATIVE HISTORY
    # --------------------------------------------------------

    if (
        PERFORMANCE_HISTORY_PATH
        .exists()
    ):

        try:

            history = pd.read_csv(
                PERFORMANCE_HISTORY_PATH,
                low_memory=False,
            )

            if not history.empty:

                frames.append(
                    history
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # DAILY SUMMARIES
    # --------------------------------------------------------

    if POSTGAME_ROOT.exists():

        daily_files = sorted(
            POSTGAME_ROOT.glob(
                "*/daily_summary.csv"
            )
        )

        for path in daily_files:

            try:

                daily = pd.read_csv(
                    path,
                    low_memory=False,
                )

            except Exception:

                continue

            if daily.empty:
                continue

            frames.append(
                daily
            )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined = (
        normalize_performance_data(
            combined
        )
    )

    if combined.empty:
        return combined

    # --------------------------------------------------------
    # DEDUPLICATE PITCHER-GAMES
    # --------------------------------------------------------

    if {
        "game_pk",
        "pitcher_id",
    }.issubset(
        combined.columns
    ):

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "game_pk",
                    "pitcher_id",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

    return combined


# ============================================================
# MLB SCOREBOARD
# ============================================================


@st.cache_data(
    ttl=3600
)
def load_mlb_scoreboard(
    game_date: str,
) -> pd.DataFrame:
    """
    Fetch matchup and score information from MLB.

    This does NOT determine model accuracy.
    It is only presentation data for the dashboard.
    """

    url = (
        "https://statsapi.mlb.com/"
        "api/v1/schedule"
    )

    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": (
            "team,"
            "probablePitcher"
        ),
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        payload = (
            response.json()
        )

    except Exception:

        return pd.DataFrame()

    rows = []

    for date_entry in payload.get(
        "dates",
        [],
    ):

        for game in date_entry.get(
            "games",
            [],
        ):

            teams = game.get(
                "teams",
                {},
            )

            away = teams.get(
                "away",
                {},
            )

            home = teams.get(
                "home",
                {},
            )

            rows.append(
                {
                    "game_pk":
                        game.get(
                            "gamePk"
                        ),

                    "away_team":
                        away.get(
                            "team",
                            {},
                        ).get(
                            "name",
                            "Away",
                        ),

                    "home_team":
                        home.get(
                            "team",
                            {},
                        ).get(
                            "name",
                            "Home",
                        ),

                    "away_score":
                        away.get(
                            "score"
                        ),

                    "home_score":
                        home.get(
                            "score"
                        ),

                    "away_pitcher":
                        away.get(
                            "probablePitcher",
                            {},
                        ).get(
                            "fullName"
                        ),

                    "home_pitcher":
                        home.get(
                            "probablePitcher",
                            {},
                        ).get(
                            "fullName"
                        ),

                    "status":
                        game.get(
                            "status",
                            {},
                        ).get(
                            "detailedState",
                            "Unknown",
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ACCURACY HELPERS
# ============================================================


def weighted_accuracy(
    data: pd.DataFrame,
    *,
    correct_column: str,
    accuracy_column: str,
) -> float | None:
    """
    Calculate pitch-weighted accuracy.

    Prefer exact correct counts when available.
    """

    if data.empty:
        return None

    # --------------------------------------------------------
    # EXACT COUNTS
    # --------------------------------------------------------

    if (
        correct_column
        in data.columns
        and
        "pitch_count"
        in data.columns
    ):

        correct = safe_numeric(
            data[
                correct_column
            ]
        )

        pitches = safe_numeric(
            data[
                "pitch_count"
            ]
        )

        valid = (
            correct.notna()
            & pitches.notna()
        )

        total_pitches = float(
            pitches[
                valid
            ].sum()
        )

        if total_pitches > 0:

            return float(
                correct[
                    valid
                ].sum()
                / total_pitches
            )

    # --------------------------------------------------------
    # ACCURACY × PITCH COUNT FALLBACK
    # --------------------------------------------------------

    if (
        accuracy_column
        in data.columns
        and
        "pitch_count"
        in data.columns
    ):

        accuracy = safe_numeric(
            data[
                accuracy_column
            ]
        )

        pitches = safe_numeric(
            data[
                "pitch_count"
            ]
        )

        valid = (
            accuracy.notna()
            & pitches.notna()
        )

        total_pitches = float(
            pitches[
                valid
            ].sum()
        )

        if total_pitches > 0:

            return float(
                (
                    accuracy[
                        valid
                    ]
                    * pitches[
                        valid
                    ]
                ).sum()
                / total_pitches
            )

    return None


def pitcher_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:

    if (
        data.empty
        or
        "pitcher_name"
        not in data.columns
    ):

        return pd.DataFrame()

    rows = []

    for pitcher_name, group in (
        data.groupby(
            "pitcher_name"
        )
    ):

        pitches = int(
            safe_numeric(
                group[
                    "pitch_count"
                ]
            )
            .fillna(0)
            .sum()
        )

        model_accuracy = (
            weighted_accuracy(
                group,
                correct_column=(
                    "model_correct"
                ),
                accuracy_column=(
                    "model_accuracy"
                ),
            )
        )

        baseline_accuracy = (
            weighted_accuracy(
                group,
                correct_column=(
                    "baseline_correct"
                ),
                accuracy_column=(
                    "baseline_accuracy"
                ),
            )
        )

        if model_accuracy is None:
            continue

        if (
            "model_correct"
            in group.columns
        ):

            model_correct = int(
                safe_numeric(
                    group[
                        "model_correct"
                    ]
                )
                .fillna(0)
                .sum()
            )

        else:

            model_correct = int(
                round(
                    model_accuracy
                    * pitches
                )
            )

        relative = (
            relative_improvement(
                model_accuracy,
                baseline_accuracy,
            )
        )

        rows.append(
            {
                "Pitcher":
                    pitcher_name,

                "Games":
                    int(
                        len(group)
                    ),

                "Pitches Evaluated":
                    pitches,

                "Correct Predictions":
                    model_correct,

                "Model Accuracy":
                    model_accuracy,

                "Baseline Accuracy":
                    baseline_accuracy,

                "Relative Improvement":
                    relative,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD DATA
# ============================================================

data = load_performance_data()


# ============================================================
# SAFETY CHECK
# ============================================================

if data.empty:

    st.title(
        "⚾ MLB Pitch Prediction"
    )

    st.warning(
        "No completed prediction results were found."
    )

    st.stop()


available_seasons = sorted(
    [
        int(value)
        for value
        in data[
            "season"
        ]
        .dropna()
        .unique()
    ],
    reverse=True,
)


# Extra protection against the exact error you encountered.
if not available_seasons:

    st.title(
        "⚾ MLB Pitch Prediction"
    )

    st.error(
        "Prediction results were found, but no valid "
        "season could be determined from game_date."
    )

    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

NAVIGATION = [
    "Overview",
    "Performance Over Time",
    "Leaderboards",
    "Game Detail",
]


requested_view = str(
    query_value(
        "view",
        "Overview",
    )
)


if (
    requested_view
    not in NAVIGATION
):

    requested_view = (
        "Overview"
    )


st.sidebar.markdown(
    "### Dashboard"
)


view = st.sidebar.selectbox(
    "View",
    options=NAVIGATION,
    index=(
        NAVIGATION.index(
            requested_view
        )
    ),
)


# ============================================================
# SEASON
# ============================================================

requested_season = (
    query_value(
        "season"
    )
)


try:

    requested_season = int(
        requested_season
    )

except (
    TypeError,
    ValueError,
):

    requested_season = (
        available_seasons[0]
    )


if (
    requested_season
    not in available_seasons
):

    requested_season = (
        available_seasons[0]
    )


season = st.sidebar.selectbox(
    "Season",
    options=available_seasons,
    index=(
        available_seasons.index(
            requested_season
        )
    ),
)


season_data = data[
    data[
        "season"
    ]
    == season
].copy()


# ============================================================
# DATE
# ============================================================

selected_date = None


if view in {
    "Overview",
    "Game Detail",
}:

    available_dates = sorted(
        {
            timestamp.date()
            for timestamp
            in season_data[
                "game_date"
            ]
            .dropna()
        },
        reverse=True,
    )

    if available_dates:

        requested_date = (
            query_value(
                "date"
            )
        )

        try:

            requested_date = (
                pd.Timestamp(
                    requested_date
                )
                .date()
            )

        except Exception:

            requested_date = (
                available_dates[0]
            )

        if (
            requested_date
            not in available_dates
        ):

            requested_date = (
                available_dates[0]
            )

        selected_date = (
            st.sidebar.selectbox(
                "Game Date",
                options=available_dates,
                index=(
                    available_dates.index(
                        requested_date
                    )
                ),
                format_func=lambda value: (
                    value.strftime(
                        "%b %d, %Y"
                    )
                ),
            )
        )


# ============================================================
# TITLE
# ============================================================

st.title(
    "⚾ MLB Pitch Prediction"
)

st.caption(
    "Random Forest pitch-type predictions "
    "vs. stratified baseline"
)


# ============================================================
# OVERVIEW
# ============================================================

if view == "Overview":

    if selected_date is None:

        st.info(
            "No evaluated games are available."
        )

        st.stop()

    day_data = season_data[
        season_data[
            "game_date"
        ].dt.date
        == selected_date
    ].copy()

    model_accuracy = (
        weighted_accuracy(
            day_data,
            correct_column=(
                "model_correct"
            ),
            accuracy_column=(
                "model_accuracy"
            ),
        )
    )

    baseline_accuracy = (
        weighted_accuracy(
            day_data,
            correct_column=(
                "baseline_correct"
            ),
            accuracy_column=(
                "baseline_accuracy"
            ),
        )
    )

    relative = relative_improvement(
        model_accuracy,
        baseline_accuracy,
    )

    total_pitches = int(
        safe_numeric(
            day_data[
                "pitch_count"
            ]
        )
        .fillna(0)
        .sum()
    )

    pitcher_games = int(
        len(day_data)
    )

    st.subheader(
        selected_date.strftime(
            "%B %d, %Y"
        )
    )

    metrics = st.columns(
        5
    )

    metrics[0].metric(
        "Model Accuracy",
        format_accuracy(
            model_accuracy
        ),
    )

    metrics[1].metric(
        "Stratified Baseline",
        format_accuracy(
            baseline_accuracy
        ),
    )

    metrics[2].metric(
        "Relative Improvement",
        format_relative(
            relative
        ),
    )

    metrics[3].metric(
        "Pitches Evaluated",
        f"{total_pitches:,}",
    )

    metrics[4].metric(
        "Pitcher-Games",
        f"{pitcher_games:,}",
    )

    st.divider()

    # ========================================================
    # GAMES
    # ========================================================

    st.subheader(
        "Games"
    )

    st.caption(
        "Select a game to open its prediction details."
    )

    scoreboard = (
        load_mlb_scoreboard(
            selected_date.isoformat()
        )
    )

    scoreboard_lookup = {}

    if not scoreboard.empty:

        for row in (
            scoreboard.itertuples(
                index=False
            )
        ):

            if pd.notna(
                row.game_pk
            ):

                scoreboard_lookup[
                    int(
                        row.game_pk
                    )
                ] = row

    game_rows = []

    for game_pk, group in (
        day_data.groupby(
            "game_pk"
        )
    ):

        game_pk_int = int(
            game_pk
        )

        mlb_game = (
            scoreboard_lookup.get(
                game_pk_int
            )
        )

        evaluated_pitchers = (
            group[
                "pitcher_name"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if mlb_game is not None:

            matchup = (
                f"{mlb_game.away_team} "
                f"@ "
                f"{mlb_game.home_team}"
            )

            if (
                pd.notna(
                    mlb_game.away_score
                )
                and
                pd.notna(
                    mlb_game.home_score
                )
            ):

                score = (
                    f"{int(mlb_game.away_score)}"
                    f" - "
                    f"{int(mlb_game.home_score)}"
                )

            else:

                score = "—"

            starters = [
                pitcher
                for pitcher
                in [
                    mlb_game.away_pitcher,
                    mlb_game.home_pitcher,
                ]
                if pitcher
            ]

            if starters:

                starting_pitchers = (
                    " vs ".join(
                        starters
                    )
                )

            else:

                starting_pitchers = (
                    " vs ".join(
                        evaluated_pitchers
                    )
                )

        else:

            matchup = (
                f"Game {game_pk_int}"
            )

            score = "—"

            starting_pitchers = (
                " vs ".join(
                    evaluated_pitchers
                )
            )

        game_rows.append(
            {
                "_game_pk":
                    game_pk_int,

                "Matchup":
                    matchup,

                "Score":
                    score,

                "Starting Pitchers":
                    starting_pitchers,

                "Evaluated":
                    len(group),
            }
        )

    games = pd.DataFrame(
        game_rows
    )

    if games.empty:

        st.info(
            "No evaluated games found "
            "for this date."
        )

    else:

        display_games = (
            games[
                [
                    "Matchup",
                    "Score",
                    "Starting Pitchers",
                    "Evaluated",
                ]
            ]
            .copy()
        )

        event = st.dataframe(
            display_games,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_rows = (
            event.selection.rows
        )

        if selected_rows:

            selected_index = (
                selected_rows[0]
            )

            selected_game_pk = int(
                games.iloc[
                    selected_index
                ][
                    "_game_pk"
                ]
            )

            st.query_params[
                "view"
            ] = "Game Detail"

            st.query_params[
                "season"
            ] = str(
                season
            )

            st.query_params[
                "date"
            ] = (
                selected_date
                .isoformat()
            )

            st.query_params[
                "game_pk"
            ] = str(
                selected_game_pk
            )

            st.rerun()


# ============================================================
# PERFORMANCE OVER TIME
# ============================================================

elif (
    view
    == "Performance Over Time"
):

    st.subheader(
        f"{season} Performance Over Time"
    )

    st.caption(
        "Daily accuracy is pitch-weighted across "
        "all evaluated starts."
    )

    daily_rows = []

    grouped = (
        season_data
        .dropna(
            subset=[
                "game_date"
            ]
        )
        .groupby(
            season_data[
                "game_date"
            ].dt.date
        )
    )

    for game_date, group in grouped:

        model = weighted_accuracy(
            group,
            correct_column=(
                "model_correct"
            ),
            accuracy_column=(
                "model_accuracy"
            ),
        )

        baseline = weighted_accuracy(
            group,
            correct_column=(
                "baseline_correct"
            ),
            accuracy_column=(
                "baseline_accuracy"
            ),
        )

        if (
            model is None
            or baseline is None
        ):

            continue

        daily_rows.append(
            {
                "Date":
                    pd.Timestamp(
                        game_date
                    ),

                "Model Accuracy":
                    model
                    * 100,

                "Stratified Baseline":
                    baseline
                    * 100,
            }
        )

    performance = pd.DataFrame(
        daily_rows
    )

    if performance.empty:

        st.info(
            "Not enough historical data "
            "for a trend chart yet."
        )

    else:

        performance = (
            performance
            .sort_values(
                "Date"
            )
            .set_index(
                "Date"
            )
        )

        st.line_chart(
            performance,
            height=500,
        )


# ============================================================
# LEADERBOARDS
# ============================================================

elif view == "Leaderboards":

    st.subheader(
        f"{season} Pitcher Leaderboards"
    )

    summaries = pitcher_summary(
        season_data
    )

    if summaries.empty:

        st.info(
            "Not enough pitcher data "
            "for leaderboards yet."
        )

        st.stop()

    # ========================================================
    # MOST TOTAL CORRECT PITCHES
    # ========================================================

    st.markdown(
        "### Most Correct Predictions"
    )

    st.caption(
        "Pitchers with the most total pitches "
        "correctly predicted by the model."
    )

    total_leaderboard = (
        summaries
        .sort_values(
            [
                "Correct Predictions",
                "Model Accuracy",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(15)
    )

    total_display = pd.DataFrame(
        {
            "Pitcher":
                total_leaderboard[
                    "Pitcher"
                ],

            "Games":
                total_leaderboard[
                    "Games"
                ],

            "Pitches":
                total_leaderboard[
                    "Pitches Evaluated"
                ],

            "Correct Predictions":
                total_leaderboard[
                    "Correct Predictions"
                ],

            "Accuracy":
                total_leaderboard[
                    "Model Accuracy"
                ].map(
                    format_percent_text
                ),
        }
    )

    st.dataframe(
        total_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ========================================================
    # HIGHEST PREDICTION PROPORTION
    # ========================================================

    st.markdown(
        "### Highest Prediction Accuracy"
    )

    maximum_pitches = max(
        1,
        int(
            summaries[
                "Pitches Evaluated"
            ].max()
        ),
    )

    default_minimum = min(
        300,
        maximum_pitches,
    )

    minimum_pitches = (
        st.sidebar.number_input(
            "Leaderboard minimum pitches",
            min_value=1,
            max_value=(
                maximum_pitches
            ),
            value=max(
                1,
                default_minimum,
            ),
            step=50,
        )
    )

    eligible = summaries[
        summaries[
            "Pitches Evaluated"
        ]
        >= minimum_pitches
    ].copy()

    if eligible.empty:

        st.info(
            "No pitchers meet the current "
            "minimum-pitch requirement."
        )

    else:

        accuracy_leaderboard = (
            eligible
            .sort_values(
                [
                    "Model Accuracy",
                    "Pitches Evaluated",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(15)
        )

        accuracy_display = pd.DataFrame(
            {
                "Pitcher":
                    accuracy_leaderboard[
                        "Pitcher"
                    ],

                "Games":
                    accuracy_leaderboard[
                        "Games"
                    ],

                "Pitches":
                    accuracy_leaderboard[
                        "Pitches Evaluated"
                    ],

                "Accuracy":
                    accuracy_leaderboard[
                        "Model Accuracy"
                    ].map(
                        format_percent_text
                    ),

                "Baseline":
                    accuracy_leaderboard[
                        "Baseline Accuracy"
                    ].map(
                        format_percent_text
                    ),

                "Relative Improvement":
                    accuracy_leaderboard[
                        "Relative Improvement"
                    ].map(
                        format_relative
                    ),
            }
        )

        st.caption(
            f"Minimum {minimum_pitches:,} "
            "evaluated pitches."
        )

        st.dataframe(
            accuracy_display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# GAME DETAIL
# ============================================================

elif view == "Game Detail":

    game_pk_value = (
        query_value(
            "game_pk"
        )
    )

    try:

        selected_game_pk = int(
            game_pk_value
        )

    except (
        TypeError,
        ValueError,
    ):

        selected_game_pk = None

    # --------------------------------------------------------
    # USE CLICKED GAME
    # --------------------------------------------------------

    if (
        selected_game_pk
        is not None
        and
        "game_pk"
        in season_data.columns
        and
        selected_game_pk
        in set(
            season_data[
                "game_pk"
            ]
            .dropna()
            .astype(int)
        )
    ):

        game_data = season_data[
            season_data[
                "game_pk"
            ]
            == selected_game_pk
        ].copy()

        game_date = (
            game_data[
                "game_date"
            ]
            .iloc[0]
            .date()
        )

    # --------------------------------------------------------
    # MANUAL GAME SELECTION
    # --------------------------------------------------------

    else:

        if selected_date is None:

            st.info(
                "Select a game from the Overview page."
            )

            st.stop()

        date_data = season_data[
            season_data[
                "game_date"
            ].dt.date
            == selected_date
        ].copy()

        game_pks = sorted(
            date_data[
                "game_pk"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not game_pks:

            st.info(
                "No evaluated games are available."
            )

            st.stop()

        selected_game_pk = (
            st.selectbox(
                "Game",
                options=game_pks,
            )
        )

        game_data = date_data[
            date_data[
                "game_pk"
            ]
            == selected_game_pk
        ].copy()

        game_date = (
            selected_date
        )

    # ========================================================
    # SCORE
    # ========================================================

    scoreboard = (
        load_mlb_scoreboard(
            game_date.isoformat()
        )
    )

    mlb_game = None

    if not scoreboard.empty:

        match = scoreboard[
            pd.to_numeric(
                scoreboard[
                    "game_pk"
                ],
                errors="coerce",
            )
            == selected_game_pk
        ]

        if not match.empty:

            mlb_game = (
                match.iloc[0]
            )

    if mlb_game is not None:

        st.subheader(
            f"{mlb_game['away_team']} "
            f"@ "
            f"{mlb_game['home_team']}"
        )

        if (
            pd.notna(
                mlb_game[
                    "away_score"
                ]
            )
            and
            pd.notna(
                mlb_game[
                    "home_score"
                ]
            )
        ):

            st.markdown(
                f"## "
                f"{int(mlb_game['away_score'])}"
                f" – "
                f"{int(mlb_game['home_score'])}"
            )

    else:

        st.subheader(
            f"Game {selected_game_pk}"
        )

    st.caption(
        game_date.strftime(
            "%B %d, %Y"
        )
    )

    # ========================================================
    # STARTER
    # ========================================================

    pitchers = (
        game_data[
            "pitcher_name"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not pitchers:

        st.info(
            "No evaluated pitcher was found."
        )

        st.stop()

    if len(pitchers) == 1:

        selected_pitcher = (
            pitchers[0]
        )

        st.markdown(
            f"### {selected_pitcher}"
        )

    else:

        selected_pitcher = (
            st.selectbox(
                "Starting Pitcher",
                options=pitchers,
            )
        )

    pitcher_row = (
        game_data[
            game_data[
                "pitcher_name"
            ]
            == selected_pitcher
        ]
        .iloc[0]
    )

    model_accuracy = float(
        pitcher_row[
            "model_accuracy"
        ]
    )

    baseline_accuracy = float(
        pitcher_row[
            "baseline_accuracy"
        ]
    )

    relative = (
        relative_improvement(
            model_accuracy,
            baseline_accuracy,
        )
    )

    pitch_count = int(
        pitcher_row[
            "pitch_count"
        ]
    )

    metrics = st.columns(
        4
    )

    metrics[0].metric(
        "Model Accuracy",
        format_accuracy(
            model_accuracy
        ),
    )

    metrics[1].metric(
        "Stratified Baseline",
        format_accuracy(
            baseline_accuracy
        ),
    )

    metrics[2].metric(
        "Relative Improvement",
        format_relative(
            relative
        ),
    )

    metrics[3].metric(
        "Pitches Evaluated",
        f"{pitch_count:,}",
    )

    st.divider()

    # ========================================================
    # PITCH LOG
    # ========================================================

    st.subheader(
        "Pitch-by-Pitch Predictions"
    )

    prediction_path_value = (
        pitcher_row.get(
            "predictions_path"
        )
    )

    if (
        prediction_path_value
        is None
        or
        pd.isna(
            prediction_path_value
        )
    ):

        st.info(
            "No pitch-level prediction file "
            "was recorded for this start."
        )

        st.stop()

    prediction_path = Path(
        str(
            prediction_path_value
        )
    )

    if not prediction_path.is_absolute():

        prediction_path = (
            PROJECT_ROOT
            / prediction_path
        )

    if not prediction_path.exists():

        st.info(
            "The pitch-level prediction file "
            "could not be found."
        )

        st.stop()

    pitch_log = pd.read_csv(
        prediction_path,
        low_memory=False,
    )

    preferred_columns = [
        "pitch_number_of_game",
        "inning",
        "inning_topbot",
        "count",
        "pitch_type_of_prev_pitch",
        "model_prediction",
        "actual_pitch",
        "model_confidence",
        "model_correct",
        "baseline_prediction",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column
        in pitch_log.columns
    ]

    pitch_display = (
        pitch_log[
            available_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # CLEAN CONFIDENCE DISPLAY
    # --------------------------------------------------------

    if (
        "model_confidence"
        in pitch_display.columns
    ):

        pitch_display[
            "model_confidence"
        ] = (
            safe_numeric(
                pitch_display[
                    "model_confidence"
                ]
            )
            .map(
                format_percent_text
            )
        )

    # --------------------------------------------------------
    # CLEAN BOOLEAN DISPLAY
    # --------------------------------------------------------

    if (
        "model_correct"
        in pitch_display.columns
    ):

        correct_text = (
            pitch_display[
                "model_correct"
            ]
            .astype(str)
            .str.lower()
        )

        pitch_display[
            "model_correct"
        ] = correct_text.map(
            {
                "true": "✓",
                "false": "✗",
            }
        ).fillna(
            pitch_display[
                "model_correct"
            ]
        )

    pitch_display = (
        pitch_display.rename(
            columns={
                "pitch_number_of_game":
                    "Pitch #",

                "inning":
                    "Inning",

                "inning_topbot":
                    "Half",

                "count":
                    "Count",

                "pitch_type_of_prev_pitch":
                    "Previous Pitch",

                "model_prediction":
                    "Prediction",

                "actual_pitch":
                    "Actual",

                "model_confidence":
                    "Confidence",

                "model_correct":
                    "Correct",

                "baseline_prediction":
                    "Baseline Guess",
            }
        )
    )

    st.dataframe(
        pitch_display,
        use_container_width=True,
        hide_index=True,
    )