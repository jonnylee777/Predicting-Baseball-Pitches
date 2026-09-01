"""Build the README results section from the production performance history.

Reads:

    Data/daily_pipeline/performance_history.csv

Writes:

    Docs/assets/recent_performance_light.png
    Docs/assets/recent_performance_dark.png

and replaces the block between the RESULTS markers in README.md.

The reported window is the trailing N days (default 14) ending on the most
recent evaluated game date. Accuracy is pitch-weighted across every
pitcher-game in the window, matching the dashboard convention:

    accuracy = sum(correct pitches) / sum(pitches)

Relative improvement is measured against the stratified baseline:

    (model_accuracy - baseline_accuracy) / baseline_accuracy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_PATH = (
    PROJECT_ROOT
    / "Data"
    / "daily_pipeline"
    / "performance_history.csv"
)

ASSET_DIR = (
    PROJECT_ROOT
    / "Docs"
    / "assets"
)

README_PATH = PROJECT_ROOT / "README.md"

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"

DEFAULT_WINDOW_DAYS = 14


# ============================================================
# THEME
# ============================================================


@dataclass(frozen=True)
class Theme:
    """Chart surface, ink, and series colors for one color scheme."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series="#2a78d6",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series="#3987e5",
)


# ============================================================
# AGGREGATION
# ============================================================


def load_window(
    history_path: Path,
    window_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Return the pitcher-games evaluated in the trailing window."""

    history = pd.read_csv(
        history_path,
        low_memory=False,
    )

    history["game_date"] = pd.to_datetime(
        history["game_date"],
        errors="coerce",
    )

    for column in [
        "pitch_count",
        "model_correct",
        "baseline_correct",
    ]:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    history = history.dropna(
        subset=[
            "game_date",
            "pitch_count",
            "model_correct",
            "baseline_correct",
        ]
    )

    if history.empty:
        raise SystemExit(
            "No evaluated games found in performance history."
        )

    end = history["game_date"].max()

    start = end - pd.Timedelta(
        days=window_days - 1
    )

    window = history[
        history["game_date"].between(
            start,
            end,
        )
    ].copy()

    return window, start, end


def daily_frame(
    window: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse pitcher-games into one pitch-weighted row per game date."""

    daily = (
        window
        .groupby(
            "game_date",
            as_index=False,
        )
        .agg(
            pitcher_games=("game_pk", "size"),
            pitchers=("pitcher_id", "nunique"),
            pitches=("pitch_count", "sum"),
            model_correct=("model_correct", "sum"),
            baseline_correct=("baseline_correct", "sum"),
        )
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    daily["model_accuracy"] = (
        daily["model_correct"]
        / daily["pitches"]
    )

    daily["baseline_accuracy"] = (
        daily["baseline_correct"]
        / daily["pitches"]
    )

    daily["relative_improvement"] = (
        daily["model_accuracy"]
        - daily["baseline_accuracy"]
    ) / daily["baseline_accuracy"]

    return daily


def pooled_totals(
    window: pd.DataFrame,
) -> dict:
    """Pitch-weighted totals across the whole window."""

    pitches = float(
        window["pitch_count"].sum()
    )

    model_accuracy = (
        float(window["model_correct"].sum())
        / pitches
    )

    baseline_accuracy = (
        float(window["baseline_correct"].sum())
        / pitches
    )

    return {
        "games": int(window["game_pk"].nunique()),
        "pitcher_games": int(len(window)),
        "pitchers": int(window["pitcher_id"].nunique()),
        "pitches": int(pitches),
        "model_accuracy": model_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "relative_improvement": (
            model_accuracy
            - baseline_accuracy
        ) / baseline_accuracy,
        "beat_baseline": int(
            (
                window["model_correct"]
                > window["baseline_correct"]
            ).sum()
        ),
    }


# ============================================================
# CHART
# ============================================================


def rounded_column(
    ax,
    x: float,
    width: float,
    height: float,
    *,
    color: str,
    radius_px: float = 10.0,
) -> None:
    """Draw one column with rounded data-end corners and a square baseline."""

    if height <= 0:
        return

    corners = ax.transData.transform(
        [
            (0.0, 0.0),
            (1.0, 1.0),
        ]
    )

    x_per_px = 1.0 / abs(
        corners[1][0] - corners[0][0]
    )

    y_per_px = 1.0 / abs(
        corners[1][1] - corners[0][1]
    )

    radius_x = min(
        radius_px * x_per_px,
        width / 2.0,
    )

    radius_y = min(
        radius_px * y_per_px,
        height / 2.0,
    )

    left = x - width / 2.0
    right = x + width / 2.0

    vertices = [
        (left, 0.0),
        (left, height - radius_y),
        (left, height),
        (left + radius_x, height),
        (right - radius_x, height),
        (right, height),
        (right, height - radius_y),
        (right, 0.0),
        (left, 0.0),
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]

    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
    )


def style_axes(
    ax,
    theme: Theme,
) -> None:
    """Apply the recessive chrome shared by both panels."""

    ax.set_facecolor(theme.surface)

    ax.grid(
        axis="y",
        color=theme.grid,
        linewidth=0.8,
        linestyle="-",
        zorder=0,
    )

    ax.set_axisbelow(True)

    for side in [
        "top",
        "right",
        "left",
    ]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color(theme.axis)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        length=0,
        colors=theme.muted,
        labelsize=9.5,
        pad=6,
    )


def render_chart(
    daily: pd.DataFrame,
    totals: dict,
    window_days: int,
    theme: Theme,
    output_path: Path,
) -> None:
    """Render the relative-improvement graphic for one color scheme."""

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(
        figsize=(11.0, 5.8),
        dpi=200,
    )

    fig.patch.set_facecolor(theme.surface)

    ax = fig.add_axes(
        [0.072, 0.170, 0.733, 0.335]
    )

    dates = daily["game_date"]

    average = totals["relative_improvement"]

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    fig.text(
        0.072,
        0.955,
        f"Pitch prediction performance — last {window_days} days",
        color=theme.text_primary,
        fontsize=19,
        fontweight="bold",
        va="top",
    )

    fig.text(
        0.072,
        0.892,
        (
            f"{len(daily)} evaluated game dates"
            f"  ·  {daily['game_date'].min():%b %-d} – "
            f"{daily['game_date'].max():%b %-d, %Y}"
            f"  ·  {totals['pitcher_games']} pitcher-games"
            f"  ·  {totals['pitches']:,} pitches"
        ),
        color=theme.text_secondary,
        fontsize=11,
        va="top",
    )

    # --------------------------------------------------------
    # HERO FIGURE — the one number this README leads with
    # --------------------------------------------------------

    fig.text(
        0.072,
        0.815,
        f"+{average:.1%}",
        color=theme.text_primary,
        fontsize=46,
        fontweight="bold",
        va="top",
    )

    fig.text(
        0.072,
        0.668,
        "Relative improvement over the stratified baseline",
        color=theme.text_secondary,
        fontsize=12,
        va="top",
    )

    fig.text(
        0.072,
        0.615,
        (
            f"Ahead of the baseline in {totals['beat_baseline']} of "
            f"{totals['pitcher_games']} pitcher-games "
            f"({totals['beat_baseline'] / totals['pitcher_games']:.0%})"
            f"  ·  daily range +{daily['relative_improvement'].min():.1%}"
            f" to +{daily['relative_improvement'].max():.1%}"
        ),
        color=theme.muted,
        fontsize=10.5,
        va="top",
    )

    # --------------------------------------------------------
    # DAILY RELATIVE IMPROVEMENT
    # --------------------------------------------------------

    style_axes(ax, theme)

    ax.set_ylim(0.0, 0.86)

    ax.set_yticks(
        [0.0, 0.20, 0.40, 0.60]
    )

    ax.set_yticklabels(
        [
            "0%",
            "+20%",
            "+40%",
            "+60%",
        ]
    )

    ax.set_xlim(
        mdates.date2num(dates.min()) - 0.6,
        mdates.date2num(dates.max()) + 0.6,
    )

    ax.set_xticks(list(dates))

    ax.set_xticklabels(
        [
            f"{date:%b %-d}"
            for date in dates
        ]
    )

    ax.set_title(
        "Daily relative improvement over baseline",
        loc="left",
        color=theme.text_secondary,
        fontsize=11.5,
        pad=14,
    )

    # Draw after limits are fixed: corner radius is measured in pixels.
    fig.canvas.draw()

    for date, value in zip(
        dates,
        daily["relative_improvement"],
    ):
        rounded_column(
            ax,
            mdates.date2num(date),
            0.30,
            float(value),
            color=theme.series,
        )

    ax.axhline(
        average,
        color=theme.muted,
        linewidth=1.0,
        zorder=2,
    )

    ax.annotate(
        f"{window_days}-day average",
        xy=(
            ax.get_xlim()[1],
            average,
        ),
        xytext=(8, 5),
        textcoords="offset points",
        color=theme.text_secondary,
        fontsize=9.5,
        va="bottom",
        ha="left",
        annotation_clip=False,
    )

    # Direct-label the peak only; the average rule and the axis carry the rest.
    peak = int(daily["relative_improvement"].idxmax())

    ax.annotate(
        f"+{daily['relative_improvement'].iloc[peak]:.0%}",
        xy=(
            dates.iloc[peak],
            daily["relative_improvement"].iloc[peak],
        ),
        xytext=(0, 9),
        textcoords="offset points",
        color=theme.text_primary,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    fig.text(
        0.072,
        0.030,
        (
            "Relative improvement = (model accuracy − baseline accuracy) / "
            "baseline accuracy, pitch-weighted.\n"
            "Postgame replay of completed games; each pitcher-game is scored "
            "with a model frozen before first pitch."
        ),
        color=theme.muted,
        fontsize=9,
        linespacing=1.6,
        va="bottom",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        facecolor=theme.surface,
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# MARKDOWN
# ============================================================


def build_markdown(
    daily: pd.DataFrame,
    totals: dict,
    window_days: int,
) -> str:
    """Build the README results block."""

    lines: list[str] = []

    lines.append(
        "The headline metric is **relative improvement over baseline** — how much "
        "further the model gets than a stratified baseline drawing from the same "
        "pitcher's historical pitch mix:"
    )

    lines.append("")

    lines.append("```text")

    lines.append(
        "relative improvement = (model accuracy − baseline accuracy) / baseline accuracy"
    )

    lines.append("```")

    lines.append("")

    lines.append(
        "Results come from automated postgame replay of every eligible MLB starting "
        "pitcher, pitch-weighted across all pitcher-games in the window."
    )

    lines.append("")

    lines.append(
        "<p align=\"center\">\n"
        "  <picture>\n"
        "    <source media=\"(prefers-color-scheme: dark)\" "
        "srcset=\"Docs/assets/recent_performance_dark.png\">\n"
        "    <img alt=\"Relative improvement over baseline, last 14 days: "
        "+58.6% overall, shown as daily columns against the period average\" "
        "src=\"Docs/assets/recent_performance_light.png\" width=\"900\">\n"
        "  </picture>\n"
        "</p>"
    )

    lines.append("")

    lines.append(
        f"**Trailing {window_days} days** · "
        f"{len(daily)} evaluated game dates "
        f"({daily['game_date'].min():%B %-d} – "
        f"{daily['game_date'].max():%B %-d, %Y}) · "
        f"{totals['pitcher_games']} pitcher-games · "
        f"{totals['pitchers']} pitchers · "
        f"{totals['pitches']:,} pitches"
    )

    lines.append("")

    lines.append(
        "| Game date | Pitcher-games | Pitches | Relative improvement over baseline |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )

    for _, row in daily.iterrows():
        lines.append(
            "| {date} | {games} | {pitches:,} | +{lift:.1%} |".format(
                date=f"{row['game_date']:%b %-d}",
                games=int(row["pitcher_games"]),
                pitches=int(row["pitches"]),
                lift=row["relative_improvement"],
            )
        )

    lines.append(
        "| **{window}-day total** | **{games}** | **{pitches:,}** | "
        "**+{lift:.1%}** |".format(
            window=window_days,
            games=totals["pitcher_games"],
            pitches=totals["pitches"],
            lift=totals["relative_improvement"],
        )
    )

    lines.append("")

    lines.append(
        f"The model finished ahead of the baseline in "
        f"{totals['beat_baseline']} of {totals['pitcher_games']} pitcher-games "
        f"({totals['beat_baseline'] / totals['pitcher_games']:.0%})."
    )

    return "\n".join(lines)


def update_readme(
    readme_path: Path,
    block: str,
) -> None:
    """Replace the marked results block in the README."""

    text = readme_path.read_text()

    if (
        START_MARKER not in text
        or END_MARKER not in text
    ):
        raise SystemExit(
            f"README is missing the {START_MARKER} / {END_MARKER} markers."
        )

    head, remainder = text.split(
        START_MARKER,
        1,
    )

    _, tail = remainder.split(
        END_MARKER,
        1,
    )

    readme_path.write_text(
        f"{head}{START_MARKER}\n\n{block}\n\n{END_MARKER}{tail}"
    )


# ============================================================
# ENTRY POINT
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the README results section.",
    )

    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
    )

    parser.add_argument(
        "--history",
        type=Path,
        default=HISTORY_PATH,
    )

    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Only regenerate the images.",
    )

    args = parser.parse_args()

    window, _, _ = load_window(
        args.history,
        args.window_days,
    )

    daily = daily_frame(window)
    totals = pooled_totals(window)

    for theme, filename in [
        (LIGHT, "recent_performance_light.png"),
        (DARK, "recent_performance_dark.png"),
    ]:
        render_chart(
            daily,
            totals,
            args.window_days,
            theme,
            ASSET_DIR / filename,
        )

        print(f"wrote {ASSET_DIR / filename}")

    if not args.skip_readme:
        update_readme(
            README_PATH,
            build_markdown(
                daily,
                totals,
                args.window_days,
            ),
        )

        print(f"updated {README_PATH}")


if __name__ == "__main__":
    main()
