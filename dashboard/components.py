"""Presentation markup for the live Gameday view.

These functions return HTML strings and touch no Streamlit APIs, so the exact
markup that ships can be unit-tested and rendered to a static page for visual
review. ``dashboard/live.py`` wraps each one in ``st.markdown``.

Colors come from the project's validated palette. The probability chart uses
the emphasis form -- one accent hue for the predicted pitch, de-emphasis gray
for the rest -- so it needs no legend and no categorical CVD pair. That pair
clears CVD separation at dE 15.9 and 3:1 contrast in both modes. Status colors
are reserved and always ship with a dot and a word, never color alone: good and
critical sit only dE 4.1 apart under deuteranopia, so hue cannot carry that
distinction by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

THEME_CSS = """
<style>
.viz {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --accent:         #2a78d6;
  --deemphasis:     #898781;
  --good:           #0ca30c;
  --critical:       #d03b3b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --accent:         #3987e5;
  }
}
:root[data-theme="dark"] .viz, .viz[data-theme="dark"] {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #898781;
  --baseline:       #383835;
  --border:         rgba(255,255,255,0.10);
  --accent:         #3987e5;
}

.viz-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}

.tile-label {
  font-size: 0.74rem; letter-spacing: 0.02em;
  color: var(--text-muted); margin-bottom: 2px;
}
.tile-value {
  font-size: 1.5rem; font-weight: 600; color: var(--text-primary);
  line-height: 1.15;
}
.tile-sub { font-size: 0.74rem; color: var(--text-secondary); }

/* Hero figure: exactly one per view -- the pitch being predicted. */
.hero-label { font-size: 0.78rem; color: var(--text-muted); }
.hero-value {
  font-size: 3.6rem; font-weight: 600; line-height: 1.05;
  color: var(--text-primary); letter-spacing: -0.01em;
}
.hero-note { font-size: 0.86rem; color: var(--text-secondary); }

/* Bars: 20px thick (under the 24px cap), 4px rounded data-end, square at the
   baseline, 2px surface gap between neighbours. Values ride the tips, which
   is why the chart carries no axis. */
.bar-row { display: flex; align-items: center; margin-bottom: 2px; }
.bar-key {
  width: 46px; flex: 0 0 46px;
  font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}
.bar-track { flex: 1 1 auto; height: 20px; position: relative; }
.bar-fill { height: 20px; border-radius: 0 4px 4px 0; background: var(--deemphasis); }
.bar-fill.is-predicted { background: var(--accent); }
.bar-value {
  position: absolute; top: 0; height: 20px; line-height: 20px;
  font-size: 0.76rem; color: var(--text-secondary);
  font-variant-numeric: tabular-nums; padding-left: 6px; white-space: nowrap;
}

.pill { display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.82rem; font-weight: 600; }
.pill-dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 9px; }
.pill.good .pill-dot { background: var(--good); }
.pill.bad  .pill-dot { background: var(--critical); }
.pill.good { color: var(--good); }
.pill.bad  { color: var(--critical); }

.score-team { font-size: 0.95rem; color: var(--text-secondary); }
.score-num  { font-size: 1.6rem; font-weight: 600; color: var(--text-primary);
              font-variant-numeric: tabular-nums; }
.count-dots { display: flex; gap: 4px; align-items: center; }
.dot { width: 10px; height: 10px; border-radius: 50%;
       border: 1px solid var(--baseline); }
.dot.on { background: var(--text-primary); border-color: var(--text-primary); }
.meta { font-size: 0.78rem; color: var(--text-muted); }
</style>
"""


@dataclass
class GameView:
    away: str
    home: str
    away_score: int
    home_score: int
    inning: int
    topbot: str
    balls: int
    strikes: int
    outs: int
    status: str
    on_1b: bool
    on_2b: bool
    on_3b: bool


def dots_html(count: int, total: int) -> str:
    return "".join(
        f'<span class="dot{" on" if index < count else ""}"></span>'
        for index in range(total)
    )


def diamond_html(view: GameView) -> str:
    """Base occupancy. Filled bases use primary ink, never a series color."""

    def base(occupied: bool, x: int, y: int) -> str:
        fill = "var(--text-primary)" if occupied else "none"
        return (
            f'<rect x="{x}" y="{y}" width="13" height="13" rx="2" '
            f'transform="rotate(45 {x + 6.5} {y + 6.5})" '
            f'fill="{fill}" stroke="var(--baseline)" stroke-width="1.5"/>'
        )

    return (
        '<svg width="76" height="62" viewBox="0 0 76 62" '
        'aria-label="Runners on base">'
        + base(view.on_2b, 31, 6)
        + base(view.on_3b, 9, 28)
        + base(view.on_1b, 53, 28)
        + "</svg>"
    )


def scoreboard_html(view: GameView) -> str:
    return f"""
    <div class="viz-card" style="display:flex;align-items:center;gap:26px;
         flex-wrap:wrap;">
      <div style="min-width:104px;">
        <div class="score-team">{view.away}</div>
        <div class="score-num">{view.away_score}</div>
      </div>
      <div style="min-width:104px;">
        <div class="score-team">{view.home}</div>
        <div class="score-num">{view.home_score}</div>
      </div>
      <div style="min-width:92px;">
        <div class="tile-label">Inning</div>
        <div class="tile-value" style="font-size:1.15rem;">
          {view.topbot} {view.inning}</div>
      </div>
      <div>{diamond_html(view)}</div>
      <div style="min-width:150px;">
        <div class="count-dots" style="margin-bottom:4px;">{dots_html(view.balls, 3)}
          <span class="meta" style="margin-left:6px;">balls</span></div>
        <div class="count-dots" style="margin-bottom:4px;">{dots_html(view.strikes, 2)}
          <span class="meta" style="margin-left:6px;">strikes</span></div>
        <div class="count-dots">{dots_html(view.outs, 3)}
          <span class="meta" style="margin-left:6px;">outs</span></div>
      </div>
      <div style="margin-left:auto;text-align:right;min-width:110px;">
        <div class="tile-label">Status</div>
        <div class="tile-sub">{view.status}</div>
      </div>
    </div>
    """


def tile_html(label: str, value: str, sub: str) -> str:
    return (
        f'<div class="viz-card"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-sub">{sub}</div></div>'
    )


def probability_bars_html(
    probabilities: dict[str, float], predicted: str
) -> str:
    """Emphasis bars: the predicted pitch in the accent, the rest recede.

    Bars are scaled to the largest probability so the leading bar always spans
    the track. Values are labelled at the tip, which replaces an axis.
    """

    rows = sorted(probabilities.items(), key=lambda item: -item[1])
    top = rows[0][1] if rows else 0.0
    parts = []
    for name, probability in rows:
        width = (probability / top * 100) if top > 0 else 0.0
        # Keep the tip label inside the card when a bar nearly fills the track.
        label_style = (
            f"right:0;padding-left:0;padding-right:6px;color:var(--surface-1);"
            if width > 88
            else f"left:{width:.1f}%;"
        )
        parts.append(
            f'<div class="bar-row"><div class="bar-key">{name}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill{" is-predicted" if name == predicted else ""}"'
            f' style="width:{width:.1f}%;"></div>'
            f'<div class="bar-value" style="{label_style}">{probability:.0%}</div>'
            f"</div></div>"
        )
    return "".join(parts)


def prediction_html(
    *,
    predicted: str | None,
    confidence: float | None,
    probabilities: dict[str, float] | None,
    batter: str | None,
    balls: int | None,
    strikes: int | None,
    pitch_number: int | None,
) -> str:
    if predicted is None:
        return (
            '<div class="viz-card">'
            '<div class="hero-label">Next pitch</div>'
            '<div class="hero-value" style="color:var(--text-muted);">&ndash;&ndash;</div>'
            '<div class="hero-note">Waiting for the pitcher to face a batter.</div>'
            "</div>"
        )
    return f"""
    <div class="viz-card">
      <div class="hero-label">Next pitch &middot; {batter} &middot; {balls}-{strikes}</div>
      <div class="hero-value">{predicted}</div>
      <div class="hero-note" style="margin-bottom:12px;">
        {confidence:.0%} confidence &middot; pitch {pitch_number} of the game</div>
      <div class="tile-label">Model probability by pitch type</div>
      {probability_bars_html(probabilities or {}, predicted)}
    </div>
    """


def last_pitch_html(
    *,
    batter: str | None,
    predicted: str | None,
    actual: str | None,
    correct: bool | None,
    timing: str | None,
    lead_seconds: float | None,
) -> str:
    if predicted is None:
        return (
            '<div class="viz-card"><div class="tile-label">Last pitch</div>'
            '<div class="tile-sub">No pitch scored yet.</div></div>'
        )
    good = bool(correct)
    pill = (
        f'<span class="pill {"good" if good else "bad"}">'
        f'<span class="pill-dot"></span>{"Correct" if good else "Missed"}</span>'
    )
    if timing == "late":
        lead = "Arrived after the pitch, so it is excluded from accuracy"
    elif lead_seconds is not None:
        lead = f"Predicted {lead_seconds:.1f}s before the pitch"
    else:
        lead = "Timing unknown"
    return f"""
    <div class="viz-card">
      <div class="tile-label">Last pitch &middot; {batter}</div>
      <div style="display:flex;gap:26px;align-items:baseline;margin:6px 0 4px;">
        <div><div class="tile-label">Predicted</div>
             <div class="tile-value">{predicted}</div></div>
        <div><div class="tile-label">Actual</div>
             <div class="tile-value">{actual or "?"}</div></div>
        <div style="margin-left:auto;text-align:right;">{pill}</div>
      </div>
      <div class="tile-sub">{lead}</div>
    </div>
    """
