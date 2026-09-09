"""Feature sets for live serving.

The production KG4 frame includes the previous pitch's Statcast measurements.
MLB publishes those about 19 seconds after the pitch, while pitches arrive
about 20 seconds apart, so a predictor that waits for them has roughly a second
of headroom and misses about half the time.

Excluding them leaves a frame whose remaining unknowns are *enumerable*: the
next count has at most three outcomes inside an at-bat, and the previous pitch
type is drawn from the pitcher's repertoire. A prediction can therefore be
computed for every candidate state before the current pitch is even thrown, and
the matching one selected once the feed catches up.

Measured cost of the exclusion, over 46 pitcher-games and 4,068 pitches: 2.4
points of relative improvement over baseline (+62.6% to +60.2%), which is not
statistically distinguishable from zero (paired p=0.15). Also dropping pitch
sequencing costs a further 3.3 points (p=0.02), so sequencing is kept and
enumerated instead.
"""

from __future__ import annotations

from ..feature_engineering import ROLLING_PITCH_TYPES

# The previous pitch's continuous measurements. Not enumerable, so they cannot
# be guessed ahead, and they arrive too late to wait for.
LATE_MEASUREMENT_COLUMNS = (
    "release_speed_of_prev_pitch",
    "zone_of_prev_pitch",
    "description_of_prev_pitch",
    "type_of_prev_pitch",
    "launch_speed_of_prev_pitch",
    "launch_angle_of_prev_pitch",
    "hit_distance_sc_of_prev_pitch",
)

# The previous plate appearance's batted-ball detail, late for the same reason.
LATE_AT_BAT_COLUMNS = (
    "events_of_prev_ab",
    "hit_location_of_prev_ab",
    "bb_type_of_prev_ab",
    "launch_speed_angle_of_prev_ab",
)

# Withhold these from training and prediction to get a model that can be run a
# pitch ahead.
PREDICT_AHEAD_EXCLUSIONS = LATE_MEASUREMENT_COLUMNS + LATE_AT_BAT_COLUMNS

# Sequencing features. Published just as late, but enumerable over the
# pitcher's repertoire, so they are kept and varied across candidates.
SEQUENCE_COLUMNS = ("pitch_type_of_prev_pitch",) + tuple(
    f"prev3_pitch_rate_{name}" for name in ROLLING_PITCH_TYPES
)
