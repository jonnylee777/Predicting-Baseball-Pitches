from __future__ import annotations

import unittest

from pitch_prediction.live.mapping import (
    CALL_CODE_TO_DESCRIPTION,
    DESCRIPTION_TO_TYPE,
    TranslationWarnings,
    batted_ball_type,
    hit_location,
    launch_speed_angle,
    pitch_description,
    pitch_type_code,
    plate_appearance_event,
)


class CallCodeTranslationTests(unittest.TestCase):
    def test_every_call_code_maps_to_a_known_pitch_type(self) -> None:
        for code, description in CALL_CODE_TO_DESCRIPTION.items():
            self.assertIn(
                description,
                DESCRIPTION_TO_TYPE,
                f"call code {code} maps to unknown description {description}",
            )
            self.assertIn(DESCRIPTION_TO_TYPE[description], {"B", "S", "X"})

    def test_observed_live_call_codes_translate_to_savant_descriptions(self) -> None:
        warnings = TranslationWarnings()
        expected = {
            "B": ("ball", "B"),
            "*B": ("blocked_ball", "B"),
            "C": ("called_strike", "S"),
            "S": ("swinging_strike", "S"),
            "W": ("swinging_strike_blocked", "S"),
            "F": ("foul", "S"),
            "T": ("foul_tip", "S"),
            "L": ("foul_bunt", "S"),
            "X": ("hit_into_play", "X"),
            "D": ("hit_into_play", "X"),
            "E": ("hit_into_play", "X"),
            "H": ("hit_by_pitch", "B"),
        }
        for code, (description, pitch_type) in expected.items():
            self.assertEqual(pitch_description(code, warnings), description)
            self.assertEqual(pitch_type_code(description), pitch_type)
        self.assertTrue(warnings.is_empty())

    def test_unknown_call_code_is_reported_rather_than_silently_dropped(self) -> None:
        warnings = TranslationWarnings()
        self.assertIsNone(pitch_description("ZZ", warnings))
        self.assertFalse(warnings.is_empty())
        self.assertIn("ZZ", warnings.unknown_call_codes)


class BattedBallTranslationTests(unittest.TestCase):
    def test_bunt_trajectories_collapse_into_savant_categories(self) -> None:
        warnings = TranslationWarnings()
        self.assertEqual(batted_ball_type("bunt_grounder", warnings), "ground_ball")
        self.assertEqual(batted_ball_type("bunt_popup", warnings), "popup")
        self.assertEqual(batted_ball_type("line_drive", warnings), "line_drive")
        self.assertTrue(warnings.is_empty())

    def test_gap_hit_locations_become_missing(self) -> None:
        # Savant records the fielder credited with the ball, so the feed's gap
        # codes have no equivalent and must not be coerced to one fielder.
        warnings = TranslationWarnings()
        self.assertEqual(hit_location("8", warnings), 8.0)
        self.assertIsNone(hit_location("78", warnings))
        self.assertIsNone(hit_location("89", warnings))
        self.assertEqual(warnings.unmapped_hit_locations, {"78", "89"})

    def test_base_running_events_are_not_plate_appearance_results(self) -> None:
        warnings = TranslationWarnings()
        self.assertEqual(plate_appearance_event("strikeout", warnings), "strikeout")
        self.assertEqual(plate_appearance_event("field_out", warnings), "field_out")
        self.assertIsNone(plate_appearance_event("caught_stealing_2b", warnings))
        self.assertIsNone(plate_appearance_event("pickoff_1b", warnings))


class LaunchSpeedAngleTests(unittest.TestCase):
    def test_lookup_recovers_savant_classes(self) -> None:
        # Savant's launch_speed_angle is a deterministic function of exit velo
        # and launch angle; the grid reproduces it to 99.8% on held-out data.
        self.assertEqual(launch_speed_angle(107.1, 28.0), 6.0)
        self.assertEqual(launch_speed_angle(40.0, -20.0), 1.0)
        for speed, angle in ((95.0, 10.0), (70.0, 45.0), (85.0, 2.0)):
            self.assertIn(launch_speed_angle(speed, angle), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

    def test_missing_inputs_yield_missing_class(self) -> None:
        self.assertIsNone(launch_speed_angle(None, 20.0))
        self.assertIsNone(launch_speed_angle(90.0, None))
        self.assertIsNone(launch_speed_angle(float("nan"), 20.0))


if __name__ == "__main__":
    unittest.main()
