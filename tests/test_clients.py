from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock

from pitch_prediction.clients import MlbStatsClient


class MlbStatsClientTests(unittest.TestCase):
    def test_probable_starters_tracks_confirmed_and_missing_pitchers(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "dates": [
                {
                    "games": [
                        {
                            "gamePk": 123,
                            "officialDate": "2025-06-01",
                            "gameDate": "2025-06-01T18:00:00Z",
                            "status": {"detailedState": "Scheduled"},
                            "teams": {
                                "away": {
                                    "team": {"id": 10, "name": "Away"},
                                    "probablePitcher": {
                                        "id": 99,
                                        "fullName": "Test Pitcher",
                                    },
                                },
                                "home": {"team": {"id": 20, "name": "Home"}},
                            },
                        }
                    ]
                }
            ]
        }
        session = Mock()
        session.get.return_value = response

        result = MlbStatsClient(session=session).probable_starters(date(2025, 6, 1))

        self.assertEqual(result.game_count, 1)
        self.assertEqual([starter.pitcher_id for starter in result.starters], [99])
        self.assertEqual(len(result.missing_probables), 1)
        self.assertEqual(result.missing_probables[0].team_name, "Home")


if __name__ == "__main__":
    unittest.main()
