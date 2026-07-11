from datetime import date
from pathlib import Path

import pandas as pd
import requests


OUTPUT_DIRECTORY = Path("data/schedules")


def download_schedule(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    url = "https://statsapi.mlb.com/api/v1/schedule"

    params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": "probablePitcher",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    games = []

    for schedule_date in data.get("dates", []):
        actual_date = schedule_date.get(
            "date",
            target_date,
        )

        for game in schedule_date.get("games", []):
            games.append(
                {
                    "date": actual_date,
                    "game_id": game.get("gamePk"),
                    "away_team": (
                        game.get("teams", {})
                        .get("away", {})
                        .get("team", {})
                        .get("name")
                    ),
                    "home_team": (
                        game.get("teams", {})
                        .get("home", {})
                        .get("team", {})
                        .get("name")
                    ),
                    "away_pitcher": (
                        game.get("teams", {})
                        .get("away", {})
                        .get("probablePitcher", {})
                        .get("fullName")
                    ),
                    "home_pitcher": (
                        game.get("teams", {})
                        .get("home", {})
                        .get("probablePitcher", {})
                        .get("fullName")
                    ),
                    "status": (
                        game.get("status", {})
                        .get("detailedState")
                    ),
                }
            )

    schedule = pd.DataFrame(
        games,
        columns=[
            "date",
            "game_id",
            "away_team",
            "home_team",
            "away_pitcher",
            "home_pitcher",
            "status",
        ],
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    schedule.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(schedule)} games "
        f"to {output_path}"
    )

    if schedule.empty:
        print(
            f"No MLB games were found for {target_date}."
        )
    else:
        print(schedule.to_string(index=False))

    return schedule


if __name__ == "__main__":
    download_schedule()
