from datetime import date
from pathlib import Path

import pandas as pd
import requests


OUTPUT_DIRECTORY = Path("data/pitchers")


def download_pitchers(target_date=None):
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

    rows = []

    for schedule_date in data.get("dates", []):
        actual_date = schedule_date.get(
            "date",
            target_date,
        )

        for game in schedule_date.get("games", []):

            for side in ["away", "home"]:

                pitcher = (
                    game.get("teams", {})
                    .get(side, {})
                    .get("probablePitcher")
                )

                if pitcher is None:
                    continue

                rows.append(
                    {
                        "date": actual_date,
                        "game_id": game.get("gamePk"),
                        "team": (
                            game.get("teams", {})
                            .get(side, {})
                            .get("team", {})
                            .get("name")
                        ),
                        "pitcher_id": pitcher.get("id"),
                        "pitcher_name": pitcher.get("fullName"),
                        "side": side,
                    }
                )

    pitchers = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    pitchers.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(pitchers)} pitchers to {output_path}"
    )

    if not pitchers.empty:
        print(pitchers.to_string(index=False))

    return pitchers


if __name__ == "__main__":
    download_pitchers()
