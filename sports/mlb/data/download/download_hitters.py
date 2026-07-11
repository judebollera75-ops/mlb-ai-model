from datetime import date
from pathlib import Path

import pandas as pd
import requests


SCHEDULE_DIRECTORY = Path("data/schedules")
OUTPUT_DIRECTORY = Path("data/hitters")


def download_hitters(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule_path = (
        SCHEDULE_DIRECTORY
        / f"{target_date}.csv"
    )

    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Missing schedule file: {schedule_path}"
        )

    schedule = pd.read_csv(schedule_path)

    required_columns = {
        "game_id",
        "away_team",
        "home_team",
        "status",
    }

    missing_columns = required_columns - set(schedule.columns)

    if missing_columns:
        raise KeyError(
            f"{schedule_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    for _, game in schedule.iterrows():
        game_id = pd.to_numeric(
            game.get("game_id"),
            errors="coerce",
        )

        if pd.isna(game_id):
            continue

        game_id = int(game_id)

        url = (
            f"https://statsapi.mlb.com/api/v1/"
            f"game/{game_id}/boxscore"
        )

        try:
            response = requests.get(
                url,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            print(
                f"Skipped game {game_id}: {exc}"
            )
            continue

        teams = data.get("teams", {})

        for side in ["away", "home"]:
            team_data = teams.get(side, {})
            batting_order = team_data.get(
                "battingOrder",
                [],
            )
            players = team_data.get(
                "players",
                {},
            )

            team_name = game.get(
                f"{side}_team",
                "",
            )

            opponent = (
                game.get("home_team", "")
                if side == "away"
                else game.get("away_team", "")
            )

            for batting_position, player_id in enumerate(
                batting_order,
                start=1,
            ):
                player = players.get(
                    f"ID{player_id}",
                    {},
                )

                person = player.get(
                    "person",
                    {},
                )

                position = player.get(
                    "position",
                    {},
                )

                player_name = person.get(
                    "fullName"
                )

                if not player_name:
                    continue

                rows.append(
                    {
                        "date": target_date,
                        "game_id": game_id,
                        "team": team_name,
                        "opponent": opponent,
                        "side": side,
                        "player_id": player_id,
                        "player_name": player_name,
                        "batting_order": batting_position,
                        "position": position.get(
                            "abbreviation"
                        ),
                        "status": game.get("status"),
                    }
                )

    columns = [
        "date",
        "game_id",
        "team",
        "opponent",
        "side",
        "player_id",
        "player_name",
        "batting_order",
        "position",
        "status",
    ]

    hitters = pd.DataFrame(
        rows,
        columns=columns,
    )

    hitters = hitters.drop_duplicates(
        subset=[
            "game_id",
            "player_id",
        ],
        keep="first",
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    hitters.to_csv(
        output_path,
        index=False,
    )

    if hitters.empty:
        print(
            f"No confirmed batting orders found "
            f"for {target_date}."
        )
        print(
            "Try again closer to first pitch."
        )
    else:
        print(
            f"Saved {len(hitters)} hitters "
            f"to {output_path}"
        )
        print(
            hitters.head(30).to_string(
                index=False
            )
        )

    return hitters


if __name__ == "__main__":
    download_hitters()
