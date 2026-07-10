import os
import requests
import pandas as pd

TARGET_DATE = "2026-07-10"


def download_hitters(target_date=TARGET_DATE):
    os.makedirs("data/hitters", exist_ok=True)

    schedule = pd.read_csv(f"data/schedules/{target_date}.csv")
    rows = []

    for _, game in schedule.iterrows():
        game_id = int(game["game_id"])

        url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            print(f"Skipped game {game_id}: {exc}")
            continue

        teams = data.get("teams", {})

        for side in ["away", "home"]:
            team_data = teams.get(side, {})
            batting_order = team_data.get("battingOrder", [])
            players = team_data.get("players", {})

            for batting_position, player_id in enumerate(
                batting_order,
                start=1
            ):
                player = players.get(f"ID{player_id}", {})
                person = player.get("person", {})
                position = player.get("position", {})

                rows.append({
                    "date": target_date,
                    "game_id": game_id,
                    "team": game.get(f"{side}_team"),
                    "opponent": (
                        game.get("home_team")
                        if side == "away"
                        else game.get("away_team")
                    ),
                    "side": side,
                    "player_id": player_id,
                    "player_name": person.get("fullName"),
                    "batting_order": batting_position,
                    "position": position.get("abbreviation"),
                    "status": game.get("status"),
                })

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

    hitters = pd.DataFrame(rows, columns=columns)

    output_path = f"data/hitters/{target_date}.csv"
    hitters.to_csv(output_path, index=False)

    if hitters.empty:
        print(f"No confirmed batting orders found for {target_date}.")
        print("Try again closer to first pitch.")
    else:
        print(f"Saved {len(hitters)} hitters to {output_path}")
        print(hitters.head(30).to_string(index=False))

    return hitters


if __name__ == "__main__":
    download_hitters()
