import os
import requests
import pandas as pd


TARGET_DATE = "2026-07-09"


def download_hitters(target_date=TARGET_DATE):
    os.makedirs("data/hitters", exist_ok=True)

    schedule = pd.read_csv(f"data/schedules/{target_date}.csv")
    rows = []

    for _, game in schedule.iterrows():
        game_id = int(game["game_id"])

        url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        teams = data.get("teams", {})

        for side in ["away", "home"]:
            team_data = teams.get(side, {})
            team_name = game.get(f"{side}_team")

            batting_order = team_data.get("battingOrder", [])
            players = team_data.get("players", {})

            for batting_position, player_id in enumerate(
                batting_order,
                start=1
            ):
                player_key = f"ID{player_id}"
                player = players.get(player_key, {})

                person = player.get("person", {})
                position = player.get("position", {})

                rows.append({
                    "date": target_date,
                    "game_id": game_id,
                    "team": team_name,
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

    hitters = pd.DataFrame(rows)

    output_path = f"data/hitters/{target_date}.csv"
    hitters.to_csv(output_path, index=False)

    print(f"Saved {len(hitters)} hitters to {output_path}")
    print(hitters.head(30).to_string(index=False))

    return hitters


if __name__ == "__main__":
    download_hitters()
