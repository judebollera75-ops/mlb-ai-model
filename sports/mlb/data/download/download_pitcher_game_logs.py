import os
import requests
import pandas as pd


def download_game_logs(target_date="2026-07-09"):

    os.makedirs("data/game_logs", exist_ok=True)

    pitchers = pd.read_csv(f"data/pitchers/{target_date}.csv")

    rows = []

    for _, p in pitchers.iterrows():

        pitcher_id = int(p["pitcher_id"])

        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"

        params = {
            "stats": "gameLog",
            "group": "pitching",
            "season": "2026"
        }

        data = requests.get(url, params=params).json()

        splits = data.get("stats", [{}])[0].get("splits", [])

        for game in splits[:5]:

            stat = game["stat"]

            rows.append({

                "pitcher_id": pitcher_id,
                "pitcher_name": p["pitcher_name"],
                "team": p["team"],

                "game_date": game["date"],

                "innings": stat.get("inningsPitched"),
                "strikeouts": stat.get("strikeOuts"),
                "walks": stat.get("baseOnBalls"),
                "hits": stat.get("hits"),
                "earned_runs": stat.get("earnedRuns"),
                "home_runs": stat.get("homeRuns"),
                "era": stat.get("era")

            })

    df = pd.DataFrame(rows)

    df.to_csv(
        f"data/game_logs/{target_date}.csv",
        index=False
    )

    return df


if __name__ == "__main__":
    print(download_game_logs())
