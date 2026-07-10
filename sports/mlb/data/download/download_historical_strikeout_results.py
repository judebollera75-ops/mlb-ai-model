import os
import time
import requests
import pandas as pd

def download_historical_strikeout_results():
    os.makedirs("data/historical", exist_ok=True)

    starts = pd.read_csv("data/historical/pitcher_starts.csv")

    rows = []

    for _, row in starts.iterrows():
        pitcher_id = int(row["pitcher_id"])
        game_id = int(row["game_id"])

        print("Getting", row["pitcher_name"], row["date"])

        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        params = {
            "stats": "gameLog",
            "group": "pitching",
            "season": "2026"
        }

        data = requests.get(url, params=params).json()
        splits = data.get("stats", [{}])[0].get("splits", [])

        for g in splits:
            if int(g["game"]["gamePk"]) == game_id:
                stat = g["stat"]

                rows.append({
                    "date": row["date"],
                    "game_id": game_id,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": row["pitcher_name"],
                    "team": row["team"],
                    "opponent": row["opponent"],
                    "side": row["side"],
                    "actual_strikeouts": stat.get("strikeOuts"),
                    "actual_ip": stat.get("inningsPitched"),
                    "actual_walks": stat.get("baseOnBalls"),
                    "actual_hits": stat.get("hits"),
                    "actual_earned_runs": stat.get("earnedRuns")
                })

        time.sleep(0.1)

    df = pd.DataFrame(rows)
    df.to_csv("data/historical/strikeout_results.csv", index=False)

    return df

if __name__ == "__main__":
    print(download_historical_strikeout_results())
