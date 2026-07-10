import os
import requests
import pandas as pd

def download_pitcher_stats(target_date="2026-07-09", season="2026"):
    os.makedirs("data/pitcher_stats", exist_ok=True)

    pitchers = pd.read_csv(f"data/pitchers/{target_date}.csv")

    rows = []

    for _, p in pitchers.iterrows():
        pitcher_id = int(p["pitcher_id"])

        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        params = {
            "stats": "season",
            "group": "pitching",
            "season": season
        }

        data = requests.get(url, params=params).json()

        splits = data.get("stats", [{}])[0].get("splits", [])

        if not splits:
            continue

        stat = splits[0]["stat"]

        rows.append({
            "date": target_date,
            "season": season,
            "pitcher_id": pitcher_id,
            "pitcher_name": p["pitcher_name"],
            "team": p["team"],

            "games": stat.get("gamesPlayed"),
            "games_started": stat.get("gamesStarted"),
            "innings_pitched": stat.get("inningsPitched"),
            "era": stat.get("era"),
            "whip": stat.get("whip"),
            "strikeouts": stat.get("strikeOuts"),
            "walks": stat.get("baseOnBalls"),
            "hits_allowed": stat.get("hits"),
            "earned_runs": stat.get("earnedRuns"),
            "home_runs_allowed": stat.get("homeRuns"),
            "batters_faced": stat.get("battersFaced"),
            "wins": stat.get("wins"),
            "losses": stat.get("losses")
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"data/pitcher_stats/{target_date}.csv", index=False)

    return df

if __name__ == "__main__":
    print(download_pitcher_stats())
