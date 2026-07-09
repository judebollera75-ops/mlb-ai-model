import requests
import pandas as pd
from datetime import date

def download_schedule(target_date=None):
    if target_date is None:
        target_date = str(date.today())

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": "probablePitcher"
    }

    data = requests.get(url, params=params).json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append({
                "date": target_date,
                "game_id": g["gamePk"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_pitcher": g["teams"]["away"].get("probablePitcher", {}).get("fullName"),
                "home_pitcher": g["teams"]["home"].get("probablePitcher", {}).get("fullName"),
                "status": g["status"]["detailedState"]
            })

    df = pd.DataFrame(games)
    df.to_csv(f"data/schedules/{target_date}.csv", index=False)
    return df

if __name__ == "__main__":
    print(download_schedule("2026-07-09"))
