import os
import time
import requests
import pandas as pd
from datetime import date, timedelta

def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

def download_historical_pitcher_starts(start_date="2026-07-01", end_date="2026-07-09"):
    os.makedirs("data/historical", exist_ok=True)

    rows = []

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    for day in daterange(start, end):
        day_str = str(day)
        print("Downloading", day_str)

        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "sportId": 1,
            "date": day_str,
            "hydrate": "probablePitcher,linescore"
        }

        data = requests.get(url, params=params).json()

        for d in data.get("dates", []):
            for g in d.get("games", []):
                game_id = g["gamePk"]

                for side in ["away", "home"]:
                    pitcher = g["teams"][side].get("probablePitcher")
                    if not pitcher:
                        continue

                    opponent_side = "home" if side == "away" else "away"

                    rows.append({
                        "date": day_str,
                        "game_id": game_id,
                        "team": g["teams"][side]["team"]["name"],
                        "opponent": g["teams"][opponent_side]["team"]["name"],
                        "side": side,
                        "pitcher_id": pitcher["id"],
                        "pitcher_name": pitcher["fullName"],
                        "status": g["status"]["detailedState"],
                    })

        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df.to_csv("data/historical/pitcher_starts.csv", index=False)
    return df

if __name__ == "__main__":
    print(download_historical_pitcher_starts())
