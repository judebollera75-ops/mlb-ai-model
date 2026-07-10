import requests
import pandas as pd

def download_pitchers(target_date="2026-07-10"):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": "probablePitcher"
    }

    data = requests.get(url, params=params).json()

    rows = []

    for d in data.get("dates", []):
        for g in d.get("games", []):
            for side in ["away", "home"]:
                pitcher = g["teams"][side].get("probablePitcher")

                if pitcher:
                    rows.append({
                        "date": target_date,
                        "game_id": g["gamePk"],
                        "team": g["teams"][side]["team"]["name"],
                        "pitcher_id": pitcher["id"],
                        "pitcher_name": pitcher["fullName"],
                        "side": side
                    })

    df = pd.DataFrame(rows)
    df.to_csv(f"data/pitchers/{target_date}.csv", index=False)
    return df

if __name__ == "__main__":
    print(download_pitchers())
