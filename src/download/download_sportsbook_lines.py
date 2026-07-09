import os
import requests
import pandas as pd

def download_sportsbook_lines():
    os.makedirs("data/odds", exist_ok=True)

    api_key = os.getenv("ODDS_API_KEY")

    if not api_key:
        raise ValueError("Missing ODDS_API_KEY. Add your API key first.")

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"

    params = {
        "apiKey": api_key
    }

    events = requests.get(url, params=params).json()

    rows = []

    for event in events:
        rows.append({
            "event_id": event.get("id"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "commence_time": event.get("commence_time")
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/odds/mlb_events.csv", index=False)

    return df

if __name__ == "__main__":
    print(download_sportsbook_lines())
