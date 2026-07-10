import os
import requests
import pandas as pd

def download_player_props():
    os.makedirs("data/odds", exist_ok=True)

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("Missing ODDS_API_KEY")

    events = pd.read_csv("data/odds/mlb_events.csv")

    rows = []

    for _, event in events.iterrows():
        event_id = event["event_id"]

        url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"

        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "pitcher_strikeouts",
            "oddsFormat": "american"
        }

        data = requests.get(url, params=params).json()

        for bookmaker in data.get("bookmakers", []):
            book = bookmaker.get("title")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "pitcher_strikeouts":
                    continue

                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "sportsbook": book,
                        "pitcher": outcome.get("description"),
                        "side": outcome.get("name"),
                        "line": outcome.get("point"),
                        "sportsbook_odds": outcome.get("price"),
                        "home_team": event["home_team"],
                        "away_team": event["away_team"],
                        "commence_time": event["commence_time"]
                    })

    df = pd.DataFrame(rows)
    df.to_csv("data/odds/player_props.csv", index=False)

    print(df.head(30))
    print("Rows:", len(df))

if __name__ == "__main__":
    download_player_props()
