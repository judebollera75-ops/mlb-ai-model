import os
from pathlib import Path

import pandas as pd
import requests


API_KEY = os.environ.get("ODDS_API_KEY")
SPORT = "baseball_mlb"
OUTPUT_PATH = Path("data/platform_lines.csv")

MARKETS = [
    "pitcher_strikeouts",
    "pitcher_outs",
    "batter_hits",
    "batter_total_bases",
]

MARKET_MAP = {
    "pitcher_strikeouts": "pitcher_strikeouts",
    "pitcher_outs": "pitcher_outs",
    "batter_hits": "hitter_hits",
    "batter_total_bases": "hitter_total_bases",
}


def fetch_json(url: str, params: dict) -> dict | list:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
def download_sportsbook_props() -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY is missing from the environment.")

    events_url = (
        f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    )

    events = fetch_json(
        events_url,
        {
            "apiKey": API_KEY,
            "dateFormat": "iso",
        },
    )

    rows = []

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        odds_url = (
            f"https://api.the-odds-api.com/v4/sports/"
            f"{SPORT}/events/{event_id}/odds"
        )

        try:
            odds_data = fetch_json(
                odds_url,
                {
                    "apiKey": API_KEY,
                    "regions": "us",
                    "markets": ",".join(MARKETS),
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
            )
        except requests.RequestException as exc:
            print(f"Skipped event {event_id}: {exc}")
            continue

        commence_time = odds_data.get("commence_time")
        home_team = odds_data.get("home_team")
        away_team = odds_data.get("away_team")

        for bookmaker in odds_data.get("bookmakers", []):
            platform = bookmaker.get("title")

            for market in bookmaker.get("markets", []):
                normalized_market = MARKET_MAP.get(market.get("key"))
                if not normalized_market:
                    continue

                for outcome in market.get("outcomes", []):
                    player = outcome.get("description")
                    line = outcome.get("point")

                    if not player or line is None:
                        continue

                    rows.append(
                        {
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "platform": platform,
                            "player": player,
                            "market": normalized_market,
                            "direction": outcome.get("name"),
                            "line": line,
                            "sportsbook_odds": outcome.get("price"),
                            "home_team": home_team,
                            "away_team": away_team,
                        }
                    )

    columns = [
        "event_id",
        "commence_time",
        "platform",
        "player",
        "market",
        "direction",
        "line",
        "sportsbook_odds",
        "home_team",
        "away_team",
    ]

    props = pd.DataFrame(rows, columns=columns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(props)} sportsbook prop rows to {OUTPUT_PATH}")

    if not props.empty:
        print(props.head(20).to_string(index=False))

    return props


if __name__ == "__main__":
    download_sportsbook_props()
