import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_KEY = os.environ.get("ODDS_API_KEY")
SPORT = "baseball_mlb"
OUTPUT_PATH = Path("data/platform_lines.csv")

API_MARKETS = [
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

OUTPUT_COLUMNS = [
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


def fetch_json(url: str, params: dict[str, Any]) -> dict | list:
    response = requests.get(url, params=params, timeout=30)

    remaining_requests = response.headers.get("x-requests-remaining")
    used_requests = response.headers.get("x-requests-used")

    if remaining_requests is not None:
        print(
            f"API requests used: {used_requests}; "
            f"remaining: {remaining_requests}"
        )

    response.raise_for_status()
    return response.json()


def load_existing_props() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        existing = pd.read_csv(OUTPUT_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in OUTPUT_COLUMNS:
        if column not in existing.columns:
            existing[column] = None

    return existing[OUTPUT_COLUMNS].copy()


def save_props(props: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(props)} sportsbook prop rows to {OUTPUT_PATH}")

    if not props.empty:
        print(props.head(30).to_string(index=False))


def download_sportsbook_props() -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY is missing from the environment."
        )

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

    if not isinstance(events, list):
        raise RuntimeError(
            f"Unexpected events response type: {type(events).__name__}"
        )

    print(f"Found {len(events)} live or upcoming MLB events.")

    rows: list[dict[str, Any]] = []

    for event in events:
        event_id = event.get("id")

        if not event_id:
            continue

        print(
            f"\nChecking event: "
            f"{event.get('away_team')} at {event.get('home_team')}"
        )

        odds_url = (
            f"https://api.the-odds-api.com/v4/sports/"
            f"{SPORT}/events/{event_id}/odds"
        )

        # Request each market separately so one unavailable market
        # does not prevent the other markets from being collected.
        for api_market in API_MARKETS:
            try:
                odds_data = fetch_json(
                    odds_url,
                    {
                        "apiKey": API_KEY,
                        "regions": "us",
                        "markets": api_market,
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                    },
                )
            except requests.HTTPError as exc:
                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else "unknown"
                )

                print(
                    f"Skipped {api_market} for event {event_id}. "
                    f"HTTP status: {status_code}. Error: {exc}"
                )
                continue
            except requests.RequestException as exc:
                print(
                    f"Skipped {api_market} for event {event_id}: {exc}"
                )
                continue

            if not isinstance(odds_data, dict):
                print(
                    f"Unexpected response for {event_id}, "
                    f"{api_market}: {type(odds_data).__name__}"
                )
                continue

            commence_time = odds_data.get(
                "commence_time",
                event.get("commence_time"),
            )
            home_team = odds_data.get(
                "home_team",
                event.get("home_team"),
            )
            away_team = odds_data.get(
                "away_team",
                event.get("away_team"),
            )

            market_row_count = 0

            for bookmaker in odds_data.get("bookmakers", []):
                platform = bookmaker.get("title")

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")
                    normalized_market = MARKET_MAP.get(market_key)

                    if not normalized_market:
                        continue

                    for outcome in market.get("outcomes", []):
                        player = outcome.get("description")
                        line = outcome.get("point")
                        direction = outcome.get("name")

                        if not player or line is None:
                            continue

                        rows.append(
                            {
                                "event_id": event_id,
                                "commence_time": commence_time,
                                "platform": platform,
                                "player": player,
                                "market": normalized_market,
                                "direction": direction,
                                "line": line,
                                "sportsbook_odds": outcome.get("price"),
                                "home_team": home_team,
                                "away_team": away_team,
                            }
                        )

                        market_row_count += 1

            print(
                f"{api_market}: collected "
                f"{market_row_count} outcome rows."
            )

    props = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    if not props.empty:
        props["player"] = props["player"].astype(str).str.strip()
        props["market"] = props["market"].astype(str).str.strip()
        props["platform"] = props["platform"].astype(str).str.strip()
        props["direction"] = props["direction"].astype(str).str.strip()
        props["line"] = pd.to_numeric(
            props["line"],
            errors="coerce",
        )

        props = props.dropna(
            subset=[
                "player",
                "market",
                "platform",
                "line",
            ]
        )

        props = props.drop_duplicates(
            subset=[
                "event_id",
                "platform",
                "player",
                "market",
                "direction",
                "line",
            ],
            keep="first",
        )

        props = props.sort_values(
            [
                "commence_time",
                "platform",
                "player",
                "market",
                "direction",
            ],
            ascending=True,
        ).reset_index(drop=True)

        save_props(props)
        return props

    existing_props = load_existing_props()

    if not existing_props.empty:
        print(
            "\nWARNING: The API returned zero player props."
        )
        print(
            "The existing non-empty platform_lines.csv "
            "will be preserved instead of overwritten."
        )
        print(
            f"Preserved {len(existing_props)} existing rows."
        )
        return existing_props

    print(
        "\nWARNING: The API returned zero player props and "
        "there was no previous non-empty file to preserve."
    )

    empty_props = pd.DataFrame(columns=OUTPUT_COLUMNS)
    save_props(empty_props)

    return empty_props


if __name__ == "__main__":
    download_sportsbook_props()
