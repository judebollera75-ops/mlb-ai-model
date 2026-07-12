"""Download and normalize current MLB player props from The Odds API.

Required environment variable:
    ODDS_API_KEY

Optional environment variable:
    MLB_TARGET_DATE=YYYY-MM-DD

Output:
    data/platform_lines.csv

The downloader only saves props for the requested MLB slate date. It preserves
separate rows for every sportsbook, side, player, market, and line so the
betting engine can compare exact platform availability.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_KEY = os.environ.get("ODDS_API_KEY")
SPORT = "baseball_mlb"
REGIONS = os.environ.get("ODDS_API_REGIONS", "us")
OUTPUT_PATH = Path("data/platform_lines.csv")
CENTRAL_TIME = ZoneInfo("America/Chicago")

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_PAUSE_SECONDS = 0.20
MAX_EVENT_AGE_MINUTES = 15

API_MARKETS = [
    "pitcher_strikeouts",
    "pitcher_outs",
    "batter_hits",
    "batter_total_bases",
    "batter_runs_scored",
    "batter_rbis",
    "batter_hits_runs_rbis",
    "batter_fantasy_score",
]

MARKET_MAP = {
    "pitcher_strikeouts": "pitcher_strikeouts",
    "pitcher_outs": "pitcher_outs",
    "batter_hits": "hitter_hits",
    "batter_total_bases": "hitter_total_bases",
    "batter_runs_scored": "hitter_runs",
    "batter_rbis": "hitter_rbis",
    "batter_hits_runs_rbis": "hitter_hits_runs_rbis",
    "batter_fantasy_score": "hitter_fantasy_score",
}

DIRECTION_MAP = {
    "over": "Over",
    "under": "Under",
    "more": "Over",
    "less": "Under",
    "yes": "Yes",
    "no": "No",
}

OUTPUT_COLUMNS = [
    "event_id",
    "event_date",
    "commence_time",
    "platform",
    "platform_key",
    "player",
    "market",
    "market_key",
    "direction",
    "line",
    "sportsbook_odds",
    "home_team",
    "away_team",
    "fetched_at",
    "source",
    "status",
]


def get_target_date() -> date:
    """Return the requested MLB slate date."""
    raw_value = os.environ.get("MLB_TARGET_DATE", date.today().isoformat())

    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc


def build_session() -> requests.Session:
    """Create a requests session with safe retry behavior."""
    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "mlb-ai-model/1.0",
        }
    )

    return session


def parse_utc_datetime(value: Any) -> datetime | None:
    """Parse an API ISO timestamp into an aware UTC datetime."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def event_local_date(commence_time: Any) -> date | None:
    """Convert an event timestamp into its Central Time calendar date."""
    parsed = parse_utc_datetime(commence_time)

    if parsed is None:
        return None

    return parsed.astimezone(CENTRAL_TIME).date()


def normalize_direction(value: Any) -> str | None:
    """Normalize provider side names."""
    if value is None or pd.isna(value):
        return None

    cleaned = str(value).strip()
    normalized = DIRECTION_MAP.get(cleaned.casefold())

    return normalized or cleaned.title()


def clean_text(value: Any) -> str | None:
    """Return stripped text or None."""
    if value is None or pd.isna(value):
        return None

    cleaned = str(value).strip()

    return cleaned or None


def american_odds_are_valid(value: Any) -> bool:
    """Validate American odds without rejecting plus-money prices."""
    try:
        odds = int(float(value))
    except (TypeError, ValueError):
        return False

    return odds != 0 and abs(odds) >= 100


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
) -> dict[str, Any] | list[Any]:
    """Fetch JSON and report API quota headers."""
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    print(
        "API usage — "
        f"used: {response.headers.get('x-requests-used')}, "
        f"remaining: {response.headers.get('x-requests-remaining')}, "
        f"last request cost: {response.headers.get('x-requests-last')}"
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_preview = response.text[:500]

        raise requests.HTTPError(
            f"{exc}. Response body: {response_preview}"
        ) from exc

    try:
        return response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError(
            f"The Odds API returned invalid JSON from {url}."
        ) from exc


def empty_props_frame() -> pd.DataFrame:
    """Return an empty frame using the production schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def load_existing_props() -> pd.DataFrame:
    """Read an existing platform-lines file without trusting stale rows."""
    if not OUTPUT_PATH.exists():
        return empty_props_frame()

    try:
        existing = pd.read_csv(OUTPUT_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return empty_props_frame()

    for column in OUTPUT_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA

    return existing[OUTPUT_COLUMNS].copy()


def existing_props_are_current(
    existing: pd.DataFrame,
    target_date: date,
) -> bool:
    """Only preserve a recent file for the same requested slate."""
    if existing.empty:
        return False

    existing_dates = pd.to_datetime(
        existing["event_date"],
        errors="coerce",
    ).dt.date

    if not existing_dates.eq(target_date).any():
        return False

    fetched_times = pd.to_datetime(
        existing["fetched_at"],
        errors="coerce",
        utc=True,
    ).dropna()

    if fetched_times.empty:
        return False

    newest_fetch = fetched_times.max().to_pydatetime()
    age = datetime.now(timezone.utc) - newest_fetch

    return age <= timedelta(minutes=MAX_EVENT_AGE_MINUTES)


def save_props(props: pd.DataFrame) -> None:
    """Save normalized platform lines atomically."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = OUTPUT_PATH.with_suffix(".tmp.csv")
    props.to_csv(temporary_path, index=False)
    temporary_path.replace(OUTPUT_PATH)

    print(f"\nSaved {len(props)} sportsbook rows to {OUTPUT_PATH}")

    if not props.empty:
        market_counts = (
            props.groupby("market", dropna=False)
            .size()
            .sort_values(ascending=False)
        )

        platform_counts = (
            props.groupby("platform", dropna=False)
            .size()
            .sort_values(ascending=False)
        )

        print("\nRows by market:")
        print(market_counts.to_string())

        print("\nRows by platform:")
        print(platform_counts.to_string())


def select_target_events(
    events: list[dict[str, Any]],
    target_date: date,
) -> list[dict[str, Any]]:
    """Keep only events belonging to the requested Central Time slate."""
    selected: list[dict[str, Any]] = []

    for event in events:
        commence_time = event.get("commence_time")
        local_date = event_local_date(commence_time)

        if local_date == target_date:
            selected.append(event)

    selected.sort(
        key=lambda item: str(item.get("commence_time", ""))
    )

    return selected


def outcome_to_row(
    *,
    event_id: str,
    commence_time: Any,
    bookmaker: dict[str, Any],
    market: dict[str, Any],
    outcome: dict[str, Any],
    home_team: Any,
    away_team: Any,
    fetched_at: str,
) -> dict[str, Any] | None:
    """Convert one provider outcome into the common platform schema."""
    market_key = clean_text(market.get("key"))
    normalized_market = MARKET_MAP.get(market_key or "")

    if normalized_market is None:
        return None

    player = clean_text(outcome.get("description"))
    platform = clean_text(bookmaker.get("title"))
    platform_key = clean_text(bookmaker.get("key"))
    direction = normalize_direction(outcome.get("name"))
    line = pd.to_numeric(outcome.get("point"), errors="coerce")
    odds = pd.to_numeric(outcome.get("price"), errors="coerce")

    if not player or not platform or not platform_key:
        return None

    if direction not in {"Over", "Under", "Yes", "No"}:
        return None

    if pd.isna(line):
        return None

    if pd.isna(odds) or not american_odds_are_valid(odds):
        return None

    parsed_commence_time = parse_utc_datetime(commence_time)

    if parsed_commence_time is None:
        return None

    local_event_date = parsed_commence_time.astimezone(
        CENTRAL_TIME
    ).date()

    return {
        "event_id": event_id,
        "event_date": local_event_date.isoformat(),
        "commence_time": parsed_commence_time.isoformat(),
        "platform": platform,
        "platform_key": platform_key,
        "player": player,
        "market": normalized_market,
        "market_key": market_key,
        "direction": direction,
        "line": float(line),
        "sportsbook_odds": int(float(odds)),
        "home_team": clean_text(home_team),
        "away_team": clean_text(away_team),
        "fetched_at": fetched_at,
        "source": "the_odds_api",
        "status": "open",
    }


def clean_props(
    props: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    """Apply final validation and duplicate protection."""
    if props.empty:
        return empty_props_frame()

    cleaned = props.copy()

    cleaned["event_date"] = pd.to_datetime(
        cleaned["event_date"],
        errors="coerce",
    ).dt.date

    cleaned["commence_time"] = pd.to_datetime(
        cleaned["commence_time"],
        errors="coerce",
        utc=True,
    )

    cleaned["fetched_at"] = pd.to_datetime(
        cleaned["fetched_at"],
        errors="coerce",
        utc=True,
    )

    cleaned["line"] = pd.to_numeric(
        cleaned["line"],
        errors="coerce",
    )

    cleaned["sportsbook_odds"] = pd.to_numeric(
        cleaned["sportsbook_odds"],
        errors="coerce",
    )

    required_columns = [
        "event_id",
        "event_date",
        "commence_time",
        "platform",
        "platform_key",
        "player",
        "market",
        "market_key",
        "direction",
        "line",
        "sportsbook_odds",
        "fetched_at",
    ]

    cleaned = cleaned.dropna(subset=required_columns)

    cleaned = cleaned.loc[
        cleaned["event_date"].eq(target_date)
    ].copy()

    current_time = datetime.now(timezone.utc)

    # Props for games that have already started are not actionable.
    cleaned = cleaned.loc[
        cleaned["commence_time"] > current_time
    ].copy()

    valid_markets = set(MARKET_MAP.values())
    cleaned = cleaned.loc[
        cleaned["market"].isin(valid_markets)
    ].copy()

    cleaned = cleaned.loc[
        cleaned["direction"].isin({"Over", "Under", "Yes", "No"})
    ].copy()

    cleaned = cleaned.loc[
        cleaned["sportsbook_odds"].apply(american_odds_are_valid)
    ].copy()

    # Keep the newest copy of each exact market side.
    cleaned = cleaned.sort_values(
        "fetched_at",
        ascending=True,
    )

    cleaned = cleaned.drop_duplicates(
        subset=[
            "event_id",
            "platform_key",
            "player",
            "market",
            "direction",
            "line",
        ],
        keep="last",
    )

    cleaned = cleaned.sort_values(
        [
            "commence_time",
            "platform",
            "player",
            "market",
            "line",
            "direction",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    cleaned["event_date"] = cleaned["event_date"].astype(str)
    cleaned["commence_time"] = cleaned[
        "commence_time"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cleaned["fetched_at"] = cleaned[
        "fetched_at"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return cleaned[OUTPUT_COLUMNS]


def download_sportsbook_props() -> pd.DataFrame:
    """Download current player props for the requested MLB slate."""
    if not API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY is missing from the environment."
        )

    target_date = get_target_date()
    fetched_at = datetime.now(timezone.utc).isoformat()
    existing_props = load_existing_props()
    session = build_session()

    print("=" * 70)
    print("Downloading current MLB sportsbook props")
    print(f"Slate date: {target_date.isoformat()}")
    print(f"Regions: {REGIONS}")
    print(f"Requested markets: {len(API_MARKETS)}")
    print("=" * 70)

    events_url = (
        f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    )

    events_response = fetch_json(
        session,
        events_url,
        {
            "apiKey": API_KEY,
            "dateFormat": "iso",
        },
    )

    if not isinstance(events_response, list):
        raise RuntimeError(
            "Unexpected events response type: "
            f"{type(events_response).__name__}"
        )

    events = [
        event
        for event in events_response
        if isinstance(event, dict)
    ]

    target_events = select_target_events(events, target_date)

    print(f"Upcoming MLB events returned: {len(events)}")
    print(f"Events on requested slate: {len(target_events)}")

    rows: list[dict[str, Any]] = []
    markets_parameter = ",".join(API_MARKETS)

    for event_number, event in enumerate(target_events, start=1):
        event_id = clean_text(event.get("id"))

        if not event_id:
            continue

        away_team = clean_text(event.get("away_team"))
        home_team = clean_text(event.get("home_team"))

        print(
            f"\n[{event_number}/{len(target_events)}] "
            f"{away_team} at {home_team}"
        )

        odds_url = (
            f"https://api.the-odds-api.com/v4/sports/"
            f"{SPORT}/events/{event_id}/odds"
        )

        try:
            odds_response = fetch_json(
                session,
                odds_url,
                {
                    "apiKey": API_KEY,
                    "regions": REGIONS,
                    "markets": markets_parameter,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
            )
        except requests.RequestException as exc:
            print(f"Skipped event {event_id}: {exc}")
            continue

        if not isinstance(odds_response, dict):
            print(
                f"Skipped event {event_id}: unexpected response "
                f"type {type(odds_response).__name__}"
            )
            continue

        commence_time = odds_response.get(
            "commence_time",
            event.get("commence_time"),
        )

        bookmakers = odds_response.get("bookmakers", [])

        if not isinstance(bookmakers, list):
            bookmakers = []

        event_row_count = 0

        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue

            markets = bookmaker.get("markets", [])

            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, dict):
                    continue

                outcomes = market.get("outcomes", [])

                if not isinstance(outcomes, list):
                    continue

                for outcome in outcomes:
                    if not isinstance(outcome, dict):
                        continue

                    row = outcome_to_row(
                        event_id=event_id,
                        commence_time=commence_time,
                        bookmaker=bookmaker,
                        market=market,
                        outcome=outcome,
                        home_team=odds_response.get(
                            "home_team",
                            home_team,
                        ),
                        away_team=odds_response.get(
                            "away_team",
                            away_team,
                        ),
                        fetched_at=fetched_at,
                    )

                    if row is not None:
                        rows.append(row)
                        event_row_count += 1

        print(
            f"Bookmakers returned: {len(bookmakers)} | "
            f"Valid prop rows: {event_row_count}"
        )

        time.sleep(REQUEST_PAUSE_SECONDS)

    raw_props = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    props = clean_props(raw_props, target_date)

    print(f"\nRaw rows collected: {len(raw_props)}")
    print(f"Validated current rows: {len(props)}")

    if not props.empty:
        save_props(props)
        return props

    if existing_props_are_current(existing_props, target_date):
        print(
            "\nWARNING: The API returned no current props. "
            "Preserving the existing file because it belongs to the "
            "same slate and was fetched within the last "
            f"{MAX_EVENT_AGE_MINUTES} minutes."
        )
        return existing_props

    print(
        "\nWARNING: No valid current sportsbook props were available. "
        "Writing an empty file so stale props cannot appear in the app."
    )

    empty_props = empty_props_frame()
    save_props(empty_props)

    return empty_props


if __name__ == "__main__":
    download_sportsbook_props()
