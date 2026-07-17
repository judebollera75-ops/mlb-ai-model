"""Download and normalize current MLB player props from ParlayAPI.

Required environment variable:
    PARLAY_API_KEY

Optional environment variable:
    MLB_TARGET_DATE=YYYY-MM-DD

Output:
    data/platform_lines.csv

The downloader retrieves player props across traditional sportsbooks, DFS
pick'em platforms, and supported exchanges in one API request. It converts
each provider row into separate Over and Under records while preserving exact
platform, player, market, line, and price availability.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_KEY = os.environ.get("PARLAY_API_KEY")
SPORT = "baseball_mlb"

BASE_URL = "https://parlay-api.com/v1"
PROPS_URL = f"{BASE_URL}/sports/{SPORT}/props"
DFS_PICKS_URL = f"{BASE_URL}/sports/{SPORT}/dfs/picks"

OUTPUT_PATH = Path("data/platform_lines.csv")
CENTRAL_TIME = ZoneInfo("America/Chicago")

REQUEST_TIMEOUT_SECONDS = 45
MAX_EVENT_AGE_MINUTES = 15

API_MARKETS = [
    "player_strikeouts",
    "player_pitcher_outs",
    "player_hits",
    "player_total_bases",
    "player_runs",
    "player_rbis",
    "player_hits_runs_rbis",
    "player_fantasy_score",
]

MARKET_MAP = {
    "player_strikeouts": "pitcher_strikeouts",
    "player_pitcher_outs": "pitcher_outs",
    "player_hits": "hitter_hits",
    "player_total_bases": "hitter_total_bases",
    "player_runs": "hitter_runs",
    "player_rbis": "hitter_rbis",
    "player_hits_runs_rbis": "hitter_hits_runs_rbis",
    "player_fantasy_score": "hitter_fantasy_score",
}

PLATFORM_TITLE_MAP = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "caesars": "Caesars",
    "betmgm": "BetMGM",
    "fanatics": "Fanatics",
    "pinnacle": "Pinnacle",
    "fliff": "Fliff",
    "bet365": "bet365",
    "betrivers": "BetRivers",
    "hardrock": "Hard Rock",
    "bovada": "Bovada",
    "novig": "Novig",
    "prophetx": "ProphetX",
    "kalshi": "Kalshi",
    "prizepicks": "PrizePicks",
    "underdog": "Underdog",
    "betr": "Betr",
    "sleeper": "Sleeper",
    "pick6": "Pick6",
    "parlayplay": "ParlayPlay",
}

DFS_PLATFORM_KEYS = {
    "prizepicks",
    "underdog",
    "betr",
    "sleeper",
    "pick6",
    "parlayplay",
}

EXCHANGE_PLATFORM_KEYS = {
    "kalshi",
    "novig",
    "prophetx",
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
    raw_value = (
        os.environ.get("MLB_TARGET_DATE")
        or date.today().isoformat()
    )

    try:
        return datetime.strptime(
            raw_value,
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc


def build_session() -> requests.Session:
    """Create a requests session with retry protection."""
    retry_strategy = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset(
            {"GET"}
        ),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "mlb-ai-model/2.0",
            "X-API-Key": API_KEY or "",
        }
    )

    return session


def clean_text(
    value: Any,
) -> str | None:
    """Return stripped text or None."""
    if value is None or pd.isna(value):
        return None

    cleaned = str(value).strip()

    return cleaned or None


def parse_api_datetime(
    value: Any,
) -> datetime | None:
    """Parse ISO timestamps or epoch timestamps into UTC."""
    if value is None or pd.isna(value):
        return None

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        numeric = float(value)

        if numeric > 10_000_000_000:
            numeric /= 1000.0

        try:
            return datetime.fromtimestamp(
                numeric,
                tz=timezone.utc,
            )
        except (
            OSError,
            OverflowError,
            ValueError,
        ):
            return None

    text = str(value).strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError:
        numeric = pd.to_numeric(
            text,
            errors="coerce",
        )

        if pd.isna(numeric):
            return None

        numeric = float(numeric)

        if numeric > 10_000_000_000:
            numeric /= 1000.0

        try:
            return datetime.fromtimestamp(
                numeric,
                tz=timezone.utc,
            )
        except (
            OSError,
            OverflowError,
            ValueError,
        ):
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def american_odds_are_valid(
    value: Any,
) -> bool:
    """Return True when a value resembles American odds."""
    try:
        odds = int(
            round(
                float(value)
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False

    return (
        odds != 0
        and abs(odds) >= 100
    )


def normalize_platform_key(
    value: Any,
) -> str | None:
    """Normalize a ParlayAPI bookmaker key."""
    cleaned = clean_text(value)

    if cleaned is None:
        return None

    return (
        cleaned
        .casefold()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def canonical_platform_key(
    value: Any,
) -> str | None:
    """Map common provider spellings to stable keys."""
    normalized = normalize_platform_key(
        value
    )

    if normalized is None:
        return None

    aliases = {
        "draftkings": "draftkings",
        "fanduel": "fanduel",
        "caesars": "caesars",
        "betmgm": "betmgm",
        "fanatics": "fanatics",
        "pinnacle": "pinnacle",
        "fliff": "fliff",
        "bet365": "bet365",
        "betrivers": "betrivers",
        "hardrock": "hardrock",
        "bovada": "bovada",
        "novig": "novig",
        "prophetx": "prophetx",
        "kalshi": "kalshi",
        "prizepicks": "prizepicks",
        "prizepick": "prizepicks",
        "prizepicksfantasy": "prizepicks",
        "prizepicksmobile": "prizepicks",
        "underdog": "underdog",
        "underdogfantasy": "underdog",
        "betr": "betr",
        "sleeper": "sleeper",
        "pick6": "pick6",
        "draftkingspick6": "pick6",
        "parlayplay": "parlayplay",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def get_platform_title(
    key: str,
    supplied_title: Any,
) -> str:
    """Return a clean user-facing platform title."""
    supplied = clean_text(
        supplied_title
    )

    if supplied:
        return supplied

    return PLATFORM_TITLE_MAP.get(
        key,
        key.replace(
            "_",
            " ",
        ).title(),
    )


def normalize_market_key(
    value: Any,
) -> str | None:
    """Normalize provider market aliases."""
    cleaned = clean_text(value)

    if cleaned is None:
        return None

    normalized = (
        cleaned.casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "pitcher_strikeouts": "player_strikeouts",
        "player_pitcher_strikeouts": "player_strikeouts",
        "player_strikeouts": "player_strikeouts",
        "pitcher_outs": "player_pitcher_outs",
        "player_outs": "player_pitcher_outs",
        "player_pitcher_outs": "player_pitcher_outs",
        "batter_hits": "player_hits",
        "player_hits": "player_hits",
        "batter_total_bases": "player_total_bases",
        "player_total_bases": "player_total_bases",
        "batter_runs_scored": "player_runs",
        "player_runs_scored": "player_runs",
        "player_runs": "player_runs",
        "batter_rbis": "player_rbis",
        "player_rbi": "player_rbis",
        "player_rbis": "player_rbis",
        "batter_hits_runs_rbis": "player_hits_runs_rbis",
        "player_hits_runs_rbis": "player_hits_runs_rbis",
        "batter_fantasy_score": "player_fantasy_score",
        "player_fantasy_points": "player_fantasy_score",
        "player_fantasy_score": "player_fantasy_score",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def empty_props_frame() -> pd.DataFrame:
    """Return an empty production-schema frame."""
    return pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )


def load_existing_props() -> pd.DataFrame:
    """Load a prior platform-lines file."""
    if not OUTPUT_PATH.exists():
        return empty_props_frame()

    try:
        existing = pd.read_csv(
            OUTPUT_PATH
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return empty_props_frame()

    for column in OUTPUT_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA

    return existing[
        OUTPUT_COLUMNS
    ].copy()


def existing_props_are_current(
    existing: pd.DataFrame,
    target_date: date,
) -> bool:
    """Return True only for a recent same-slate file."""
    if existing.empty:
        return False

    existing_dates = pd.to_datetime(
        existing["event_date"],
        errors="coerce",
    ).dt.date

    if not existing_dates.eq(
        target_date
    ).any():
        return False

    fetched_times = pd.to_datetime(
        existing["fetched_at"],
        errors="coerce",
        utc=True,
    ).dropna()

    if fetched_times.empty:
        return False

    newest_fetch = (
        fetched_times
        .max()
        .to_pydatetime()
    )

    age = (
        datetime.now(
            timezone.utc
        )
        - newest_fetch
    )

    return age <= timedelta(
        minutes=MAX_EVENT_AGE_MINUTES
    )



def extract_api_rows(
    payload: Any,
) -> list[dict[str, Any]]:
    """Extract prop-like dictionaries from common API response shapes."""
    rows: list[dict[str, Any]] = []

    def walk(
        value: Any,
        inherited: dict[str, Any] | None = None,
    ) -> None:
        inherited = dict(inherited or {})

        if isinstance(value, list):
            for item in value:
                walk(item, inherited)
            return

        if not isinstance(value, dict):
            return

        local = dict(inherited)

        # Carry useful parent metadata into nested rows.
        for key in (
            "bookmaker",
            "bookmaker_key",
            "bookmaker_title",
            "platform",
            "platform_key",
            "platform_title",
            "event_id",
            "canonical_event_id",
            "commence_time",
            "start_time",
            "home_team",
            "away_team",
            "market",
            "market_key",
        ):
            if value.get(key) is not None:
                local[key] = value.get(key)

        candidate = dict(local)
        candidate.update(value)

        has_player = any(
            candidate.get(key) is not None
            for key in (
                "player",
                "player_name",
                "description",
                "participant_name",
                "name",
            )
        )

        has_market = any(
            candidate.get(key) is not None
            for key in (
                "market",
                "market_key",
                "stat_type",
                "stat",
            )
        )

        has_line = any(
            candidate.get(key) is not None
            for key in (
                "line",
                "projection",
                "stat_value",
                "value",
            )
        )

        if has_player and has_market and has_line:
            rows.append(candidate)

        for key, child in value.items():
            if isinstance(child, (list, dict)):
                walk(child, local)

    walk(payload)

    # Deduplicate exact dictionaries generated through nested traversal.
    unique_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        marker = repr(sorted(row.items(), key=lambda item: str(item[0])))

        if marker in seen:
            continue

        seen.add(marker)
        unique_rows.append(row)

    return unique_rows


def request_prop_rows(
    session: requests.Session,
    *,
    url: str,
    parameters: dict[str, Any],
    label: str,
    required: bool,
) -> list[dict[str, Any]]:
    """Request and decode one ParlayAPI prop source."""
    try:
        response = session.get(
            url,
            params=parameters,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        print(
            f"{label} usage — "
            f"used: {response.headers.get('x-requests-used')}, "
            f"remaining: {response.headers.get('x-requests-remaining')}, "
            f"last request cost: {response.headers.get('x-requests-last')}"
        )

        response.raise_for_status()
        payload = response.json()

    except (
        requests.RequestException,
        requests.JSONDecodeError,
    ) as exc:
        if required:
            raise RuntimeError(
                f"{label} request failed: {exc}"
            ) from exc

        print(
            f"WARNING: {label} request failed and will be skipped: {exc}"
        )
        return []

    rows = extract_api_rows(payload)

    # The standard /props response is commonly already a flat list.
    if not rows and isinstance(payload, list):
        rows = [
            row
            for row in payload
            if isinstance(row, dict)
        ]

    if not rows and isinstance(payload, dict):
        for key in (
            "data",
            "props",
            "results",
            "rows",
            "picks",
            "projections",
        ):
            candidate = payload.get(key)

            if isinstance(candidate, list):
                rows = [
                    row
                    for row in candidate
                    if isinstance(row, dict)
                ]
                if rows:
                    break

    print(f"{label} raw rows: {len(rows):,}")

    return rows


def raw_platform_key(
    raw: dict[str, Any],
) -> str | None:
    """Read a platform key from several possible provider fields."""
    return canonical_platform_key(
        raw.get("bookmaker")
        or raw.get("bookmaker_key")
        or raw.get("platform")
        or raw.get("platform_key")
        or raw.get("operator")
        or raw.get("source")
        or raw.get("platform_key")
        or raw.get("operator")
        or raw.get("source")
    )


def print_raw_diagnostics(
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    """Print raw platform and market counts before normalization."""
    if not rows:
        print(f"{label}: no rows returned")
        return

    platform_counts: dict[str, int] = {}
    market_counts: dict[str, int] = {}

    for row in rows:
        platform = (
            raw_platform_key(row)
            or clean_text(
                row.get("bookmaker")
                or row.get("platform")
                or row.get("operator")
                or row.get("source")
            )
            or "<missing>"
        )

        market = clean_text(
            row.get("market_key")
            or row.get("market")
            or row.get("stat_type")
            or row.get("stat")
        ) or "<missing>"

        platform_counts[platform] = platform_counts.get(platform, 0) + 1
        market_counts[market] = market_counts.get(market, 0) + 1

    print(f"\n{label} raw rows by platform:")
    for key, count in sorted(
        platform_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{key}: {count:,}")

    print(f"\n{label} raw rows by market:")
    for key, count in sorted(
        market_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{key}: {count:,}")


def fetch_props(
    session: requests.Session,
) -> list[dict[str, Any]]:
    """Fetch standard props plus explicit PrizePicks/DFS fallbacks."""
    common_parameters = {
        "markets": ",".join(API_MARKETS),
        "dfsOdds": "midpoint",
        "limit": 10000,
    }

    all_rows = request_prop_rows(
        session,
        url=PROPS_URL,
        parameters=common_parameters,
        label="All-platform /props",
        required=True,
    )

    prizepicks_rows = request_prop_rows(
        session,
        url=PROPS_URL,
        parameters={
            **common_parameters,
            "bookmakers": "prizepicks_mobile",
        },
        label="PrizePicks-only /props",
        required=False,
    )

    # PrizePicks is exposed through /props. ParlayAPI's current MLB
    # coverage reports identify its live source key as prizepicks_mobile.
    dfs_rows: list[dict[str, Any]] = []

    print_raw_diagnostics(
        prizepicks_rows,
        "PrizePicks-only /props",
    )

    combined = all_rows + prizepicks_rows

    # Keep duplicate removal conservative here; final side-level dedupe still
    # happens in clean_props().
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for row in combined:
        marker = (
            raw_platform_key(row),
            clean_text(
                row.get("player")
                or row.get("player_name")
                or row.get("description")
                or row.get("participant_name")
                or row.get("name")
            ),
            clean_text(
                row.get("market_key")
                or row.get("market")
                or row.get("stat_type")
                or row.get("stat")
            ),
            row.get("line")
            if row.get("line") is not None
            else row.get("projection")
            if row.get("projection") is not None
            else row.get("stat_value")
            if row.get("stat_value") is not None
            else row.get("value"),
            row.get("over_price"),
            row.get("under_price"),
            row.get("canonical_event_id")
            or row.get("event_id")
            or row.get("id"),
        )

        if marker in seen:
            continue

        seen.add(marker)
        unique_rows.append(row)

    print(
        f"Combined unique raw rows: {len(unique_rows):,}"
    )

    return unique_rows


def build_side_row(
    *,
    raw: dict[str, Any],
    direction: str,
    price_field: str,
    target_date: date,
    fetched_at: str,
) -> dict[str, Any] | None:
    """Convert one ParlayAPI prop side into the app schema."""
    platform_key = canonical_platform_key(
        raw.get("bookmaker")
        or raw.get("bookmaker_key")
        or raw.get("platform")
    )

    if platform_key is None:
        return None

    platform = get_platform_title(
        key=platform_key,
        supplied_title=(
            raw.get("bookmaker_title")
            or raw.get("platform_title")
        ),
    )

    market_key = normalize_market_key(
        raw.get("market_key")
        or raw.get("market")
        or raw.get("stat_type")
        or raw.get("stat")
    )

    normalized_market = MARKET_MAP.get(
        market_key or ""
    )

    if normalized_market is None:
        return None

    player = clean_text(
        raw.get("player")
        or raw.get("player_name")
        or raw.get("description")
        or raw.get("participant_name")
        or raw.get("name")
    )

    if player is None:
        return None

    line = pd.to_numeric(
        raw.get("line")
        if raw.get("line") is not None
        else raw.get("projection")
        if raw.get("projection") is not None
        else raw.get("stat_value")
        if raw.get("stat_value") is not None
        else raw.get("value"),
        errors="coerce",
    )

    price = pd.to_numeric(
        raw.get(price_field),
        errors="coerce",
    )

    if pd.isna(line):
        return None

    # DFS projection feeds do not always include side prices. With
    # dfsOdds=midpoint, represent an available side as even-money so the
    # downstream schema remains consistent.
    if pd.isna(price) and platform_key in DFS_PLATFORM_KEYS:
        side_available = raw.get(
            "over_available"
            if direction == "Over"
            else "under_available"
        )

        if side_available is False:
            return None

        price = 100

    if (
        pd.isna(price)
        or not american_odds_are_valid(
            price
        )
    ):
        return None

    commence_time = parse_api_datetime(
        raw.get("commence_time")
        or raw.get("start_time")
        or raw.get("game_time")
        or raw.get("scheduled_at")
    )

    if commence_time is None:
        return None

    event_date = (
        commence_time
        .astimezone(
            CENTRAL_TIME
        )
        .date()
    )

    if event_date != target_date:
        return None

    if commence_time <= datetime.now(
        timezone.utc
    ):
        return None

    event_id = clean_text(
        raw.get("canonical_event_id")
        or raw.get("event_id")
        or raw.get("id")
    )

    if event_id is None:
        event_id = (
            f"{event_date.isoformat()}|"
            f"{clean_text(raw.get('away_team'))}|"
            f"{clean_text(raw.get('home_team'))}"
        )

    return {
        "event_id": event_id,
        "event_date": event_date.isoformat(),
        "commence_time": commence_time.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "platform": platform,
        "platform_key": platform_key,
        "player": player,
        "market": normalized_market,
        "market_key": market_key,
        "direction": direction,
        "line": float(line),
        "sportsbook_odds": int(
            round(
                float(price)
            )
        ),
        "home_team": clean_text(
            raw.get("home_team")
        ),
        "away_team": clean_text(
            raw.get("away_team")
        ),
        "fetched_at": fetched_at,
        "source": "parlay_api",
        "status": "open",
    }


def normalize_props(
    raw_rows: list[dict[str, Any]],
    target_date: date,
    fetched_at: str,
) -> pd.DataFrame:
    """Convert ParlayAPI rows into one row per available side."""
    normalized_rows: list[
        dict[str, Any]
    ] = []

    for raw in raw_rows:
        over_row = build_side_row(
            raw=raw,
            direction="Over",
            price_field="over_price",
            target_date=target_date,
            fetched_at=fetched_at,
        )

        if over_row is not None:
            normalized_rows.append(
                over_row
            )

        under_row = build_side_row(
            raw=raw,
            direction="Under",
            price_field="under_price",
            target_date=target_date,
            fetched_at=fetched_at,
        )

        if under_row is not None:
            normalized_rows.append(
                under_row
            )

    return pd.DataFrame(
        normalized_rows,
        columns=OUTPUT_COLUMNS,
    )


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

    cleaned = cleaned.dropna(
        subset=[
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
    )

    cleaned = cleaned.loc[
        cleaned["event_date"].eq(
            target_date
        )
    ].copy()

    cleaned = cleaned.loc[
        cleaned["commence_time"]
        > datetime.now(
            timezone.utc
        )
    ].copy()

    cleaned = cleaned.loc[
        cleaned["market"].isin(
            set(
                MARKET_MAP.values()
            )
        )
    ].copy()

    cleaned = cleaned.loc[
        cleaned["direction"].isin(
            {
                "Over",
                "Under",
            }
        )
    ].copy()

    cleaned = cleaned.loc[
        cleaned[
            "sportsbook_odds"
        ].apply(
            american_odds_are_valid
        )
    ].copy()

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
        ]
    ).reset_index(
        drop=True
    )

    cleaned["event_date"] = (
        cleaned["event_date"]
        .astype(str)
    )

    cleaned["commence_time"] = (
        cleaned["commence_time"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    cleaned["fetched_at"] = (
        cleaned["fetched_at"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    return cleaned[
        OUTPUT_COLUMNS
    ]


def save_props(
    props: pd.DataFrame,
) -> None:
    """Save normalized platform lines atomically."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        OUTPUT_PATH.with_suffix(
            ".tmp.csv"
        )
    )

    props.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )

    print(
        f"\nSaved {len(props):,} platform rows "
        f"to {OUTPUT_PATH}"
    )

    if props.empty:
        return

    platform_counts = (
        props.groupby(
            [
                "platform_key",
                "platform",
            ],
            dropna=False,
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    market_counts = (
        props.groupby(
            "market",
            dropna=False,
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print("\nRows by platform:")
    print(platform_counts.to_string())

    print("\nRows by market:")
    print(market_counts.to_string())

    dfs_count = int(
        props[
            "platform_key"
        ].isin(
            DFS_PLATFORM_KEYS
        ).sum()
    )

    exchange_count = int(
        props[
            "platform_key"
        ].isin(
            EXCHANGE_PLATFORM_KEYS
        ).sum()
    )

    print(
        f"\nDFS/pick'em rows: {dfs_count:,}"
    )

    print(
        f"Exchange rows: {exchange_count:,}"
    )

    for platform_key in [
        "prizepicks",
        "underdog",
        "fliff",
        "kalshi",
    ]:
        count = int(
            props[
                "platform_key"
            ].eq(
                platform_key
            ).sum()
        )

        print(
            f"{platform_key}: {count:,} rows"
        )


def download_sportsbook_props() -> pd.DataFrame:
    """Download all current MLB platform props from ParlayAPI."""
    if not API_KEY:
        raise RuntimeError(
            "PARLAY_API_KEY is missing from the environment."
        )

    target_date = get_target_date()

    fetched_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    existing_props = load_existing_props()

    print("=" * 72)
    print("DOWNLOADING CURRENT MLB PLATFORM PROPS")
    print("=" * 72)
    print("Provider: ParlayAPI")
    print(
        f"Slate date: "
        f"{target_date.isoformat()}"
    )
    print(
        f"Props endpoint: {PROPS_URL}"
    )
    print(
        f"Requested markets: "
        f"{len(API_MARKETS)}"
    )
    print("=" * 72)

    session = build_session()

    raw_rows = fetch_props(
        session
    )

    print(
        f"Raw ParlayAPI rows: "
        f"{len(raw_rows):,}"
    )

    normalized = normalize_props(
        raw_rows=raw_rows,
        target_date=target_date,
        fetched_at=fetched_at,
    )

    props = clean_props(
        normalized,
        target_date,
    )

    print(
        f"Normalized side rows: "
        f"{len(normalized):,}"
    )

    print(
        f"Validated current rows: "
        f"{len(props):,}"
    )

    if not props.empty:
        save_props(
            props
        )

        return props

    if existing_props_are_current(
        existing_props,
        target_date,
    ):
        print(
            "\nWARNING: ParlayAPI returned no usable current props. "
            "Preserving the existing same-slate file because it was "
            f"fetched within {MAX_EVENT_AGE_MINUTES} minutes."
        )

        return existing_props

    print(
        "\nWARNING: No valid current platform props were available. "
        "Writing an empty file so stale recommendations cannot remain "
        "visible in the app."
    )

    empty = empty_props_frame()

    save_props(
        empty
    )

    return empty


if __name__ == "__main__":
    download_sportsbook_props()
