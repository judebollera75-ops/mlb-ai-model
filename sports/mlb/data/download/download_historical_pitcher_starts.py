"""Backfill historical MLB starting-pitcher matchups.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD
    MLB_HITTER_LOG_PATH=/optional/path/to/hitter/logs.csv

Inputs:
    data/hitter_game_logs/<season>.csv
    data/historical/pitcher_starts.csv (optional existing archive)

Output:
    data/historical/pitcher_starts.csv

The date range is taken automatically from the hitter game-log dataset so
historical starter coverage aligns with the hitter training rows.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[4]

HITTER_LOG_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "hitter_game_logs"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "historical"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "pitcher_starts.csv"
)

SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
)

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_PAUSE_SECONDS = 0.15
MAX_REQUEST_ATTEMPTS = 3

OUTPUT_COLUMNS = [
    "date",
    "game_id",
    "team",
    "opponent",
    "side",
    "is_home",
    "pitcher_id",
    "pitcher_name",
    "opposing_pitcher_id",
    "opposing_pitcher_name",
    "status",
    "status_code",
    "venue_id",
    "venue",
]


def get_target_date() -> date:
    """Return the active workflow target date."""
    raw_value = (
        os.getenv("MLB_TARGET_DATE")
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


def get_hitter_log_path() -> Path:
    """Return the hitter game-log path used for date discovery."""
    custom_path = os.getenv(
        "MLB_HITTER_LOG_PATH"
    )

    if custom_path:
        return (
            Path(custom_path)
            .expanduser()
            .resolve()
        )

    season = get_target_date().year

    return (
        HITTER_LOG_DIRECTORY
        / f"{season}.csv"
    )


def daterange(
    start_date: date,
    end_date: date,
) -> Iterator[date]:
    """Yield every calendar date in an inclusive range."""
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def get_training_date_range() -> tuple[date, date]:
    """Find the date range represented by hitter training logs."""
    hitter_log_path = get_hitter_log_path()

    if not hitter_log_path.exists():
        raise FileNotFoundError(
            "Hitter game-log file was not found: "
            f"{hitter_log_path}"
        )

    logs = pd.read_csv(
        hitter_log_path,
        usecols=["date"],
    )

    logs["date"] = pd.to_datetime(
        logs["date"],
        errors="coerce",
    )

    logs = logs.dropna(
        subset=["date"]
    )

    if logs.empty:
        raise ValueError(
            f"No valid dates were found in {hitter_log_path}"
        )

    start_date = logs["date"].min().date()
    latest_log_date = logs["date"].max().date()

    # Never request a future date relative to the active slate.
    end_date = min(
        latest_log_date,
        get_target_date(),
    )

    if start_date > end_date:
        raise ValueError(
            "Invalid historical date range: "
            f"{start_date} through {end_date}"
        )

    return start_date, end_date


def safe_request_schedule(
    day: date,
) -> dict[str, Any]:
    """Download one schedule date with retries."""
    day_string = day.isoformat()

    params = {
        "sportId": 1,
        "date": day_string,
        "hydrate": (
            "probablePitcher,"
            "team,"
            "venue"
        ),
    }

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_REQUEST_ATTEMPTS + 1,
    ):
        try:
            response = requests.get(
                SCHEDULE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:
            last_error = exc

            print(
                f"  Request attempt {attempt} failed: {exc}"
            )

            if attempt < MAX_REQUEST_ATTEMPTS:
                time.sleep(
                    attempt * 1.5
                )

    raise RuntimeError(
        f"Could not download MLB schedule for {day_string}"
    ) from last_error


def get_team_data(
    game: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    """Return one side's schedule data."""
    return (
        game.get("teams", {})
        .get(side, {})
    )


def get_team_name(
    game: dict[str, Any],
    side: str,
) -> str | None:
    """Return one side's team name."""
    return (
        get_team_data(game, side)
        .get("team", {})
        .get("name")
    )


def get_probable_pitcher(
    game: dict[str, Any],
    side: str,
) -> dict[str, Any] | None:
    """Return a valid probable-pitcher object."""
    pitcher = get_team_data(
        game,
        side,
    ).get("probablePitcher")

    if not isinstance(
        pitcher,
        dict,
    ):
        return None

    if pitcher.get("id") is None:
        return None

    return pitcher


def build_game_rows(
    game: dict[str, Any],
    game_date: str,
) -> list[dict[str, Any]]:
    """Create one starter row for each available side."""
    rows: list[dict[str, Any]] = []

    status = game.get(
        "status",
        {},
    )

    venue = game.get(
        "venue",
        {},
    )

    game_id = game.get(
        "gamePk"
    )

    for side in [
        "away",
        "home",
    ]:
        pitcher = get_probable_pitcher(
            game,
            side,
        )

        if pitcher is None:
            continue

        opponent_side = (
            "home"
            if side == "away"
            else "away"
        )

        opposing_pitcher = get_probable_pitcher(
            game,
            opponent_side,
        )

        rows.append(
            {
                "date": game_date,
                "game_id": game_id,
                "team": get_team_name(
                    game,
                    side,
                ),
                "opponent": get_team_name(
                    game,
                    opponent_side,
                ),
                "side": side,
                "is_home": int(
                    side == "home"
                ),
                "pitcher_id": pitcher.get(
                    "id"
                ),
                "pitcher_name": pitcher.get(
                    "fullName"
                ),
                "opposing_pitcher_id": (
                    opposing_pitcher.get("id")
                    if opposing_pitcher
                    else pd.NA
                ),
                "opposing_pitcher_name": (
                    opposing_pitcher.get(
                        "fullName"
                    )
                    if opposing_pitcher
                    else pd.NA
                ),
                "status": status.get(
                    "detailedState"
                ),
                "status_code": status.get(
                    "statusCode"
                ),
                "venue_id": venue.get(
                    "id"
                ),
                "venue": venue.get(
                    "name"
                ),
            }
        )

    return rows


def load_existing_archive() -> pd.DataFrame:
    """Load the current historical archive when it exists."""
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    try:
        existing = pd.read_csv(
            OUTPUT_PATH
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    for column in OUTPUT_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA

    return existing[
        OUTPUT_COLUMNS
    ].copy()


def clean_archive(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and deduplicate starter records."""
    cleaned = frame.copy()

    if cleaned.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
    )

    for column in [
        "game_id",
        "pitcher_id",
        "opposing_pitcher_id",
        "venue_id",
        "is_home",
    ]:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(
        subset=[
            "date",
            "game_id",
            "pitcher_id",
            "team",
        ]
    ).copy()

    cleaned["game_id"] = (
        cleaned["game_id"]
        .astype("int64")
    )

    cleaned["pitcher_id"] = (
        cleaned["pitcher_id"]
        .astype("int64")
    )

    cleaned = cleaned.drop_duplicates(
        subset=[
            "date",
            "game_id",
            "pitcher_id",
        ],
        keep="last",
    )

    cleaned = cleaned.sort_values(
        [
            "date",
            "game_id",
            "side",
        ]
    ).reset_index(drop=True)

    cleaned["date"] = (
        cleaned["date"]
        .dt.strftime("%Y-%m-%d")
    )

    return cleaned[
        OUTPUT_COLUMNS
    ]


def download_historical_pitcher_starts(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Backfill historical starters across the hitter-log date range."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if start_date is None and end_date is None:
        start, end = (
            get_training_date_range()
        )
    elif start_date is not None and end_date is not None:
        try:
            start = date.fromisoformat(
                start_date
            )

            end = date.fromisoformat(
                end_date
            )
        except ValueError as exc:
            raise ValueError(
                "start_date and end_date must use YYYY-MM-DD."
            ) from exc
    else:
        raise ValueError(
            "Provide both start_date and end_date, or neither."
        )

    if start > end:
        raise ValueError(
            f"Start date {start} is after end date {end}."
        )

    existing = load_existing_archive()

    existing_dates = set(
        pd.to_datetime(
            existing.get(
                "date",
                pd.Series(dtype=str),
            ),
            errors="coerce",
        )
        .dropna()
        .dt.strftime("%Y-%m-%d")
    )

    requested_days = list(
        daterange(
            start,
            end,
        )
    )

    print("=" * 72)
    print("BACKFILLING HISTORICAL PITCHER STARTS")
    print("=" * 72)
    print(f"Start date: {start}")
    print(f"End date: {end}")
    print(f"Calendar days: {len(requested_days):,}")
    print(f"Existing archive rows: {len(existing):,}")

    new_rows: list[
        dict[str, Any]
    ] = []

    downloaded_days = 0
    skipped_days = 0
    failed_days: list[str] = []

    for position, day in enumerate(
        requested_days,
        start=1,
    ):
        day_string = day.isoformat()

        if day_string in existing_dates:
            print(
                f"[{position}/{len(requested_days)}] "
                f"Skipping existing {day_string}"
            )

            skipped_days += 1
            continue

        print(
            f"[{position}/{len(requested_days)}] "
            f"Downloading {day_string}"
        )

        try:
            data = safe_request_schedule(
                day
            )
        except RuntimeError as exc:
            print(
                f"WARNING: {exc}"
            )

            failed_days.append(
                day_string
            )
            continue

        day_rows = 0

        for schedule_date in data.get(
            "dates",
            [],
        ):
            actual_date = (
                schedule_date.get(
                    "date",
                    day_string,
                )
            )

            for game in schedule_date.get(
                "games",
                [],
            ):
                game_rows = build_game_rows(
                    game=game,
                    game_date=actual_date,
                )

                new_rows.extend(
                    game_rows
                )

                day_rows += len(
                    game_rows
                )

        print(
            f"  Starter rows collected: {day_rows}"
        )

        downloaded_days += 1

        time.sleep(
            REQUEST_PAUSE_SECONDS
        )

    new_frame = pd.DataFrame(
        new_rows,
        columns=OUTPUT_COLUMNS,
    )

    combined = pd.concat(
        [
            existing,
            new_frame,
        ],
        ignore_index=True,
        sort=False,
    )

    archive = clean_archive(
        combined
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.csv"
    )

    archive.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )

    print("\n" + "=" * 72)
    print("HISTORICAL PITCHER-START BACKFILL COMPLETE")
    print("=" * 72)
    print(f"Downloaded days: {downloaded_days:,}")
    print(f"Skipped existing days: {skipped_days:,}")
    print(f"Failed days: {len(failed_days):,}")
    print(f"New starter rows: {len(new_frame):,}")
    print(f"Total archive rows: {len(archive):,}")
    print(f"Saved to: {OUTPUT_PATH}")

    if failed_days:
        print(
            "Failed dates: "
            + ", ".join(
                failed_days
            )
        )

    return archive


if __name__ == "__main__":
    download_historical_pitcher_starts()
