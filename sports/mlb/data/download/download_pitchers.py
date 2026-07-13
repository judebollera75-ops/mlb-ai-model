"""Download probable MLB starting pitchers for one slate.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Output:
    data/pitchers/<target-date>.csv

Each row represents one probable starting pitcher and includes the opposing
starter, opponent team, game location, venue, and status.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[4]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pitchers"
)

SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
)

REQUEST_TIMEOUT_SECONDS = 30

OUTPUT_COLUMNS = [
    "date",
    "game_id",
    "game_status",
    "game_status_code",
    "venue_id",
    "venue",
    "team",
    "opponent",
    "pitcher_id",
    "pitcher_name",
    "opposing_pitcher_id",
    "opposing_pitcher_name",
    "side",
    "is_home",
    "home_team",
    "away_team",
]


def get_target_date() -> str:
    """Return the requested MLB slate date."""
    raw_value = os.getenv(
        "MLB_TARGET_DATE",
        date.today().isoformat(),
    )

    try:
        parsed = datetime.strptime(
            raw_value,
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc

    return parsed.isoformat()


def get_team_data(
    game: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    """Return one side of the game's team data."""
    return (
        game.get("teams", {})
        .get(side, {})
    )


def get_team_name(
    game: dict[str, Any],
    side: str,
) -> str | None:
    """Return the team name for one side."""
    return (
        get_team_data(game, side)
        .get("team", {})
        .get("name")
    )


def get_probable_pitcher(
    game: dict[str, Any],
    side: str,
) -> dict[str, Any] | None:
    """Return probable-pitcher data for one side."""
    pitcher = get_team_data(
        game,
        side,
    ).get("probablePitcher")

    if not isinstance(pitcher, dict):
        return None

    if pitcher.get("id") is None:
        return None

    return pitcher


def build_pitcher_row(
    game: dict[str, Any],
    actual_date: str,
    side: str,
) -> dict[str, Any] | None:
    """Build one probable-pitcher row."""
    pitcher = get_probable_pitcher(
        game,
        side,
    )

    if pitcher is None:
        return None

    opposing_side = (
        "home"
        if side == "away"
        else "away"
    )

    opposing_pitcher = get_probable_pitcher(
        game,
        opposing_side,
    )

    home_team = get_team_name(
        game,
        "home",
    )

    away_team = get_team_name(
        game,
        "away",
    )

    team = get_team_name(
        game,
        side,
    )

    opponent = get_team_name(
        game,
        opposing_side,
    )

    status = game.get(
        "status",
        {},
    )

    venue = game.get(
        "venue",
        {},
    )

    return {
        "date": actual_date,
        "game_id": game.get("gamePk"),
        "game_status": status.get(
            "detailedState"
        ),
        "game_status_code": status.get(
            "statusCode"
        ),
        "venue_id": venue.get("id"),
        "venue": venue.get("name"),
        "team": team,
        "opponent": opponent,
        "pitcher_id": pitcher.get("id"),
        "pitcher_name": pitcher.get(
            "fullName"
        ),
        "opposing_pitcher_id": (
            opposing_pitcher.get("id")
            if opposing_pitcher
            else pd.NA
        ),
        "opposing_pitcher_name": (
            opposing_pitcher.get("fullName")
            if opposing_pitcher
            else pd.NA
        ),
        "side": side,
        "is_home": int(
            side == "home"
        ),
        "home_team": home_team,
        "away_team": away_team,
    }


def download_pitchers(
    target_date: str | None = None,
) -> pd.DataFrame:
    """Download and save probable pitchers for one slate."""
    if target_date is None:
        target_date = get_target_date()
    else:
        try:
            target_date = datetime.strptime(
                target_date,
                "%Y-%m-%d",
            ).date().isoformat()
        except ValueError as exc:
            raise ValueError(
                "target_date must use YYYY-MM-DD format. "
                f"Received: {target_date!r}"
            ) from exc

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": (
            "probablePitcher,"
            "team,"
            "venue"
        ),
    }

    response = requests.get(
        SCHEDULE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    data = response.json()

    rows: list[dict[str, Any]] = []

    for schedule_date in data.get(
        "dates",
        [],
    ):
        actual_date = schedule_date.get(
            "date",
            target_date,
        )

        for game in schedule_date.get(
            "games",
            [],
        ):
            for side in [
                "away",
                "home",
            ]:
                row = build_pitcher_row(
                    game=game,
                    actual_date=actual_date,
                    side=side,
                )

                if row is not None:
                    rows.append(row)

    pitchers = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if not pitchers.empty:
        numeric_columns = [
            "game_id",
            "venue_id",
            "pitcher_id",
            "opposing_pitcher_id",
            "is_home",
        ]

        for column in numeric_columns:
            pitchers[column] = pd.to_numeric(
                pitchers[column],
                errors="coerce",
            )

        pitchers = pitchers.dropna(
            subset=[
                "game_id",
                "pitcher_id",
                "pitcher_name",
                "team",
            ]
        ).copy()

        pitchers = pitchers.drop_duplicates(
            subset=[
                "game_id",
                "pitcher_id",
            ],
            keep="last",
        )

        pitchers = pitchers.sort_values(
            [
                "game_id",
                "side",
            ]
        ).reset_index(drop=True)

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    temporary_path = output_path.with_suffix(
        ".tmp.csv"
    )

    pitchers.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        output_path
    )

    print("=" * 72)
    print("PROBABLE PITCHER DOWNLOAD COMPLETE")
    print("=" * 72)
    print(f"Slate date: {target_date}")
    print(f"Pitchers saved: {len(pitchers):,}")
    print(f"Output: {output_path}")

    if not pitchers.empty:
        preview_columns = [
            "game_id",
            "team",
            "opponent",
            "pitcher_name",
            "opposing_pitcher_name",
            "side",
            "venue",
            "game_status",
        ]

        print("\nPitcher matchup preview:")

        print(
            pitchers[
                preview_columns
            ].to_string(
                index=False
            )
        )

    return pitchers


if __name__ == "__main__":
    download_pitchers()
