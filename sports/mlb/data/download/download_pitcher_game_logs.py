"""Download leakage-safe MLB pitcher game logs.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Input:
    data/pitchers/<target-date>.csv

Output:
    data/game_logs/<target-date>.csv

The output contains all available games before the requested slate date for
each scheduled pitcher. No game on or after the slate date is included.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[4]

PITCHERS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pitchers"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "game_logs"
)

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_PAUSE_SECONDS = 0.10

OUTPUT_COLUMNS = [
    "target_date",
    "season",
    "game_id",
    "game_date",
    "pitcher_id",
    "pitcher_name",
    "team",
    "game_team",
    "opponent",
    "is_home",
    "decision",
    "games_started",
    "innings",
    "outs_recorded",
    "batters_faced",
    "at_bats_against",
    "pitches_thrown",
    "strikes",
    "strikeouts",
    "walks",
    "intentional_walks",
    "hits",
    "doubles_allowed",
    "triples_allowed",
    "home_runs",
    "runs",
    "earned_runs",
    "hit_batters",
    "wild_pitches",
    "wins",
    "losses",
    "saves",
    "holds",
    "era",
    "whip",
    "strikeout_rate_bf",
    "walk_rate_bf",
    "k_minus_bb_rate",
    "home_runs_per_9",
    "hits_per_9",
    "walks_per_9",
    "strikeouts_per_9",
    "fip_component",
]


def get_target_date() -> date:
    """Return the requested MLB slate date."""
    raw_value = os.getenv(
        "MLB_TARGET_DATE",
        date.today().isoformat(),
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


def safe_number(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to a finite float."""
    numeric = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(numeric):
        return default

    numeric = float(numeric)

    if not np.isfinite(numeric):
        return default

    return numeric


def safe_integer(
    value: Any,
    default: int = 0,
) -> int:
    """Convert a value to an integer."""
    return int(
        round(
            safe_number(
                value,
                float(default),
            )
        )
    )


def innings_to_outs(value: Any) -> int | None:
    """Convert baseball innings notation to total outs.

    Examples:
        5.0 -> 15
        5.1 -> 16
        5.2 -> 17
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if "." in text:
        whole_text, partial_text = text.split(
            ".",
            1,
        )
    else:
        whole_text, partial_text = text, "0"

    try:
        whole_innings = int(whole_text)
        partial_outs = int(
            partial_text[:1] or "0"
        )
    except (TypeError, ValueError):
        return None

    if partial_outs not in {0, 1, 2}:
        return None

    return (
        whole_innings * 3
        + partial_outs
    )


def outs_to_decimal_innings(
    outs_recorded: int | float | None,
) -> float:
    """Convert outs into decimal innings for rate calculations."""
    if outs_recorded is None or pd.isna(outs_recorded):
        return 0.0

    return float(outs_recorded) / 3.0


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Divide safely and return missing when denominator is zero."""
    if denominator == 0:
        return float("nan")

    return numerator / denominator


def extract_team_name(
    value: Any,
) -> str | None:
    """Extract a team name from an MLB Stats API object."""
    if not isinstance(value, dict):
        return None

    name = value.get("name")

    if name:
        return str(name).strip()

    return None


def fetch_pitcher_game_logs(
    pitcher_id: int,
    season: int,
) -> list[dict[str, Any]]:
    """Request one pitcher's season game log."""
    url = (
        "https://statsapi.mlb.com/api/v1/"
        f"people/{pitcher_id}/stats"
    )

    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": str(season),
        "hydrate": "team,opponent,game",
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    data = response.json()

    stats_blocks = data.get(
        "stats",
        [],
    )

    if not stats_blocks:
        return []

    return stats_blocks[0].get(
        "splits",
        [],
    )


def build_log_row(
    game: dict[str, Any],
    pitcher: pd.Series,
    target_date: date,
    season: int,
) -> dict[str, Any] | None:
    """Convert one API split into a clean game-log row."""
    game_date_raw = game.get("date")

    game_date = pd.to_datetime(
        game_date_raw,
        errors="coerce",
    )

    if pd.isna(game_date):
        return None

    # Strictly exclude the current slate and all future games.
    if game_date.date() >= target_date:
        return None

    stat = game.get(
        "stat",
        {},
    )

    game_info = game.get(
        "game",
        {},
    )

    team_object = game.get(
        "team",
        {},
    )

    opponent_object = game.get(
        "opponent",
        {},
    )

    innings = stat.get(
        "inningsPitched"
    )

    outs_recorded = innings_to_outs(
        innings
    )

    innings_decimal = outs_to_decimal_innings(
        outs_recorded
    )

    batters_faced = safe_number(
        stat.get("battersFaced")
    )

    strikeouts = safe_number(
        stat.get("strikeOuts")
    )

    walks = safe_number(
        stat.get("baseOnBalls")
    )

    intentional_walks = safe_number(
        stat.get("intentionalWalks")
    )

    hits = safe_number(
        stat.get("hits")
    )

    home_runs = safe_number(
        stat.get("homeRuns")
    )

    hit_batters = safe_number(
        stat.get("hitBatsmen")
    )

    earned_runs = safe_number(
        stat.get("earnedRuns")
    )

    era = safe_number(
        stat.get("era"),
        float("nan"),
    )

    whip = safe_divide(
        walks + hits,
        innings_decimal,
    )

    strikeout_rate_bf = safe_divide(
        strikeouts,
        batters_faced,
    )

    walk_rate_bf = safe_divide(
        walks,
        batters_faced,
    )

    k_minus_bb_rate = (
        strikeout_rate_bf
        - walk_rate_bf
        if (
            np.isfinite(strikeout_rate_bf)
            and np.isfinite(walk_rate_bf)
        )
        else float("nan")
    )

    home_runs_per_9 = safe_divide(
        home_runs * 9.0,
        innings_decimal,
    )

    hits_per_9 = safe_divide(
        hits * 9.0,
        innings_decimal,
    )

    walks_per_9 = safe_divide(
        walks * 9.0,
        innings_decimal,
    )

    strikeouts_per_9 = safe_divide(
        strikeouts * 9.0,
        innings_decimal,
    )

    # This is the variable portion of FIP. A league/season constant can be
    # added later without changing the relative ranking.
    fip_component = safe_divide(
        (
            13.0 * home_runs
            + 3.0 * (
                walks
                + hit_batters
                - intentional_walks
            )
            - 2.0 * strikeouts
        ),
        innings_decimal,
    )

    game_team = extract_team_name(
        team_object
    )

    opponent = extract_team_name(
        opponent_object
    )

    scheduled_team = pitcher.get(
        "team"
    )

    is_home = game.get(
        "isHome"
    )

    if is_home is None:
        is_home = game_info.get(
            "isHome"
        )

    if is_home is None:
        is_home_value = pd.NA
    else:
        is_home_value = int(
            bool(is_home)
        )

    game_id = game_info.get(
        "gamePk"
    )

    if game_id is None:
        game_id = game.get(
            "gamePk"
        )

    decision = stat.get(
        "note"
    )

    wins = safe_integer(
        stat.get("wins")
    )

    losses = safe_integer(
        stat.get("losses")
    )

    saves = safe_integer(
        stat.get("saves")
    )

    holds = safe_integer(
        stat.get("holds")
    )

    return {
        "target_date": target_date.isoformat(),
        "season": season,
        "game_id": game_id,
        "game_date": game_date.date().isoformat(),
        "pitcher_id": safe_integer(
            pitcher.get("pitcher_id")
        ),
        "pitcher_name": pitcher.get(
            "pitcher_name"
        ),
        "team": scheduled_team,
        "game_team": game_team,
        "opponent": opponent,
        "is_home": is_home_value,
        "decision": decision,
        "games_started": safe_integer(
            stat.get("gamesStarted")
        ),
        "innings": innings,
        "outs_recorded": outs_recorded,
        "batters_faced": batters_faced,
        "at_bats_against": safe_number(
            stat.get("atBats")
        ),
        "pitches_thrown": safe_number(
            stat.get("numberOfPitches")
        ),
        "strikes": safe_number(
            stat.get("strikes")
        ),
        "strikeouts": strikeouts,
        "walks": walks,
        "intentional_walks": intentional_walks,
        "hits": hits,
        "doubles_allowed": safe_number(
            stat.get("doubles")
        ),
        "triples_allowed": safe_number(
            stat.get("triples")
        ),
        "home_runs": home_runs,
        "runs": safe_number(
            stat.get("runs")
        ),
        "earned_runs": earned_runs,
        "hit_batters": hit_batters,
        "wild_pitches": safe_number(
            stat.get("wildPitches")
        ),
        "wins": wins,
        "losses": losses,
        "saves": saves,
        "holds": holds,
        "era": era,
        "whip": whip,
        "strikeout_rate_bf": strikeout_rate_bf,
        "walk_rate_bf": walk_rate_bf,
        "k_minus_bb_rate": k_minus_bb_rate,
        "home_runs_per_9": home_runs_per_9,
        "hits_per_9": hits_per_9,
        "walks_per_9": walks_per_9,
        "strikeouts_per_9": strikeouts_per_9,
        "fip_component": fip_component,
    }


def download_game_logs(
    target_date: str | None = None,
    season: str | int | None = None,
) -> pd.DataFrame:
    """Download all pre-slate game logs for scheduled pitchers."""
    if target_date is None:
        slate_date = get_target_date()
    else:
        try:
            slate_date = datetime.strptime(
                target_date,
                "%Y-%m-%d",
            ).date()
        except ValueError as exc:
            raise ValueError(
                "target_date must use YYYY-MM-DD format. "
                f"Received: {target_date!r}"
            ) from exc

    if season is None:
        season_value = slate_date.year
    else:
        season_value = int(season)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pitchers_path = (
        PITCHERS_DIRECTORY
        / f"{slate_date.isoformat()}.csv"
    )

    if not pitchers_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher file: {pitchers_path}"
        )

    pitchers = pd.read_csv(
        pitchers_path
    )

    required_columns = {
        "pitcher_id",
        "pitcher_name",
        "team",
    }

    missing_columns = (
        required_columns
        - set(pitchers.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{pitchers_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    pitchers["pitcher_id"] = pd.to_numeric(
        pitchers["pitcher_id"],
        errors="coerce",
    )

    pitchers = pitchers.dropna(
        subset=[
            "pitcher_id",
            "pitcher_name",
        ]
    ).copy()

    pitchers["pitcher_id"] = (
        pitchers["pitcher_id"]
        .astype(int)
    )

    pitchers = pitchers.drop_duplicates(
        subset=["pitcher_id"],
        keep="first",
    )

    print("=" * 72)
    print("DOWNLOADING PITCHER GAME LOGS")
    print(f"Slate date: {slate_date.isoformat()}")
    print(f"Season: {season_value}")
    print(f"Pitchers: {len(pitchers):,}")
    print("=" * 72)

    rows: list[dict[str, Any]] = []

    total_pitchers = len(
        pitchers
    )

    for index, (_, pitcher) in enumerate(
        pitchers.iterrows(),
        start=1,
    ):
        pitcher_id = int(
            pitcher["pitcher_id"]
        )

        pitcher_name = str(
            pitcher["pitcher_name"]
        )

        print(
            f"[{index}/{total_pitchers}] "
            f"Downloading {pitcher_name}..."
        )

        try:
            splits = fetch_pitcher_game_logs(
                pitcher_id=pitcher_id,
                season=season_value,
            )
        except requests.RequestException as exc:
            print(
                f"WARNING: Could not download "
                f"{pitcher_name}: {exc}"
            )
            continue

        pitcher_rows = 0

        for game in splits:
            row = build_log_row(
                game=game,
                pitcher=pitcher,
                target_date=slate_date,
                season=season_value,
            )

            if row is None:
                continue

            rows.append(row)
            pitcher_rows += 1

        print(
            f"  Saved {pitcher_rows} prior games."
        )

        time.sleep(
            REQUEST_PAUSE_SECONDS
        )

    logs = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if not logs.empty:
        logs["game_date"] = pd.to_datetime(
            logs["game_date"],
            errors="coerce",
        )

        numeric_columns = [
            column
            for column in OUTPUT_COLUMNS
            if column not in {
                "target_date",
                "game_date",
                "pitcher_name",
                "team",
                "game_team",
                "opponent",
                "decision",
                "innings",
            }
        ]

        for column in numeric_columns:
            logs[column] = pd.to_numeric(
                logs[column],
                errors="coerce",
            )

        logs = logs.dropna(
            subset=[
                "game_date",
                "pitcher_id",
            ]
        )

        logs = logs.drop_duplicates(
            subset=[
                "pitcher_id",
                "game_id",
                "game_date",
            ],
            keep="last",
        )

        logs = logs.sort_values(
            [
                "pitcher_id",
                "game_date",
            ],
            ascending=[
                True,
                False,
            ],
        ).reset_index(drop=True)

        logs["game_date"] = logs[
            "game_date"
        ].dt.date.astype(str)

    output_path = (
        OUTPUT_DIRECTORY
        / f"{slate_date.isoformat()}.csv"
    )

    temporary_path = output_path.with_suffix(
        ".tmp.csv"
    )

    logs.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        output_path
    )

    print("\n" + "=" * 72)
    print("PITCHER GAME LOG DOWNLOAD COMPLETE")
    print("=" * 72)
    print(f"Rows saved: {len(logs):,}")
    print(
        "Pitchers represented: "
        f"{logs['pitcher_id'].nunique() if not logs.empty else 0:,}"
    )
    print(f"Saved to: {output_path}")

    if not logs.empty:
        preview_columns = [
            "game_date",
            "pitcher_name",
            "opponent",
            "innings",
            "outs_recorded",
            "strikeouts",
            "walks",
            "hits",
            "earned_runs",
            "whip",
            "strikeout_rate_bf",
            "walk_rate_bf",
            "k_minus_bb_rate",
            "home_runs_per_9",
            "fip_component",
        ]

        print("\nPreview:")

        print(
            logs[
                preview_columns
            ]
            .head(40)
            .to_string(index=False)
        )

    return logs


if __name__ == "__main__":
    download_game_logs()
