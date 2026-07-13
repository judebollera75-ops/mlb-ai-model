"""Download season pitching statistics for the active MLB slate.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Input:
    data/pitchers/<target-date>.csv

Output:
    data/pitcher_stats/<target-date>.csv
"""

from __future__ import annotations

import os
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
    / "pitcher_stats"
)

REQUEST_TIMEOUT_SECONDS = 30

OUTPUT_COLUMNS = [
    "date",
    "season",
    "game_id",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "side",
    "is_home",
    "games",
    "games_started",
    "innings_pitched",
    "innings_decimal",
    "era",
    "whip",
    "strikeouts",
    "walks",
    "hits_allowed",
    "earned_runs",
    "home_runs_allowed",
    "batters_faced",
    "wins",
    "losses",
    "season_k_per_start",
    "avg_ip_per_start",
    "walks_per_start",
    "hits_per_start",
    "earned_runs_per_start",
    "win_rate",
    "k_per_9",
    "bb_per_9",
    "hits_per_9",
    "hr_per_9",
    "strikeout_rate",
    "walk_rate",
    "k_minus_bb_rate",
    "fip_component",
]


def get_target_date() -> str:
    """Return the workflow slate date instead of the runner's UTC date."""
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


def safe_number(
    value: Any,
    default: float | None = None,
) -> float | None:
    """Convert a value to a finite number."""
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


def baseball_ip_to_decimal(
    ip_value: Any,
) -> float | None:
    """Convert MLB innings notation into decimal innings.

    Examples:
        5.0 -> 5.0
        5.1 -> 5.333...
        5.2 -> 5.666...
    """
    if ip_value is None or pd.isna(ip_value):
        return None

    text = str(ip_value).strip()

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
        whole = int(whole_text)
        partial = int(
            partial_text[:1] or "0"
        )
    except (TypeError, ValueError):
        return None

    if partial == 0:
        return float(whole)

    if partial == 1:
        return whole + (1.0 / 3.0)

    if partial == 2:
        return whole + (2.0 / 3.0)

    return None


def safe_divide(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    """Divide safely."""
    if numerator is None or denominator is None:
        return None

    if not np.isfinite(numerator):
        return None

    if not np.isfinite(denominator):
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def fetch_pitcher_stats(
    pitcher_id: int,
    season: int,
) -> dict[str, Any] | None:
    """Download one pitcher's season statistics."""
    url = (
        "https://statsapi.mlb.com/api/v1/"
        f"people/{pitcher_id}/stats"
    )

    params = {
        "stats": "season",
        "group": "pitching",
        "season": str(season),
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
        return None

    splits = stats_blocks[0].get(
        "splits",
        [],
    )

    if not splits:
        return None

    return splits[0].get(
        "stat",
        {},
    )


def build_pitcher_row(
    pitcher: pd.Series,
    stat: dict[str, Any],
    target_date: str,
    season: int,
) -> dict[str, Any]:
    """Build one cleaned season-stat row."""
    games_started = safe_number(
        stat.get("gamesStarted")
    )

    innings_pitched = stat.get(
        "inningsPitched"
    )

    innings_decimal = baseball_ip_to_decimal(
        innings_pitched
    )

    strikeouts = safe_number(
        stat.get("strikeOuts")
    )

    walks = safe_number(
        stat.get("baseOnBalls")
    )

    hits_allowed = safe_number(
        stat.get("hits")
    )

    earned_runs = safe_number(
        stat.get("earnedRuns")
    )

    home_runs_allowed = safe_number(
        stat.get("homeRuns")
    )

    batters_faced = safe_number(
        stat.get("battersFaced")
    )

    wins = safe_number(
        stat.get("wins")
    )

    season_k_per_start = safe_divide(
        strikeouts,
        games_started,
    )

    avg_ip_per_start = safe_divide(
        innings_decimal,
        games_started,
    )

    walks_per_start = safe_divide(
        walks,
        games_started,
    )

    hits_per_start = safe_divide(
        hits_allowed,
        games_started,
    )

    earned_runs_per_start = safe_divide(
        earned_runs,
        games_started,
    )

    win_rate = safe_divide(
        wins,
        games_started,
    )

    k_per_9 = safe_divide(
        strikeouts * 9.0
        if strikeouts is not None
        else None,
        innings_decimal,
    )

    bb_per_9 = safe_divide(
        walks * 9.0
        if walks is not None
        else None,
        innings_decimal,
    )

    hits_per_9 = safe_divide(
        hits_allowed * 9.0
        if hits_allowed is not None
        else None,
        innings_decimal,
    )

    hr_per_9 = safe_divide(
        home_runs_allowed * 9.0
        if home_runs_allowed is not None
        else None,
        innings_decimal,
    )

    strikeout_rate = safe_divide(
        strikeouts,
        batters_faced,
    )

    walk_rate = safe_divide(
        walks,
        batters_faced,
    )

    if (
        strikeout_rate is not None
        and walk_rate is not None
    ):
        k_minus_bb_rate = (
            strikeout_rate
            - walk_rate
        )
    else:
        k_minus_bb_rate = None

    # Variable portion of FIP. A league constant can be added later.
    if (
        home_runs_allowed is not None
        and walks is not None
        and strikeouts is not None
    ):
        fip_numerator = (
            13.0 * home_runs_allowed
            + 3.0 * walks
            - 2.0 * strikeouts
        )
    else:
        fip_numerator = None

    fip_component = safe_divide(
        fip_numerator,
        innings_decimal,
    )

    return {
        "date": target_date,
        "season": season,
        "game_id": pitcher.get("game_id"),
        "pitcher_id": int(
            pitcher["pitcher_id"]
        ),
        "pitcher_name": pitcher.get(
            "pitcher_name"
        ),
        "team": pitcher.get("team"),
        "opponent": pitcher.get(
            "opponent"
        ),
        "side": pitcher.get("side"),
        "is_home": pitcher.get(
            "is_home"
        ),
        "games": safe_number(
            stat.get("gamesPlayed")
        ),
        "games_started": games_started,
        "innings_pitched": innings_pitched,
        "innings_decimal": innings_decimal,
        "era": safe_number(
            stat.get("era")
        ),
        "whip": safe_number(
            stat.get("whip")
        ),
        "strikeouts": strikeouts,
        "walks": walks,
        "hits_allowed": hits_allowed,
        "earned_runs": earned_runs,
        "home_runs_allowed": home_runs_allowed,
        "batters_faced": batters_faced,
        "wins": wins,
        "losses": safe_number(
            stat.get("losses")
        ),
        "season_k_per_start": season_k_per_start,
        "avg_ip_per_start": avg_ip_per_start,
        "walks_per_start": walks_per_start,
        "hits_per_start": hits_per_start,
        "earned_runs_per_start": earned_runs_per_start,
        "win_rate": win_rate,
        "k_per_9": k_per_9,
        "bb_per_9": bb_per_9,
        "hits_per_9": hits_per_9,
        "hr_per_9": hr_per_9,
        "strikeout_rate": strikeout_rate,
        "walk_rate": walk_rate,
        "k_minus_bb_rate": k_minus_bb_rate,
        "fip_component": fip_component,
    }


def download_pitcher_stats(
    target_date: str | None = None,
    season: str | int | None = None,
) -> pd.DataFrame:
    """Download season stats for pitchers on the requested slate."""
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

    if season is None:
        season_value = date.fromisoformat(
            target_date
        ).year
    else:
        season_value = int(season)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pitchers_path = (
        PITCHERS_DIRECTORY
        / f"{target_date}.csv"
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
            "team",
        ]
    ).copy()

    pitchers["pitcher_id"] = (
        pitchers["pitcher_id"]
        .astype(int)
    )

    pitchers = pitchers.drop_duplicates(
        subset=["pitcher_id"],
        keep="last",
    )

    print("=" * 72)
    print("DOWNLOADING PITCHER SEASON STATS")
    print(f"Slate date: {target_date}")
    print(f"Season: {season_value}")
    print(f"Pitchers: {len(pitchers):,}")
    print("=" * 72)

    rows: list[dict[str, Any]] = []

    for count, (_, pitcher) in enumerate(
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
            f"[{count}/{len(pitchers)}] "
            f"Downloading {pitcher_name}..."
        )

        try:
            stat = fetch_pitcher_stats(
                pitcher_id=pitcher_id,
                season=season_value,
            )
        except requests.RequestException as exc:
            print(
                f"WARNING: Skipped {pitcher_name}: {exc}"
            )
            continue

        if stat is None:
            print(
                f"WARNING: No season stats returned for "
                f"{pitcher_name}"
            )
            continue

        rows.append(
            build_pitcher_row(
                pitcher=pitcher,
                stat=stat,
                target_date=target_date,
                season=season_value,
            )
        )

    stats = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if not stats.empty:
        numeric_columns = [
            column
            for column in OUTPUT_COLUMNS
            if column
            not in {
                "date",
                "pitcher_name",
                "team",
                "opponent",
                "side",
                "innings_pitched",
            }
        ]

        for column in numeric_columns:
            stats[column] = pd.to_numeric(
                stats[column],
                errors="coerce",
            )

        stats = stats.drop_duplicates(
            subset=["pitcher_id"],
            keep="last",
        )

        stats = stats.sort_values(
            [
                "team",
                "pitcher_name",
            ]
        ).reset_index(drop=True)

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    temporary_path = output_path.with_suffix(
        ".tmp.csv"
    )

    stats.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        output_path
    )

    print("\n" + "=" * 72)
    print("PITCHER SEASON STATS DOWNLOAD COMPLETE")
    print("=" * 72)
    print(f"Rows saved: {len(stats):,}")
    print(f"Output: {output_path}")

    if not stats.empty:
        preview_columns = [
            "pitcher_name",
            "team",
            "opponent",
            "games_started",
            "innings_decimal",
            "era",
            "whip",
            "k_per_9",
            "bb_per_9",
            "hr_per_9",
            "strikeout_rate",
            "walk_rate",
            "k_minus_bb_rate",
            "fip_component",
        ]

        preview_columns = [
            column
            for column in preview_columns
            if column in stats.columns
        ]

        print("\nPreview:")

        print(
            stats[
                preview_columns
            ].to_string(index=False)
        )

    return stats


if __name__ == "__main__":
    download_pitcher_stats()
