"""Backfill detailed historical MLB pitcher game logs.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Inputs:
    data/historical/pitcher_starts.csv
    data/historical/pitcher_game_logs.csv (optional existing archive)

Output:
    data/historical/pitcher_game_logs.csv

For every pitcher found in the historical starter archive, this script
downloads season game logs and keeps only appearances before or on the latest
historical starter date. The resulting archive is deduplicated by pitcher and
game, and can be reused by opponent-pitcher feature engineering.
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

HISTORICAL_STARTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "pitcher_starts.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "historical"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "pitcher_game_logs.csv"
)

STATS_URL_TEMPLATE = (
    "https://statsapi.mlb.com/api/v1/"
    "people/{pitcher_id}/stats"
)

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_PAUSE_SECONDS = 0.10
MAX_REQUEST_ATTEMPTS = 3

OUTPUT_COLUMNS = [
    "season",
    "pitcher_id",
    "pitcher_name",
    "game_id",
    "game_date",
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


def safe_number(
    value: Any,
    default: float | None = None,
) -> float | None:
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
    numeric = safe_number(
        value,
        float(default),
    )

    if numeric is None:
        return default

    return int(round(numeric))


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


def innings_to_outs(
    value: Any,
) -> int | None:
    """Convert MLB innings notation into total outs."""
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
        whole = int(whole_text)
        partial = int(
            partial_text[:1] or "0"
        )
    except (TypeError, ValueError):
        return None

    if partial not in {0, 1, 2}:
        return None

    return whole * 3 + partial


def extract_team_name(
    value: Any,
) -> str | None:
    """Extract a team name from an MLB API object."""
    if not isinstance(value, dict):
        return None

    name = value.get("name")

    if name is None:
        return None

    return str(name).strip()


def load_historical_starters() -> tuple[pd.DataFrame, date]:
    """Load unique historical starters and determine the archive cutoff."""
    if not HISTORICAL_STARTS_PATH.exists():
        raise FileNotFoundError(
            "Historical pitcher-start archive was not found: "
            f"{HISTORICAL_STARTS_PATH}"
        )

    starters = pd.read_csv(
        HISTORICAL_STARTS_PATH
    )

    required_columns = {
        "date",
        "pitcher_id",
        "pitcher_name",
    }

    missing_columns = (
        required_columns
        - set(starters.columns)
    )

    if missing_columns:
        raise ValueError(
            "Historical pitcher-start archive is missing columns: "
            f"{sorted(missing_columns)}"
        )

    starters["date"] = pd.to_datetime(
        starters["date"],
        errors="coerce",
    )

    starters["pitcher_id"] = pd.to_numeric(
        starters["pitcher_id"],
        errors="coerce",
    )

    starters = starters.dropna(
        subset=[
            "date",
            "pitcher_id",
            "pitcher_name",
        ]
    ).copy()

    starters["pitcher_id"] = (
        starters["pitcher_id"]
        .astype("int64")
    )

    latest_start_date = starters[
        "date"
    ].max().date()

    cutoff_date = min(
        latest_start_date,
        get_target_date(),
    )

    unique_pitchers = (
        starters[
            [
                "pitcher_id",
                "pitcher_name",
            ]
        ]
        .drop_duplicates(
            subset=["pitcher_id"],
            keep="last",
        )
        .sort_values("pitcher_name")
        .reset_index(drop=True)
    )

    return unique_pitchers, cutoff_date


def fetch_pitcher_game_logs(
    pitcher_id: int,
    season: int,
) -> list[dict[str, Any]]:
    """Download one pitcher's full season game log with retries."""
    url = STATS_URL_TEMPLATE.format(
        pitcher_id=pitcher_id
    )

    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": str(season),
        "hydrate": "team,opponent,game",
    }

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_REQUEST_ATTEMPTS + 1,
    ):
        try:
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

        except requests.RequestException as exc:
            last_error = exc

            print(
                f"  Attempt {attempt} failed: {exc}"
            )

            if attempt < MAX_REQUEST_ATTEMPTS:
                time.sleep(
                    attempt * 1.5
                )

    raise RuntimeError(
        f"Could not download game logs for pitcher {pitcher_id}"
    ) from last_error


def build_log_row(
    game: dict[str, Any],
    pitcher_id: int,
    pitcher_name: str,
    season: int,
    cutoff_date: date,
) -> dict[str, Any] | None:
    """Convert one API game-log split into a cleaned row."""
    game_date = pd.to_datetime(
        game.get("date"),
        errors="coerce",
    )

    if pd.isna(game_date):
        return None

    if game_date.date() > cutoff_date:
        return None

    stat = game.get(
        "stat",
        {},
    )

    game_info = game.get(
        "game",
        {},
    )

    innings = stat.get(
        "inningsPitched"
    )

    outs_recorded = innings_to_outs(
        innings
    )

    innings_decimal = (
        float(outs_recorded) / 3.0
        if outs_recorded is not None
        else None
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

    whip = safe_divide(
        (
            walks + hits
            if walks is not None and hits is not None
            else None
        ),
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

    if (
        strikeout_rate_bf is not None
        and walk_rate_bf is not None
    ):
        k_minus_bb_rate = (
            strikeout_rate_bf
            - walk_rate_bf
        )
    else:
        k_minus_bb_rate = None

    home_runs_per_9 = safe_divide(
        (
            home_runs * 9.0
            if home_runs is not None
            else None
        ),
        innings_decimal,
    )

    hits_per_9 = safe_divide(
        (
            hits * 9.0
            if hits is not None
            else None
        ),
        innings_decimal,
    )

    walks_per_9 = safe_divide(
        (
            walks * 9.0
            if walks is not None
            else None
        ),
        innings_decimal,
    )

    strikeouts_per_9 = safe_divide(
        (
            strikeouts * 9.0
            if strikeouts is not None
            else None
        ),
        innings_decimal,
    )

    if (
        home_runs is not None
        and walks is not None
        and strikeouts is not None
    ):
        fip_numerator = (
            13.0 * home_runs
            + 3.0 * (
                walks
                + (hit_batters or 0.0)
                - (intentional_walks or 0.0)
            )
            - 2.0 * strikeouts
        )
    else:
        fip_numerator = None

    fip_component = safe_divide(
        fip_numerator,
        innings_decimal,
    )

    game_id = game_info.get(
        "gamePk"
    )

    if game_id is None:
        game_id = game.get(
            "gamePk"
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

    return {
        "season": season,
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_name,
        "game_id": game_id,
        "game_date": game_date.date().isoformat(),
        "game_team": extract_team_name(
            game.get("team")
        ),
        "opponent": extract_team_name(
            game.get("opponent")
        ),
        "is_home": is_home_value,
        "decision": stat.get("note"),
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
        "earned_runs": safe_number(
            stat.get("earnedRuns")
        ),
        "hit_batters": hit_batters,
        "wild_pitches": safe_number(
            stat.get("wildPitches")
        ),
        "wins": safe_integer(
            stat.get("wins")
        ),
        "losses": safe_integer(
            stat.get("losses")
        ),
        "saves": safe_integer(
            stat.get("saves")
        ),
        "holds": safe_integer(
            stat.get("holds")
        ),
        "era": safe_number(
            stat.get("era")
        ),
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


def load_existing_archive() -> pd.DataFrame:
    """Load an existing historical pitcher-game-log archive."""
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
    """Normalize and deduplicate the historical log archive."""
    archive = frame.copy()

    if archive.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    archive["pitcher_id"] = pd.to_numeric(
        archive["pitcher_id"],
        errors="coerce",
    )

    archive["game_id"] = pd.to_numeric(
        archive["game_id"],
        errors="coerce",
    )

    archive["game_date"] = pd.to_datetime(
        archive["game_date"],
        errors="coerce",
    )

    archive = archive.dropna(
        subset=[
            "pitcher_id",
            "game_date",
        ]
    ).copy()

    archive["pitcher_id"] = (
        archive["pitcher_id"]
        .astype("int64")
    )

    archive["_game_identity"] = np.where(
        archive["game_id"].notna(),
        archive["game_id"].astype(
            "Int64"
        ).astype(str),
        archive["game_date"].dt.strftime(
            "%Y-%m-%d"
        ),
    )

    archive = archive.drop_duplicates(
        subset=[
            "pitcher_id",
            "_game_identity",
            "game_date",
        ],
        keep="last",
    )

    archive = archive.sort_values(
        [
            "pitcher_id",
            "game_date",
        ]
    ).reset_index(drop=True)

    archive["game_date"] = (
        archive["game_date"]
        .dt.strftime("%Y-%m-%d")
    )

    return archive[
        OUTPUT_COLUMNS
    ]


def download_historical_pitcher_game_logs() -> pd.DataFrame:
    """Download and save detailed historical pitcher game logs."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pitchers, cutoff_date = (
        load_historical_starters()
    )

    season = cutoff_date.year

    existing = load_existing_archive()

    existing_pitchers = set(
        pd.to_numeric(
            existing.get(
                "pitcher_id",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    print("=" * 72)
    print("BACKFILLING HISTORICAL PITCHER GAME LOGS")
    print("=" * 72)
    print(f"Season: {season}")
    print(f"Cutoff date: {cutoff_date}")
    print(f"Unique pitchers: {len(pitchers):,}")
    print(f"Existing archive rows: {len(existing):,}")

    new_rows: list[
        dict[str, Any]
    ] = []

    skipped_pitchers = 0
    downloaded_pitchers = 0
    failed_pitchers: list[str] = []

    for position, pitcher in enumerate(
        pitchers.itertuples(
            index=False
        ),
        start=1,
    ):
        pitcher_id = int(
            pitcher.pitcher_id
        )

        pitcher_name = str(
            pitcher.pitcher_name
        )

        if pitcher_id in existing_pitchers:
            print(
                f"[{position}/{len(pitchers)}] "
                f"Skipping existing {pitcher_name}"
            )

            skipped_pitchers += 1
            continue

        print(
            f"[{position}/{len(pitchers)}] "
            f"Downloading {pitcher_name}"
        )

        try:
            splits = fetch_pitcher_game_logs(
                pitcher_id=pitcher_id,
                season=season,
            )
        except RuntimeError as exc:
            print(
                f"WARNING: {exc}"
            )

            failed_pitchers.append(
                pitcher_name
            )
            continue

        pitcher_row_count = 0

        for game in splits:
            row = build_log_row(
                game=game,
                pitcher_id=pitcher_id,
                pitcher_name=pitcher_name,
                season=season,
                cutoff_date=cutoff_date,
            )

            if row is None:
                continue

            new_rows.append(
                row
            )

            pitcher_row_count += 1

        print(
            f"  Game-log rows collected: "
            f"{pitcher_row_count}"
        )

        downloaded_pitchers += 1

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
    print("HISTORICAL PITCHER GAME-LOG BACKFILL COMPLETE")
    print("=" * 72)
    print(
        f"Downloaded pitchers: "
        f"{downloaded_pitchers:,}"
    )
    print(
        f"Skipped existing pitchers: "
        f"{skipped_pitchers:,}"
    )
    print(
        f"Failed pitchers: "
        f"{len(failed_pitchers):,}"
    )
    print(
        f"New game-log rows: "
        f"{len(new_frame):,}"
    )
    print(
        f"Total archive rows: "
        f"{len(archive):,}"
    )
    print(
        "Pitchers represented: "
        f"{archive['pitcher_id'].nunique() if not archive.empty else 0:,}"
    )
    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    if failed_pitchers:
        print(
            "Failed pitcher names: "
            + ", ".join(
                failed_pitchers
            )
        )

    return archive


if __name__ == "__main__":
    download_historical_pitcher_game_logs()
