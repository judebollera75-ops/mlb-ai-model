"""Build leakage-safe opponent-pitcher features for hitter models.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD
    MLB_HITTER_LOG_PATH=/optional/custom/hitter_logs.csv

Inputs:
    data/hitter_game_logs/<season>.csv
    data/pitchers/*.csv
    data/game_logs/*.csv

Output:
    data/training/hitter_opponent_pitcher_features.csv

Important:
    Pitcher statistics for a hitter game are calculated using only pitching
    appearances that occurred before that hitter game. Current-game or future
    pitcher results are never included.

    Historical matchup coverage depends on having archived files in
    data/pitchers/. Hitter rows without a known opposing starter are preserved
    and receive missing opponent-pitcher features.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

HITTER_LOG_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "hitter_game_logs"
)

PITCHER_SLATE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pitchers"
)

PITCHER_LOG_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "game_logs"
)

TRAINING_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "training"
)

OUTPUT_PATH = (
    TRAINING_DIRECTORY
    / "hitter_opponent_pitcher_features.csv"
)

ROLLING_WINDOWS = (
    3,
    5,
    10,
)

PITCHER_STAT_COLUMNS = [
    "outs_recorded",
    "batters_faced",
    "strikeouts",
    "walks",
    "hits",
    "earned_runs",
    "home_runs",
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

OUTPUT_IDENTIFIER_COLUMNS = [
    "date",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "opponent_pitcher_id",
    "opponent_pitcher_name",
]


def get_target_date() -> date:
    """Return the active MLB slate date."""
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


def get_hitter_log_path() -> Path:
    """Return the hitter season-log path."""
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


def normalize_team_name(
    value: Any,
) -> str:
    """Normalize team names for fallback matching."""
    if value is None or pd.isna(value):
        return ""

    return (
        str(value)
        .casefold()
        .replace(".", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


def safe_read_csv(
    path: Path,
) -> pd.DataFrame:
    """Read a CSV safely."""
    try:
        return pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        print(
            f"WARNING: Could not read {path}"
        )

        return pd.DataFrame()


def load_hitter_logs() -> pd.DataFrame:
    """Load historical hitter game logs."""
    path = get_hitter_log_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Hitter game-log file was not found: {path}"
        )

    hitters = safe_read_csv(path)

    required_columns = {
        "date",
        "game_id",
        "player_id",
        "team",
        "opponent",
    }

    missing_columns = (
        required_columns
        - set(hitters.columns)
    )

    if missing_columns:
        raise ValueError(
            "Hitter game logs are missing columns: "
            f"{sorted(missing_columns)}"
        )

    hitters["date"] = pd.to_datetime(
        hitters["date"],
        errors="coerce",
    )

    hitters["game_id"] = pd.to_numeric(
        hitters["game_id"],
        errors="coerce",
    )

    hitters["player_id"] = pd.to_numeric(
        hitters["player_id"],
        errors="coerce",
    )

    hitters = hitters.dropna(
        subset=[
            "date",
            "game_id",
            "player_id",
            "team",
        ]
    ).copy()

    hitters["game_id"] = (
        hitters["game_id"]
        .astype("int64")
    )

    hitters["player_id"] = (
        hitters["player_id"]
        .astype("int64")
    )

    hitters["team_key"] = (
        hitters["team"]
        .apply(normalize_team_name)
    )

    hitters["opponent_key"] = (
        hitters["opponent"]
        .apply(normalize_team_name)
    )

    hitters = hitters.drop_duplicates(
        subset=[
            "date",
            "game_id",
            "player_id",
        ],
        keep="last",
    )

    return hitters.sort_values(
        [
            "date",
            "game_id",
            "player_id",
        ]
    ).reset_index(drop=True)


def load_pitcher_slates() -> pd.DataFrame:
    """Load all archived probable-pitcher slate files."""
    if not PITCHER_SLATE_DIRECTORY.exists():
        print(
            "WARNING: Pitcher slate directory does not exist: "
            f"{PITCHER_SLATE_DIRECTORY}"
        )

        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for path in sorted(
        PITCHER_SLATE_DIRECTORY.glob(
            "*.csv"
        )
    ):
        slate = safe_read_csv(path)

        if slate.empty:
            continue

        required_columns = {
            "game_id",
            "pitcher_id",
            "pitcher_name",
            "team",
        }

        if not required_columns.issubset(
            slate.columns
        ):
            print(
                f"WARNING: Skipping {path}; required "
                "pitcher columns are missing."
            )
            continue

        if "date" not in slate.columns:
            slate["date"] = path.stem

        frames.append(
            slate
        )

    if not frames:
        return pd.DataFrame()

    pitchers = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    pitchers["date"] = pd.to_datetime(
        pitchers["date"],
        errors="coerce",
    )

    pitchers["game_id"] = pd.to_numeric(
        pitchers["game_id"],
        errors="coerce",
    )

    pitchers["pitcher_id"] = pd.to_numeric(
        pitchers["pitcher_id"],
        errors="coerce",
    )

    pitchers = pitchers.dropna(
        subset=[
            "date",
            "game_id",
            "pitcher_id",
            "pitcher_name",
            "team",
        ]
    ).copy()

    pitchers["game_id"] = (
        pitchers["game_id"]
        .astype("int64")
    )

    pitchers["pitcher_id"] = (
        pitchers["pitcher_id"]
        .astype("int64")
    )

    pitchers["team_key"] = (
        pitchers["team"]
        .apply(normalize_team_name)
    )

    if "opponent" in pitchers.columns:
        pitchers["opponent_key"] = (
            pitchers["opponent"]
            .apply(normalize_team_name)
        )
    else:
        pitchers[
            "opponent_key"
        ] = ""

    pitchers = pitchers.drop_duplicates(
        subset=[
            "date",
            "game_id",
            "pitcher_id",
        ],
        keep="last",
    )

    return pitchers.sort_values(
        [
            "date",
            "game_id",
            "pitcher_id",
        ]
    ).reset_index(drop=True)


def build_hitter_pitcher_matchups(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the opposing starter to every available hitter game."""
    result = hitters.copy()

    result[
        "opponent_pitcher_id"
    ] = pd.NA

    result[
        "opponent_pitcher_name"
    ] = pd.NA

    if pitchers.empty:
        print(
            "WARNING: No archived pitcher slates were found. "
            "Opponent-pitcher matchup coverage will be zero."
        )

        return result

    # Every pitcher row belongs to one team. Hitters on the pitcher's
    # opponent team faced that pitcher.
    matchup_map = pitchers[
        [
            "date",
            "game_id",
            "team_key",
            "pitcher_id",
            "pitcher_name",
        ]
    ].copy()

    matchup_map = matchup_map.rename(
        columns={
            "team_key": "opponent_key",
            "pitcher_id": (
                "opponent_pitcher_id"
            ),
            "pitcher_name": (
                "opponent_pitcher_name"
            ),
        }
    )

    matchup_map = matchup_map.drop_duplicates(
        subset=[
            "date",
            "game_id",
            "opponent_key",
        ],
        keep="last",
    )

    result = result.drop(
        columns=[
            "opponent_pitcher_id",
            "opponent_pitcher_name",
        ]
    )

    result = result.merge(
        matchup_map,
        on=[
            "date",
            "game_id",
            "opponent_key",
        ],
        how="left",
        validate="many_to_one",
    )

    result[
        "opponent_pitcher_id"
    ] = pd.to_numeric(
        result["opponent_pitcher_id"],
        errors="coerce",
    )

    return result


def innings_to_outs(
    value: Any,
) -> float:
    """Convert baseball innings notation to outs."""
    if value is None or pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if not text:
        return float("nan")

    if "." in text:
        whole_text, partial_text = (
            text.split(".", 1)
        )
    else:
        whole_text, partial_text = (
            text,
            "0",
        )

    try:
        whole_innings = int(
            whole_text
        )

        partial_outs = int(
            partial_text[:1] or "0"
        )
    except (
        TypeError,
        ValueError,
    ):
        return float("nan")

    if partial_outs not in {
        0,
        1,
        2,
    }:
        return float("nan")

    return float(
        whole_innings * 3
        + partial_outs
    )


def load_pitcher_game_logs() -> pd.DataFrame:
    """Combine and deduplicate all archived pitcher-log snapshots."""
    if not PITCHER_LOG_DIRECTORY.exists():
        print(
            "WARNING: Pitcher game-log directory does not exist: "
            f"{PITCHER_LOG_DIRECTORY}"
        )

        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for path in sorted(
        PITCHER_LOG_DIRECTORY.glob(
            "*.csv"
        )
    ):
        logs = safe_read_csv(path)

        if logs.empty:
            continue

        required_columns = {
            "pitcher_id",
            "game_date",
        }

        if not required_columns.issubset(
            logs.columns
        ):
            print(
                f"WARNING: Skipping {path}; required "
                "game-log columns are missing."
            )
            continue

        frames.append(
            logs
        )

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    logs["pitcher_id"] = pd.to_numeric(
        logs["pitcher_id"],
        errors="coerce",
    )

    logs["game_date"] = pd.to_datetime(
        logs["game_date"],
        errors="coerce",
    )

    if "game_id" in logs.columns:
        logs["game_id"] = pd.to_numeric(
            logs["game_id"],
            errors="coerce",
        )
    else:
        logs["game_id"] = pd.NA

    if (
        "outs_recorded"
        not in logs.columns
    ):
        if "innings" in logs.columns:
            logs[
                "outs_recorded"
            ] = logs[
                "innings"
            ].apply(
                innings_to_outs
            )
        else:
            logs[
                "outs_recorded"
            ] = np.nan

    numeric_columns = set(
        PITCHER_STAT_COLUMNS
        + [
            "pitcher_id",
            "game_id",
        ]
    )

    for column in numeric_columns:
        if column not in logs.columns:
            logs[column] = np.nan

        logs[column] = pd.to_numeric(
            logs[column],
            errors="coerce",
        )

    logs = logs.dropna(
        subset=[
            "pitcher_id",
            "game_date",
        ]
    ).copy()

    logs["pitcher_id"] = (
        logs["pitcher_id"]
        .astype("int64")
    )

    # Older snapshots may contain repeated copies of the same
    # historical pitching performance.
    logs["_game_identity"] = (
        logs["game_id"]
        .astype("string")
        .fillna(
            logs[
                "game_date"
            ].dt.strftime("%Y-%m-%d")
        )
    )

    logs = logs.drop_duplicates(
        subset=[
            "pitcher_id",
            "_game_identity",
            "game_date",
        ],
        keep="last",
    )

    logs = logs.sort_values(
        [
            "pitcher_id",
            "game_date",
        ]
    ).reset_index(drop=True)

    return logs


def safe_mean(
    values: pd.Series,
) -> float:
    """Return a finite mean or missing value."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric = numeric.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if numeric.empty:
        return float("nan")

    return float(
        numeric.mean()
    )


def safe_std(
    values: pd.Series,
) -> float:
    """Return a finite population standard deviation."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric = numeric.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if numeric.empty:
        return float("nan")

    return float(
        numeric.std(
            ddof=0
        )
    )


def calculate_profile(
    pitcher_logs: pd.DataFrame,
    game_date: pd.Timestamp,
) -> dict[str, float]:
    """Calculate prior-only pitcher features for one matchup."""
    prior = pitcher_logs.loc[
        pitcher_logs["game_date"]
        < game_date
    ].copy()

    features: dict[str, float] = {
        "opponent_pitcher_history_games": float(
            len(prior)
        )
    }

    if prior.empty:
        for stat in PITCHER_STAT_COLUMNS:
            features[
                f"opponent_pitcher_season_avg_{stat}"
            ] = float("nan")

            features[
                f"opponent_pitcher_season_std_{stat}"
            ] = float("nan")

            for window in ROLLING_WINDOWS:
                features[
                    f"opponent_pitcher_last{window}_avg_{stat}"
                ] = float("nan")

        return features

    prior = prior.sort_values(
        "game_date"
    )

    for stat in PITCHER_STAT_COLUMNS:
        if stat not in prior.columns:
            continue

        features[
            f"opponent_pitcher_season_avg_{stat}"
        ] = safe_mean(
            prior[stat]
        )

        features[
            f"opponent_pitcher_season_std_{stat}"
        ] = safe_std(
            prior[stat]
        )

        for window in ROLLING_WINDOWS:
            recent = prior.tail(
                window
            )

            features[
                f"opponent_pitcher_last{window}_avg_{stat}"
            ] = safe_mean(
                recent[stat]
            )

    return features


def add_pitcher_profiles(
    matchups: pd.DataFrame,
    pitcher_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate one pitcher profile per unique hitter matchup."""
    result = matchups.copy()

    valid_matchups = result.dropna(
        subset=[
            "date",
            "game_id",
            "opponent_pitcher_id",
        ]
    )[
        [
            "date",
            "game_id",
            "opponent_pitcher_id",
        ]
    ].drop_duplicates()

    if valid_matchups.empty:
        print(
            "WARNING: No hitter rows matched an opposing pitcher."
        )

        result[
            "opponent_pitcher_match_available"
        ] = 0

        return result

    pitcher_log_groups = {
        int(pitcher_id): group.copy()
        for pitcher_id, group
        in pitcher_logs.groupby(
            "pitcher_id",
            sort=False,
        )
    }

    profile_rows: list[
        dict[str, Any]
    ] = []

    total = len(
        valid_matchups
    )

    for count, matchup in enumerate(
        valid_matchups.itertuples(
            index=False
        ),
        start=1,
    ):
        pitcher_id = int(
            matchup.opponent_pitcher_id
        )

        pitcher_history = (
            pitcher_log_groups.get(
                pitcher_id,
                pd.DataFrame(
                    columns=pitcher_logs.columns
                ),
            )
        )

        profile = calculate_profile(
            pitcher_logs=pitcher_history,
            game_date=pd.Timestamp(
                matchup.date
            ),
        )

        profile_rows.append(
            {
                "date": matchup.date,
                "game_id": int(
                    matchup.game_id
                ),
                "opponent_pitcher_id": (
                    pitcher_id
                ),
                **profile,
            }
        )

        if count % 100 == 0:
            print(
                f"Built {count:,} of "
                f"{total:,} pitcher matchup profiles."
            )

    profiles = pd.DataFrame(
        profile_rows
    )

    result = result.merge(
        profiles,
        on=[
            "date",
            "game_id",
            "opponent_pitcher_id",
        ],
        how="left",
        validate="many_to_one",
    )

    result[
        "opponent_pitcher_match_available"
    ] = (
        result[
            "opponent_pitcher_id"
        ].notna()
    ).astype(int)

    result[
        "opponent_pitcher_history_available"
    ] = (
        pd.to_numeric(
            result.get(
                "opponent_pitcher_history_games",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    return result


def finalize_output(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Finalize output without dropping unmatched hitter games."""
    output = frame.copy()

    output["date"] = pd.to_datetime(
        output["date"],
        errors="coerce",
    )

    output = output.dropna(
        subset=[
            "date",
            "game_id",
            "player_id",
        ]
    ).copy()

    numeric_columns = output.select_dtypes(
        include=[np.number]
    ).columns

    output[numeric_columns] = (
        output[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    output = output.drop_duplicates(
        subset=[
            "date",
            "game_id",
            "player_id",
        ],
        keep="last",
    )

    output = output.sort_values(
        [
            "date",
            "game_id",
            "player_id",
        ]
    ).reset_index(drop=True)

    output["date"] = (
        output["date"]
        .dt.strftime("%Y-%m-%d")
    )

    helper_columns = [
        "team_key",
        "opponent_key",
    ]

    output = output.drop(
        columns=[
            column
            for column in helper_columns
            if column in output.columns
        ]
    )

    identifier_columns = [
        column
        for column in OUTPUT_IDENTIFIER_COLUMNS
        if column in output.columns
    ]

    feature_columns = [
        column
        for column in output.columns
        if column.startswith(
            "opponent_pitcher_"
        )
        and column
        not in identifier_columns
    ]

    remaining_columns = [
        column
        for column in output.columns
        if column not in (
            identifier_columns
            + feature_columns
        )
    ]

    return output[
        identifier_columns
        + feature_columns
        + remaining_columns
    ]


def save_output(
    frame: pd.DataFrame,
) -> None:
    """Save output atomically."""
    TRAINING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        OUTPUT_PATH.with_suffix(
            ".tmp.csv"
        )
    )

    frame.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )


def build_opponent_pitcher_features() -> pd.DataFrame:
    """Build and save opponent-pitcher features."""
    print("=" * 72)
    print("BUILDING OPPONENT-PITCHER FEATURES")
    print("=" * 72)

    hitters = load_hitter_logs()

    print(
        f"Hitter rows loaded: "
        f"{len(hitters):,}"
    )

    pitcher_slates = (
        load_pitcher_slates()
    )

    print(
        f"Archived pitcher slate rows: "
        f"{len(pitcher_slates):,}"
    )

    matchups = (
        build_hitter_pitcher_matchups(
            hitters=hitters,
            pitchers=pitcher_slates,
        )
    )

    matched_count = int(
        matchups[
            "opponent_pitcher_id"
        ].notna().sum()
    )

    print(
        f"Hitter rows matched to an opposing starter: "
        f"{matched_count:,} of {len(matchups):,}"
    )

    pitcher_logs = (
        load_pitcher_game_logs()
    )

    print(
        f"Unique pitcher game-log rows: "
        f"{len(pitcher_logs):,}"
    )

    enriched = add_pitcher_profiles(
        matchups=matchups,
        pitcher_logs=pitcher_logs,
    )

    output = finalize_output(
        enriched
    )

    save_output(
        output
    )

    match_rate = (
        matched_count
        / len(output)
        if len(output)
        else 0.0
    )

    history_available = int(
        output.get(
            "opponent_pitcher_history_available",
            pd.Series(
                dtype=float
            ),
        )
        .fillna(0)
        .sum()
    )

    print("\n" + "=" * 72)
    print("OPPONENT-PITCHER FEATURES COMPLETE")
    print("=" * 72)
    print(
        f"Output rows: {len(output):,}"
    )
    print(
        f"Starter matchup coverage: "
        f"{match_rate:.1%}"
    )
    print(
        "Rows with pitcher history available: "
        f"{history_available:,}"
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    preview_columns = [
        "date",
        "player_name",
        "team",
        "opponent",
        "opponent_pitcher_name",
        "opponent_pitcher_history_games",
        "opponent_pitcher_last5_avg_era",
        "opponent_pitcher_last5_avg_whip",
        "opponent_pitcher_last5_avg_strikeout_rate_bf",
        "opponent_pitcher_last5_avg_walk_rate_bf",
        "opponent_pitcher_last5_avg_home_runs_per_9",
        "opponent_pitcher_last5_avg_fip_component",
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in output.columns
    ]

    if preview_columns:
        print("\nPreview:")

        print(
            output[
                preview_columns
            ]
            .dropna(
                subset=[
                    "opponent_pitcher_name"
                ]
            )
            .head(30)
            .to_string(index=False)
        )

    return output


if __name__ == "__main__":
    build_opponent_pitcher_features()
