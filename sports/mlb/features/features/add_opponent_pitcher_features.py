"""Build leakage-safe opponent-pitcher features for hitter models.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD
    MLB_HITTER_LOG_PATH=/optional/custom/hitter_logs.csv

Inputs:
    data/hitter_game_logs/<season>.csv
    data/historical/pitcher_starts.csv
    data/pitchers/*.csv
    data/game_logs/*.csv

Output:
    data/training/hitter_opponent_pitcher_features.csv

Pitcher statistics for each hitter game are calculated using only pitching
appearances that occurred before that hitter game.
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

HISTORICAL_PITCHER_STARTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "pitcher_starts.csv"
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

    return (
        HITTER_LOG_DIRECTORY
        / f"{get_target_date().year}.csv"
    )


def normalize_team_name(
    value: Any,
) -> str:
    """Normalize team names for reliable matching."""
    if value is None or pd.isna(value):
        return ""

    return (
        str(value)
        .casefold()
        .replace(".", "")
        .replace("-", " ")
        .replace("_", " ")
        .replace("  ", " ")
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
    ) as exc:
        print(
            f"WARNING: Could not read {path}: {exc}"
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
            "opponent",
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


def prepare_pitcher_slate(
    frame: pd.DataFrame,
    source_name: str,
    source_priority: int,
    fallback_date: str | None = None,
) -> pd.DataFrame:
    """Validate and normalize one pitcher-slate source."""
    slate = frame.copy()

    if slate.empty:
        return pd.DataFrame()

    required_columns = {
        "game_id",
        "pitcher_id",
        "pitcher_name",
        "team",
    }

    missing_columns = (
        required_columns
        - set(slate.columns)
    )

    if missing_columns:
        print(
            f"WARNING: Skipping {source_name}; missing columns: "
            f"{sorted(missing_columns)}"
        )

        return pd.DataFrame()

    if "date" not in slate.columns:
        if fallback_date is None:
            print(
                f"WARNING: Skipping {source_name}; no date column."
            )

            return pd.DataFrame()

        slate["date"] = fallback_date

    slate["date"] = pd.to_datetime(
        slate["date"],
        errors="coerce",
    )

    slate["game_id"] = pd.to_numeric(
        slate["game_id"],
        errors="coerce",
    )

    slate["pitcher_id"] = pd.to_numeric(
        slate["pitcher_id"],
        errors="coerce",
    )

    slate = slate.dropna(
        subset=[
            "date",
            "game_id",
            "pitcher_id",
            "pitcher_name",
            "team",
        ]
    ).copy()

    slate["game_id"] = (
        slate["game_id"]
        .astype("int64")
    )

    slate["pitcher_id"] = (
        slate["pitcher_id"]
        .astype("int64")
    )

    slate["team_key"] = (
        slate["team"]
        .apply(normalize_team_name)
    )

    if "opponent" in slate.columns:
        slate["opponent_key"] = (
            slate["opponent"]
            .apply(normalize_team_name)
        )
    else:
        slate["opponent_key"] = ""

    slate["_source_name"] = source_name
    slate["_source_priority"] = source_priority

    return slate


def load_pitcher_slates() -> pd.DataFrame:
    """Load historical and daily probable-pitcher files.

    The historical archive provides full-season coverage. Daily pitcher files
    receive higher priority when duplicate records are present.
    """
    frames: list[pd.DataFrame] = []

    if HISTORICAL_PITCHER_STARTS_PATH.exists():
        historical = safe_read_csv(
            HISTORICAL_PITCHER_STARTS_PATH
        )

        historical = prepare_pitcher_slate(
            frame=historical,
            source_name=str(
                HISTORICAL_PITCHER_STARTS_PATH
            ),
            source_priority=1,
        )

        if not historical.empty:
            frames.append(
                historical
            )

            print(
                "Loaded historical pitcher-start archive: "
                f"{len(historical):,} rows"
            )
    else:
        print(
            "WARNING: Historical pitcher archive was not found: "
            f"{HISTORICAL_PITCHER_STARTS_PATH}"
        )

    if PITCHER_SLATE_DIRECTORY.exists():
        daily_files = sorted(
            PITCHER_SLATE_DIRECTORY.glob(
                "*.csv"
            )
        )

        daily_row_count = 0

        for path in daily_files:
            slate = safe_read_csv(path)

            slate = prepare_pitcher_slate(
                frame=slate,
                source_name=str(path),
                source_priority=2,
                fallback_date=path.stem,
            )

            if slate.empty:
                continue

            frames.append(
                slate
            )

            daily_row_count += len(
                slate
            )

        print(
            "Loaded daily pitcher-slate rows: "
            f"{daily_row_count:,}"
        )
    else:
        print(
            "WARNING: Pitcher slate directory does not exist: "
            f"{PITCHER_SLATE_DIRECTORY}"
        )

    if not frames:
        return pd.DataFrame()

    pitchers = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    pitchers = pitchers.sort_values(
        [
            "date",
            "game_id",
            "pitcher_id",
            "_source_priority",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    # Daily files have priority because they are appended after historical
    # rows and use the greater source-priority value.
    pitchers = pitchers.drop_duplicates(
        subset=[
            "date",
            "game_id",
            "pitcher_id",
        ],
        keep="last",
    )

    pitchers = pitchers.sort_values(
        [
            "date",
            "game_id",
            "pitcher_id",
        ]
    ).reset_index(drop=True)

    historical_dates = int(
        pitchers["date"].nunique()
    )

    unique_games = int(
        pitchers["game_id"].nunique()
    )

    print(
        f"Total archived pitcher rows: "
        f"{len(pitchers):,}"
    )

    print(
        f"Pitcher dates represented: "
        f"{historical_dates:,}"
    )

    print(
        f"Games represented: "
        f"{unique_games:,}"
    )

    return pitchers


def build_hitter_pitcher_matchups(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the correct opposing starter to each hitter game."""
    result = hitters.copy()

    if pitchers.empty:
        result["opponent_pitcher_id"] = np.nan
        result["opponent_pitcher_name"] = pd.NA

        print(
            "WARNING: No pitcher slates were available. "
            "Opponent-pitcher coverage will be zero."
        )

        return result

    # Each pitcher row identifies the team the pitcher played for.
    # A hitter whose opponent matches that team faced that pitcher.
    matchup_map = pitchers[
        [
            "date",
            "game_id",
            "team_key",
            "pitcher_id",
            "pitcher_name",
            "_source_priority",
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

    matchup_map = matchup_map.sort_values(
        "_source_priority"
    )

    matchup_map = matchup_map.drop_duplicates(
        subset=[
            "date",
            "game_id",
            "opponent_key",
        ],
        keep="last",
    )

    matchup_map = matchup_map.drop(
        columns=[
            "_source_priority",
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
        result[
            "opponent_pitcher_id"
        ],
        errors="coerce",
    )

    return result


def innings_to_outs(
    value: Any,
) -> float:
    """Convert baseball innings notation to recorded outs."""
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


def safe_divide_series(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide two numeric series safely."""
    denominator = denominator.replace(
        0,
        np.nan,
    )

    result = numerator / denominator

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def add_missing_pitcher_rates(
    logs: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate pitcher rate columns when older files lack them."""
    enriched = logs.copy()

    innings_decimal = (
        enriched["outs_recorded"]
        / 3.0
    )

    enriched["whip"] = (
        enriched["whip"]
        .fillna(
            safe_divide_series(
                enriched["walks"]
                + enriched["hits"],
                innings_decimal,
            )
        )
    )

    enriched["strikeout_rate_bf"] = (
        enriched["strikeout_rate_bf"]
        .fillna(
            safe_divide_series(
                enriched["strikeouts"],
                enriched["batters_faced"],
            )
        )
    )

    enriched["walk_rate_bf"] = (
        enriched["walk_rate_bf"]
        .fillna(
            safe_divide_series(
                enriched["walks"],
                enriched["batters_faced"],
            )
        )
    )

    enriched["k_minus_bb_rate"] = (
        enriched["k_minus_bb_rate"]
        .fillna(
            enriched["strikeout_rate_bf"]
            - enriched["walk_rate_bf"]
        )
    )

    enriched["home_runs_per_9"] = (
        enriched["home_runs_per_9"]
        .fillna(
            safe_divide_series(
                enriched["home_runs"]
                * 9.0,
                innings_decimal,
            )
        )
    )

    enriched["hits_per_9"] = (
        enriched["hits_per_9"]
        .fillna(
            safe_divide_series(
                enriched["hits"]
                * 9.0,
                innings_decimal,
            )
        )
    )

    enriched["walks_per_9"] = (
        enriched["walks_per_9"]
        .fillna(
            safe_divide_series(
                enriched["walks"]
                * 9.0,
                innings_decimal,
            )
        )
    )

    enriched["strikeouts_per_9"] = (
        enriched["strikeouts_per_9"]
        .fillna(
            safe_divide_series(
                enriched["strikeouts"]
                * 9.0,
                innings_decimal,
            )
        )
    )

    enriched["fip_component"] = (
        enriched["fip_component"]
        .fillna(
            safe_divide_series(
                (
                    13.0
                    * enriched["home_runs"]
                    + 3.0
                    * enriched["walks"]
                    - 2.0
                    * enriched["strikeouts"]
                ),
                innings_decimal,
            )
        )
    )

    return enriched


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

        logs["_source_file"] = str(
            path
        )

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
        logs["game_id"] = np.nan

    if "outs_recorded" not in logs.columns:
        if "innings" in logs.columns:
            logs["outs_recorded"] = (
                logs["innings"]
                .apply(innings_to_outs)
            )
        else:
            logs["outs_recorded"] = np.nan

    for column in PITCHER_STAT_COLUMNS:
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

    logs = add_missing_pitcher_rates(
        logs
    )

    # Older rows may not have a game ID. In that case, pitcher and game date
    # still provide a stable identity for one appearance.
    logs["_game_identity"] = np.where(
        logs["game_id"].notna(),
        logs["game_id"].astype(
            "Int64"
        ).astype(str),
        logs["game_date"].dt.strftime(
            "%Y-%m-%d"
        ),
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

    for stat in PITCHER_STAT_COLUMNS:
        season_average_name = (
            f"opponent_pitcher_season_avg_{stat}"
        )

        season_std_name = (
            f"opponent_pitcher_season_std_{stat}"
        )

        if prior.empty:
            features[
                season_average_name
            ] = float("nan")

            features[
                season_std_name
            ] = float("nan")

            for window in ROLLING_WINDOWS:
                features[
                    f"opponent_pitcher_last{window}_avg_{stat}"
                ] = float("nan")

            continue

        features[
            season_average_name
        ] = safe_mean(
            prior[stat]
        )

        features[
            season_std_name
        ] = safe_std(
            prior[stat]
        )

        for window in ROLLING_WINDOWS:
            features[
                f"opponent_pitcher_last{window}_avg_{stat}"
            ] = safe_mean(
                prior.tail(window)[stat]
            )

    return features


def add_pitcher_profiles(
    matchups: pd.DataFrame,
    pitcher_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate one prior-only pitcher profile per unique matchup."""
    result = matchups.copy()

    valid_matchups = (
        result.dropna(
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
        ]
        .drop_duplicates()
    )

    if valid_matchups.empty:
        print(
            "WARNING: No hitter rows matched an opposing pitcher."
        )

        result[
            "opponent_pitcher_match_available"
        ] = 0

        result[
            "opponent_pitcher_history_available"
        ] = 0

        result[
            "opponent_pitcher_history_games"
        ] = 0.0

        return result

    if pitcher_logs.empty:
        print(
            "WARNING: No pitcher game logs were available."
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
        ] = 0

        result[
            "opponent_pitcher_history_games"
        ] = 0.0

        return result

    pitcher_log_groups = {
        int(pitcher_id): group.sort_values(
            "game_date"
        ).copy()
        for pitcher_id, group
        in pitcher_logs.groupby(
            "pitcher_id",
            sort=False,
        )
    }

    profile_rows: list[
        dict[str, Any]
    ] = []

    total_matchups = len(
        valid_matchups
    )

    for position, matchup in enumerate(
        valid_matchups.itertuples(
            index=False
        ),
        start=1,
    ):
        pitcher_id = int(
            matchup.opponent_pitcher_id
        )

        pitcher_history = pitcher_log_groups.get(
            pitcher_id,
            pd.DataFrame(
                columns=pitcher_logs.columns
            ),
        )

        profile = calculate_profile(
            pitcher_logs=pitcher_history,
            game_date=pd.Timestamp(
                matchup.date
            ),
        )

        profile_rows.append(
            {
                "date": pd.Timestamp(
                    matchup.date
                ),
                "game_id": int(
                    matchup.game_id
                ),
                "opponent_pitcher_id": (
                    pitcher_id
                ),
                **profile,
            }
        )

        if position % 100 == 0:
            print(
                f"Built {position:,} of "
                f"{total_matchups:,} pitcher profiles."
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
        "opponent_pitcher_history_games"
    ] = pd.to_numeric(
        result.get(
            "opponent_pitcher_history_games",
            0,
        ),
        errors="coerce",
    ).fillna(0.0)

    result[
        "opponent_pitcher_history_available"
    ] = (
        result[
            "opponent_pitcher_history_games"
        ]
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
        include=[
            np.number,
        ]
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
        if (
            column.startswith(
                "opponent_pitcher_"
            )
            and column
            not in identifier_columns
        )
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
        "Hitter rows matched to an opposing starter: "
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
        pd.to_numeric(
            output.get(
                "opponent_pitcher_history_available",
                pd.Series(
                    0,
                    index=output.index,
                    dtype=float,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    feature_columns = [
        column
        for column in output.columns
        if column.startswith(
            "opponent_pitcher_"
        )
    ]

    populated_feature_columns = [
        column
        for column in feature_columns
        if (
            column
            not in {
                "opponent_pitcher_name",
            }
            and pd.to_numeric(
                output[column],
                errors="coerce",
            )
            .notna()
            .any()
        )
    ]

    print("\n" + "=" * 72)
    print("OPPONENT-PITCHER FEATURES COMPLETE")
    print("=" * 72)

    print(
        f"Output rows: "
        f"{len(output):,}"
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
        "Opponent-pitcher columns created: "
        f"{len(feature_columns):,}"
    )

    print(
        "Populated opponent-pitcher columns: "
        f"{len(populated_feature_columns):,}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
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
        preview = output[
            preview_columns
        ].copy()

        if (
            "opponent_pitcher_name"
            in preview.columns
        ):
            preview = preview.dropna(
                subset=[
                    "opponent_pitcher_name",
                ]
            )

        print("\nPreview:")

        print(
            preview
            .head(30)
            .to_string(index=False)
        )

    return output


if __name__ == "__main__":
    build_opponent_pitcher_features()
