"""Build a leakage-safe MLB hitter training dataset.

The dataset contains one row per completed hitter game. Every model feature is
calculated using only games that occurred before the target game.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD
    MLB_HITTER_LOG_PATH=/optional/custom/path.csv

Outputs:
    data/training/hitter_training_dataset.csv
    data/training/hitter_feature_manifest.csv
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAINING_DIRECTORY = PROJECT_ROOT / "data" / "training"
OUTPUT_PATH = TRAINING_DIRECTORY / "hitter_training_dataset.csv"
FEATURE_MANIFEST_PATH = (
    TRAINING_DIRECTORY / "hitter_feature_manifest.csv"
)

ROLLING_WINDOWS = (3, 5, 10, 20)
MINIMUM_PRIOR_GAMES = 3

# Update these values only if your target platform uses different scoring.
SINGLE_PTS = 3.0
DOUBLE_PTS = 5.0
TRIPLE_PTS = 8.0
HR_PTS = 10.0
RUN_PTS = 2.0
RBI_PTS = 2.0
WALK_PTS = 2.0
HBP_PTS = 2.0
SB_PTS = 5.0

RAW_STAT_COLUMNS = [
    "plate_appearances",
    "at_bats",
    "hits",
    "doubles",
    "triples",
    "home_runs",
    "total_bases",
    "runs",
    "rbi",
    "walks",
    "strikeouts",
    "stolen_bases",
    "hit_by_pitch",
]

ROLLING_STAT_COLUMNS = [
    "plate_appearances",
    "at_bats",
    "hits",
    "singles",
    "doubles",
    "triples",
    "home_runs",
    "total_bases",
    "runs",
    "rbi",
    "walks",
    "strikeouts",
    "stolen_bases",
    "hit_by_pitch",
    "hits_runs_rbis",
    "fantasy_score",
]

RATE_STAT_COLUMNS = [
    "hits",
    "singles",
    "doubles",
    "triples",
    "home_runs",
    "total_bases",
    "runs",
    "rbi",
    "walks",
    "strikeouts",
    "stolen_bases",
    "hit_by_pitch",
    "hits_runs_rbis",
    "fantasy_score",
]

TARGET_COLUMNS = [
    "target_hits",
    "target_total_bases",
    "target_home_runs",
    "target_runs",
    "target_rbi",
    "target_hits_runs_rbis",
    "target_fantasy_score",
]

IDENTIFIER_COLUMNS = [
    "date",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
]


def get_target_date() -> date:
    """Return the active slate date used to determine the season."""
    raw_value = os.getenv(
        "MLB_TARGET_DATE",
        date.today().isoformat(),
    )

    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc


def get_input_path() -> Path:
    """Return the hitter log path for the active MLB season."""
    custom_path = os.getenv("MLB_HITTER_LOG_PATH")

    if custom_path:
        return Path(custom_path).expanduser().resolve()

    season = get_target_date().year

    return (
        PROJECT_ROOT
        / "data"
        / "hitter_game_logs"
        / f"{season}.csv"
    )


def require_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Raise a useful error when required raw columns are unavailable."""
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "The hitter game log is missing required columns: "
            f"{missing}"
        )


def clean_numeric_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Convert listed columns to finite numeric values."""
    cleaned = frame.copy()

    for column in columns:
        if column not in cleaned.columns:
            cleaned[column] = 0.0

        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        ).fillna(0.0)

        cleaned[column] = cleaned[column].replace(
            [np.inf, -np.inf],
            0.0,
        )

    return cleaned


def normalize_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize common identity and context columns."""
    normalized = frame.copy()

    for column in [
        "player_name",
        "team",
        "opponent",
        "home_team",
        "away_team",
        "venue",
    ]:
        if column in normalized.columns:
            normalized[column] = (
                normalized[column]
                .astype("string")
                .str.strip()
            )

    return normalized


def add_derived_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model targets and platform fantasy scoring."""
    derived = frame.copy()

    derived["singles"] = (
        derived["hits"]
        - derived["doubles"]
        - derived["triples"]
        - derived["home_runs"]
    ).clip(lower=0)

    derived["hits_runs_rbis"] = (
        derived["hits"]
        + derived["runs"]
        + derived["rbi"]
    )

    derived["fantasy_score"] = (
        derived["singles"] * SINGLE_PTS
        + derived["doubles"] * DOUBLE_PTS
        + derived["triples"] * TRIPLE_PTS
        + derived["home_runs"] * HR_PTS
        + derived["runs"] * RUN_PTS
        + derived["rbi"] * RBI_PTS
        + derived["walks"] * WALK_PTS
        + derived["hit_by_pitch"] * HBP_PTS
        + derived["stolen_bases"] * SB_PTS
    )

    derived["target_hits"] = derived["hits"]
    derived["target_total_bases"] = derived["total_bases"]
    derived["target_home_runs"] = derived["home_runs"]
    derived["target_runs"] = derived["runs"]
    derived["target_rbi"] = derived["rbi"]
    derived["target_hits_runs_rbis"] = (
        derived["hits_runs_rbis"]
    )
    derived["target_fantasy_score"] = derived["fantasy_score"]

    return derived


def add_schedule_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add rest and calendar features known before the game."""
    featured = frame.copy()
    grouped = featured.groupby("player_id", sort=False)

    featured["days_rest"] = (
        grouped["date"]
        .diff()
        .dt.days
        .clip(lower=0, upper=14)
    )

    featured["days_rest"] = featured["days_rest"].fillna(7)

    featured["is_back_to_back"] = (
        featured["days_rest"] <= 1
    ).astype(int)

    featured["is_short_rest"] = (
        featured["days_rest"] <= 2
    ).astype(int)

    featured["month"] = featured["date"].dt.month
    featured["day_of_week"] = featured["date"].dt.dayofweek

    featured["prior_games"] = grouped.cumcount()

    return featured


def add_previous_game_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add previous-game results as pregame features."""
    featured = frame.copy()
    grouped = featured.groupby("player_id", sort=False)

    for stat in ROLLING_STAT_COLUMNS:
        if stat not in featured.columns:
            continue

        featured[f"previous_game_{stat}"] = (
            grouped[stat].shift(1)
        )

    return featured


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling form and volatility features."""
    featured = frame.copy()
    grouped = featured.groupby("player_id", sort=False)

    for window in ROLLING_WINDOWS:
        for stat in ROLLING_STAT_COLUMNS:
            if stat not in featured.columns:
                continue

            shifted = grouped[stat].shift(1)

            rolling = shifted.groupby(
                featured["player_id"],
                sort=False,
            ).rolling(
                window=window,
                min_periods=1,
            )

            prefix = f"last{window}_{stat}"

            featured[f"{prefix}_avg"] = (
                rolling.mean()
                .reset_index(level=0, drop=True)
            )

            featured[f"{prefix}_median"] = (
                rolling.median()
                .reset_index(level=0, drop=True)
            )

            featured[f"{prefix}_std"] = (
                rolling.std(ddof=0)
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

            featured[f"{prefix}_min"] = (
                rolling.min()
                .reset_index(level=0, drop=True)
            )

            featured[f"{prefix}_max"] = (
                rolling.max()
                .reset_index(level=0, drop=True)
            )

    return featured


def add_season_to_date_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add expanding season averages using previous games only."""
    featured = frame.copy()
    grouped = featured.groupby("player_id", sort=False)

    for stat in ROLLING_STAT_COLUMNS:
        if stat not in featured.columns:
            continue

        shifted = grouped[stat].shift(1)

        expanding = shifted.groupby(
            featured["player_id"],
            sort=False,
        ).expanding(min_periods=1)

        featured[f"season_avg_{stat}"] = (
            expanding.mean()
            .reset_index(level=0, drop=True)
        )

        featured[f"season_std_{stat}"] = (
            expanding.std(ddof=0)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    return featured


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide while replacing invalid results with missing values."""
    denominator = denominator.replace(0, np.nan)

    result = numerator / denominator

    return result.replace([np.inf, -np.inf], np.nan)


def add_rate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add opportunity-adjusted production rates."""
    featured = frame.copy()

    for window in ROLLING_WINDOWS:
        pa_column = f"last{window}_plate_appearances_avg"
        ab_column = f"last{window}_at_bats_avg"

        if pa_column in featured.columns:
            for stat in RATE_STAT_COLUMNS:
                stat_column = f"last{window}_{stat}_avg"

                if stat_column not in featured.columns:
                    continue

                featured[
                    f"last{window}_{stat}_per_pa"
                ] = safe_divide(
                    featured[stat_column],
                    featured[pa_column],
                )

        if ab_column in featured.columns:
            for stat in [
                "hits",
                "singles",
                "doubles",
                "triples",
                "home_runs",
                "total_bases",
                "strikeouts",
            ]:
                stat_column = f"last{window}_{stat}_avg"

                if stat_column not in featured.columns:
                    continue

                featured[
                    f"last{window}_{stat}_per_ab"
                ] = safe_divide(
                    featured[stat_column],
                    featured[ab_column],
                )

    for stat in RATE_STAT_COLUMNS:
        season_stat = f"season_avg_{stat}"
        season_pa = "season_avg_plate_appearances"

        if (
            season_stat in featured.columns
            and season_pa in featured.columns
        ):
            featured[f"season_{stat}_per_pa"] = safe_divide(
                featured[season_stat],
                featured[season_pa],
            )

    return featured


def add_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure short-term form relative to longer-term form."""
    featured = frame.copy()

    for stat in ROLLING_STAT_COLUMNS:
        last3 = f"last3_{stat}_avg"
        last5 = f"last5_{stat}_avg"
        last10 = f"last10_{stat}_avg"
        last20 = f"last20_{stat}_avg"
        season = f"season_avg_{stat}"

        if last3 in featured.columns and last10 in featured.columns:
            featured[f"trend_last3_vs_last10_{stat}"] = (
                featured[last3] - featured[last10]
            )

        if last5 in featured.columns and last20 in featured.columns:
            featured[f"trend_last5_vs_last20_{stat}"] = (
                featured[last5] - featured[last20]
            )

        if last10 in featured.columns and season in featured.columns:
            featured[f"trend_last10_vs_season_{stat}"] = (
                featured[last10] - featured[season]
            )

    return featured


def add_consistency_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add coefficients of variation for volatile hitter markets."""
    featured = frame.copy()

    for window in ROLLING_WINDOWS:
        for stat in [
            "hits",
            "total_bases",
            "runs",
            "rbi",
            "hits_runs_rbis",
            "fantasy_score",
        ]:
            average_column = f"last{window}_{stat}_avg"
            standard_deviation_column = f"last{window}_{stat}_std"

            if (
                average_column not in featured.columns
                or standard_deviation_column not in featured.columns
            ):
                continue

            featured[
                f"last{window}_{stat}_coefficient_variation"
            ] = safe_divide(
                featured[standard_deviation_column],
                featured[average_column].abs(),
            )

    return featured


def add_opportunity_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add features representing expected hitter opportunities."""
    featured = frame.copy()

    if "last5_plate_appearances_avg" in featured.columns:
        featured["expected_plate_appearances"] = (
            0.50 * featured["last5_plate_appearances_avg"]
            + 0.30 * featured["last10_plate_appearances_avg"]
            + 0.20 * featured["season_avg_plate_appearances"]
        )

    if "last5_at_bats_avg" in featured.columns:
        featured["expected_at_bats"] = (
            0.50 * featured["last5_at_bats_avg"]
            + 0.30 * featured["last10_at_bats_avg"]
            + 0.20 * featured["season_avg_at_bats"]
        )

    if (
        "expected_plate_appearances" in featured.columns
        and "season_hits_per_pa" in featured.columns
    ):
        featured["expected_hits_from_opportunity"] = (
            featured["expected_plate_appearances"]
            * featured["season_hits_per_pa"]
        )

    if (
        "expected_plate_appearances" in featured.columns
        and "season_hits_runs_rbis_per_pa" in featured.columns
    ):
        featured[
            "expected_hits_runs_rbis_from_opportunity"
        ] = (
            featured["expected_plate_appearances"]
            * featured["season_hits_runs_rbis_per_pa"]
        )

    if (
        "expected_plate_appearances" in featured.columns
        and "season_fantasy_score_per_pa" in featured.columns
    ):
        featured["expected_fantasy_from_opportunity"] = (
            featured["expected_plate_appearances"]
            * featured["season_fantasy_score_per_pa"]
        )

    return featured


def infer_home_indicator(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a home-game indicator when source context is available."""
    featured = frame.copy()

    if "is_home" in featured.columns:
        featured["is_home"] = (
            pd.to_numeric(
                featured["is_home"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
        return featured

    if {
        "team",
        "home_team",
    }.issubset(featured.columns):
        featured["is_home"] = (
            featured["team"].astype(str).str.casefold()
            == featured["home_team"].astype(str).str.casefold()
        ).astype(int)

    return featured


def add_split_features(
    frame: pd.DataFrame,
    split_column: str,
    split_name: str,
) -> pd.DataFrame:
    """Add prior-only averages for a categorical split."""
    if split_column not in frame.columns:
        return frame

    featured = frame.copy()

    valid_split = featured[split_column].notna()

    if not valid_split.any():
        return featured

    group_columns = ["player_id", split_column]
    grouped = featured.groupby(group_columns, sort=False)

    for stat in [
        "hits",
        "total_bases",
        "runs",
        "rbi",
        "hits_runs_rbis",
        "fantasy_score",
        "plate_appearances",
    ]:
        if stat not in featured.columns:
            continue

        shifted = grouped[stat].shift(1)

        expanding = shifted.groupby(
            [
                featured["player_id"],
                featured[split_column],
            ],
            sort=False,
        ).expanding(min_periods=1)

        featured[
            f"{split_name}_avg_{stat}"
        ] = (
            expanding.mean()
            .reset_index(level=[0, 1], drop=True)
        )

    return featured


def add_optional_context_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Use matchup context when it already exists in the source logs."""
    featured = infer_home_indicator(frame)

    if "is_home" in featured.columns:
        featured = add_split_features(
            featured,
            "is_home",
            "location_split",
        )

    if "opponent" in featured.columns:
        featured = add_split_features(
            featured,
            "opponent",
            "opponent_split",
        )

    if "batting_order" in featured.columns:
        featured["batting_order"] = pd.to_numeric(
            featured["batting_order"],
            errors="coerce",
        )

        featured["top_of_order"] = (
            featured["batting_order"].between(1, 3)
        ).astype(int)

        featured["middle_of_order"] = (
            featured["batting_order"].between(4, 6)
        ).astype(int)

        featured["bottom_of_order"] = (
            featured["batting_order"].between(7, 9)
        ).astype(int)

    return featured


def finalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and normalize feature values."""
    finalized = frame.copy()

    finalized = finalized.loc[
        finalized["prior_games"] >= MINIMUM_PRIOR_GAMES
    ].copy()

    finalized = finalized.dropna(
        subset=[
            "date",
            "player_id",
            *TARGET_COLUMNS,
        ]
    )

    numeric_columns = finalized.select_dtypes(
        include=[np.number]
    ).columns

    finalized[numeric_columns] = finalized[
        numeric_columns
    ].replace([np.inf, -np.inf], np.nan)

    # Do not fill the current-game targets from other games.
    feature_numeric_columns = [
        column
        for column in numeric_columns
        if column not in TARGET_COLUMNS
    ]

    finalized[feature_numeric_columns] = finalized[
        feature_numeric_columns
    ].fillna(0.0)

    finalized = finalized.sort_values(
        ["date", "game_id", "player_id"]
    ).reset_index(drop=True)

    return finalized


def identify_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return numeric pregame columns eligible for model training."""
    excluded_columns = set(
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
        + RAW_STAT_COLUMNS
        + [
            "singles",
            "hits_runs_rbis",
            "fantasy_score",
        ]
    )

    feature_columns: list[str] = []

    for column in frame.columns:
        if column in excluded_columns:
            continue

        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue

        feature_columns.append(column)

    return sorted(feature_columns)


def save_feature_manifest(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Save exact model feature names and basic metadata."""
    manifest_rows = []

    for position, column in enumerate(feature_columns):
        manifest_rows.append(
            {
                "feature_order": position,
                "feature_name": column,
                "dtype": str(frame[column].dtype),
                "missing_rate": float(
                    frame[column].isna().mean()
                ),
                "unique_values": int(
                    frame[column].nunique(dropna=True)
                ),
            }
        )

    manifest = pd.DataFrame(manifest_rows)

    manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
    )


def build_hitter_training_dataset() -> pd.DataFrame:
    """Build and save the production hitter training dataset."""
    input_path = get_input_path()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Hitter game log was not found: {input_path}"
        )

    TRAINING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Building leakage-safe hitter training dataset")
    print(f"Input: {input_path}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 72)

    frame = pd.read_csv(input_path)

    require_columns(
        frame,
        [
            "date",
            "game_id",
            "player_id",
        ],
    )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["date", "game_id", "player_id"]
    ).copy()

    frame = normalize_text_columns(frame)
    frame = clean_numeric_columns(
        frame,
        RAW_STAT_COLUMNS,
    )

    frame = frame.sort_values(
        ["player_id", "date", "game_id"]
    ).reset_index(drop=True)

    frame = add_derived_targets(frame)
    frame = add_schedule_features(frame)
    frame = add_previous_game_features(frame)
    frame = add_rolling_features(frame)
    frame = add_season_to_date_features(frame)
    frame = add_rate_features(frame)
    frame = add_trend_features(frame)
    frame = add_consistency_features(frame)
    frame = add_opportunity_features(frame)
    frame = add_optional_context_features(frame)
    frame = finalize_dataset(frame)

    feature_columns = identify_feature_columns(frame)

    frame.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    save_feature_manifest(
        frame,
        feature_columns,
    )

    print("\nHitter training dataset created successfully.")
    print(f"Rows: {len(frame):,}")
    print(f"Players: {frame['player_id'].nunique():,}")
    print(
        "Date range: "
        f"{frame['date'].min().date()} to "
        f"{frame['date'].max().date()}"
    )
    print(f"Feature count: {len(feature_columns):,}")
    print(f"Saved dataset: {OUTPUT_PATH}")
    print(f"Saved manifest: {FEATURE_MANIFEST_PATH}")

    preview_columns = [
        "date",
        "player_name",
        "prior_games",
        "days_rest",
        "expected_plate_appearances",
        "last5_hits_avg",
        "last10_hits_avg",
        "season_avg_hits",
        "last5_hits_runs_rbis_avg",
        "last5_fantasy_score_avg",
        "target_hits",
        "target_total_bases",
        "target_hits_runs_rbis",
        "target_fantasy_score",
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in frame.columns
    ]

    if preview_columns:
        print("\nPreview:")
        print(
            frame[preview_columns]
            .head(15)
            .to_string(index=False)
        )

    return frame


if __name__ == "__main__":
    build_hitter_training_dataset()
