"""Build the current-slate MLB pitcher master dataset.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Inputs:
    data/pitchers/<target-date>.csv
    data/pitcher_stats/<target-date>.csv
    data/features/pitcher_features.csv

Output:
    data/final/master_dataset.csv
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

PITCHERS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pitchers"
)

PITCHER_STATS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pitcher_stats"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "pitcher_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "master_dataset.csv"
)


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


def normalize_pitcher_id(
    frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Validate and normalize pitcher IDs."""
    normalized = frame.copy()

    if "pitcher_id" not in normalized.columns:
        raise KeyError(
            f"{label} is missing pitcher_id"
        )

    normalized["pitcher_id"] = pd.to_numeric(
        normalized["pitcher_id"],
        errors="coerce",
    )

    normalized = normalized.dropna(
        subset=["pitcher_id"]
    ).copy()

    normalized["pitcher_id"] = (
        normalized["pitcher_id"]
        .astype(int)
    )

    return normalized


def build_master_dataset(
    target_date: str | None = None,
) -> pd.DataFrame:
    """Merge slate, season, and recent-form pitcher data."""
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

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pitchers_path = (
        PITCHERS_DIRECTORY
        / f"{target_date}.csv"
    )

    stats_path = (
        PITCHER_STATS_DIRECTORY
        / f"{target_date}.csv"
    )

    if not pitchers_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher file: {pitchers_path}"
        )

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher stats file: {stats_path}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing pitcher features file: {FEATURES_PATH}"
        )

    pitchers = pd.read_csv(
        pitchers_path
    )

    stats = pd.read_csv(
        stats_path
    )

    features = pd.read_csv(
        FEATURES_PATH
    )

    required_pitcher_columns = {
        "pitcher_id",
        "pitcher_name",
        "team",
        "game_id",
        "side",
    }

    missing_pitcher_columns = (
        required_pitcher_columns
        - set(pitchers.columns)
    )

    if missing_pitcher_columns:
        raise KeyError(
            f"{pitchers_path} is missing columns: "
            f"{sorted(missing_pitcher_columns)}"
        )

    pitchers = normalize_pitcher_id(
        pitchers,
        str(pitchers_path),
    )

    stats = normalize_pitcher_id(
        stats,
        str(stats_path),
    )

    features = normalize_pitcher_id(
        features,
        str(FEATURES_PATH),
    )

    if "date" in stats.columns:
        stats["date"] = (
            stats["date"]
            .astype(str)
        )

        stats = stats.loc[
            stats["date"].eq(target_date)
        ].copy()

    if "date" in features.columns:
        features["date"] = (
            features["date"]
            .astype(str)
        )

        features = features.loc[
            features["date"].eq(target_date)
        ].copy()

    stats = stats.drop_duplicates(
        subset=["pitcher_id"],
        keep="last",
    )

    features = features.drop_duplicates(
        subset=["pitcher_id"],
        keep="last",
    )

    master = pitchers.merge(
        stats,
        on="pitcher_id",
        how="left",
        suffixes=(
            "",
            "_season",
        ),
        validate="many_to_one",
    )

    master = master.merge(
        features,
        on="pitcher_id",
        how="left",
        suffixes=(
            "",
            "_recent",
        ),
        validate="many_to_one",
    )

    master = master.drop_duplicates(
        subset=[
            "game_id",
            "pitcher_id",
        ],
        keep="first",
    ).reset_index(drop=True)

    master.insert(
        0,
        "slate_date",
        target_date,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.csv"
    )

    master.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )

    print("=" * 72)
    print("PITCHER MASTER DATASET COMPLETE")
    print("=" * 72)
    print(f"Slate date: {target_date}")
    print(
        f"Saved {len(master)} current-slate pitcher rows "
        f"to {OUTPUT_PATH}"
    )

    display_columns = [
        "pitcher_name",
        "team",
        "opponent",
        "games_started",
        "strikeouts",
        "season_k_per_start",
        "avg_k",
        "avg_whip",
        "avg_k_rate",
        "avg_fip_component",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in master.columns
    ]

    if display_columns:
        print(
            master[
                display_columns
            ]
            .head(30)
            .to_string(index=False)
        )

    return master


if __name__ == "__main__":
    build_master_dataset()
