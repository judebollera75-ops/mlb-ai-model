"""Build current-slate pitcher features from historical game logs.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Input:
    data/game_logs/<target-date>.csv

Output:
    data/features/pitcher_features.csv
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

GAME_LOGS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "game_logs"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "pitcher_features.csv"
)

RECENT_GAME_LIMIT = 5

OUTPUT_COLUMNS = [
    "date",
    "pitcher_id",
    "pitcher_name",
    "team",
    "recent_games",
    "avg_ip",
    "avg_outs",
    "avg_k",
    "avg_bb",
    "avg_hits",
    "avg_er",
    "avg_hr",
    "avg_era",
    "avg_whip",
    "avg_batters_faced",
    "avg_k_rate",
    "avg_bb_rate",
    "avg_k_minus_bb_rate",
    "avg_k_per_9",
    "avg_bb_per_9",
    "avg_hits_per_9",
    "avg_hr_per_9",
    "avg_fip_component",
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


def innings_to_decimal(
    value: Any,
) -> float | None:
    """Convert MLB innings notation to decimal innings.

    Examples:
        5.0 -> 5.0
        5.1 -> 5.333...
        5.2 -> 5.666...
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


def innings_to_outs(
    value: Any,
) -> float | None:
    """Convert MLB innings notation to recorded outs."""
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

    return float(
        whole * 3
        + partial
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide safely and replace invalid values with missing."""
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


def build_pitcher_features(
    target_date: str | None = None,
) -> pd.DataFrame:
    """Build recent-form pitcher features for the requested slate."""
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

    logs_path = (
        GAME_LOGS_DIRECTORY
        / f"{target_date}.csv"
    )

    if not logs_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher game-log file: {logs_path}"
        )

    logs = pd.read_csv(
        logs_path
    )

    required_columns = {
        "pitcher_id",
        "pitcher_name",
        "team",
        "innings",
        "strikeouts",
        "walks",
        "hits",
        "earned_runs",
        "home_runs",
        "era",
    }

    missing_columns = (
        required_columns
        - set(logs.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{logs_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if logs.empty:
        empty_features = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

        empty_features.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print(
            f"No pitcher game logs found for {target_date}."
        )

        return empty_features

    logs["pitcher_id"] = pd.to_numeric(
        logs["pitcher_id"],
        errors="coerce",
    )

    logs = logs.dropna(
        subset=[
            "pitcher_id",
            "pitcher_name",
        ]
    ).copy()

    logs["pitcher_id"] = (
        logs["pitcher_id"]
        .astype(int)
    )

    logs["innings_decimal"] = logs[
        "innings"
    ].apply(
        innings_to_decimal
    )

    if "outs_recorded" in logs.columns:
        logs["outs_recorded"] = pd.to_numeric(
            logs["outs_recorded"],
            errors="coerce",
        )
    else:
        logs["outs_recorded"] = logs[
            "innings"
        ].apply(
            innings_to_outs
        )

    numeric_columns = [
        "innings_decimal",
        "outs_recorded",
        "strikeouts",
        "walks",
        "hits",
        "earned_runs",
        "home_runs",
        "era",
        "whip",
        "batters_faced",
        "strikeout_rate_bf",
        "walk_rate_bf",
        "k_minus_bb_rate",
        "strikeouts_per_9",
        "walks_per_9",
        "hits_per_9",
        "home_runs_per_9",
        "fip_component",
    ]

    for column in numeric_columns:
        if column not in logs.columns:
            logs[column] = np.nan

        logs[column] = pd.to_numeric(
            logs[column],
            errors="coerce",
        )

    # Backfill rates when an older game-log file lacks the newer columns.
    logs["whip"] = logs["whip"].fillna(
        safe_divide(
            logs["walks"] + logs["hits"],
            logs["innings_decimal"],
        )
    )

    logs["strikeout_rate_bf"] = (
        logs["strikeout_rate_bf"].fillna(
            safe_divide(
                logs["strikeouts"],
                logs["batters_faced"],
            )
        )
    )

    logs["walk_rate_bf"] = (
        logs["walk_rate_bf"].fillna(
            safe_divide(
                logs["walks"],
                logs["batters_faced"],
            )
        )
    )

    logs["k_minus_bb_rate"] = (
        logs["k_minus_bb_rate"].fillna(
            logs["strikeout_rate_bf"]
            - logs["walk_rate_bf"]
        )
    )

    logs["strikeouts_per_9"] = (
        logs["strikeouts_per_9"].fillna(
            safe_divide(
                logs["strikeouts"] * 9.0,
                logs["innings_decimal"],
            )
        )
    )

    logs["walks_per_9"] = (
        logs["walks_per_9"].fillna(
            safe_divide(
                logs["walks"] * 9.0,
                logs["innings_decimal"],
            )
        )
    )

    logs["hits_per_9"] = (
        logs["hits_per_9"].fillna(
            safe_divide(
                logs["hits"] * 9.0,
                logs["innings_decimal"],
            )
        )
    )

    logs["home_runs_per_9"] = (
        logs["home_runs_per_9"].fillna(
            safe_divide(
                logs["home_runs"] * 9.0,
                logs["innings_decimal"],
            )
        )
    )

    logs["fip_component"] = (
        logs["fip_component"].fillna(
            safe_divide(
                (
                    13.0 * logs["home_runs"]
                    + 3.0 * logs["walks"]
                    - 2.0 * logs["strikeouts"]
                ),
                logs["innings_decimal"],
            )
        )
    )

    if "game_date" in logs.columns:
        logs["game_date"] = pd.to_datetime(
            logs["game_date"],
            errors="coerce",
        )

        cutoff = pd.to_datetime(
            target_date
        )

        logs = logs[
            logs["game_date"] < cutoff
        ].copy()

        logs = logs.sort_values(
            [
                "pitcher_id",
                "game_date",
            ],
            ascending=[
                True,
                False,
            ],
        )

        logs = (
            logs.groupby(
                "pitcher_id",
                group_keys=False,
            )
            .head(
                RECENT_GAME_LIMIT
            )
            .copy()
        )

    features = (
        logs.groupby(
            [
                "pitcher_id",
                "pitcher_name",
                "team",
            ],
            dropna=False,
        )
        .agg(
            recent_games=(
                "pitcher_id",
                "size",
            ),
            avg_ip=(
                "innings_decimal",
                "mean",
            ),
            avg_outs=(
                "outs_recorded",
                "mean",
            ),
            avg_k=(
                "strikeouts",
                "mean",
            ),
            avg_bb=(
                "walks",
                "mean",
            ),
            avg_hits=(
                "hits",
                "mean",
            ),
            avg_er=(
                "earned_runs",
                "mean",
            ),
            avg_hr=(
                "home_runs",
                "mean",
            ),
            avg_era=(
                "era",
                "mean",
            ),
            avg_whip=(
                "whip",
                "mean",
            ),
            avg_batters_faced=(
                "batters_faced",
                "mean",
            ),
            avg_k_rate=(
                "strikeout_rate_bf",
                "mean",
            ),
            avg_bb_rate=(
                "walk_rate_bf",
                "mean",
            ),
            avg_k_minus_bb_rate=(
                "k_minus_bb_rate",
                "mean",
            ),
            avg_k_per_9=(
                "strikeouts_per_9",
                "mean",
            ),
            avg_bb_per_9=(
                "walks_per_9",
                "mean",
            ),
            avg_hits_per_9=(
                "hits_per_9",
                "mean",
            ),
            avg_hr_per_9=(
                "home_runs_per_9",
                "mean",
            ),
            avg_fip_component=(
                "fip_component",
                "mean",
            ),
        )
        .reset_index()
    )

    features.insert(
        0,
        "date",
        target_date,
    )

    numeric_feature_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column
        not in {
            "date",
            "pitcher_name",
            "team",
        }
    ]

    for column in numeric_feature_columns:
        if column not in features.columns:
            features[column] = np.nan

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        ).round(4)

    features = features[
        OUTPUT_COLUMNS
    ].copy()

    features = features.sort_values(
        [
            "pitcher_name",
        ],
        ascending=True,
    ).reset_index(drop=True)

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.csv"
    )

    features.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )

    print("=" * 72)
    print("PITCHER FEATURE BUILD COMPLETE")
    print("=" * 72)
    print(f"Slate date: {target_date}")
    print(
        f"Saved {len(features)} pitcher feature rows "
        f"to {OUTPUT_PATH}"
    )

    preview_columns = [
        "pitcher_name",
        "team",
        "recent_games",
        "avg_ip",
        "avg_outs",
        "avg_k",
        "avg_bb",
        "avg_era",
        "avg_whip",
        "avg_k_rate",
        "avg_bb_rate",
        "avg_k_per_9",
        "avg_fip_component",
    ]

    print(
        features[
            preview_columns
        ]
        .head(30)
        .to_string(index=False)
    )

    return features


if __name__ == "__main__":
    build_pitcher_features()
