"""Generate current-slate MLB pitcher outs projections.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Inputs:
    data/pitchers/<target-date>.csv
    data/game_logs/<target-date>.csv

Output:
    outputs/pitcher_outs_projections.csv

This remains a weighted historical projection, not a trained calibrated model.
Uncertainty fields describe historical variation and are clearly labeled so
the probability engine does not mistake them for leakage-safe holdout errors.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

PITCHERS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pitchers"
)

GAME_LOGS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "game_logs"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "pitcher_outs_projections.csv"
)

MINIMUM_HISTORY_GAMES = 2
MINIMUM_PROJECTED_OUTS = 3.0
MAXIMUM_PROJECTED_OUTS = 27.0

RECENT_WINDOWS = (3, 5, 10)

OUTPUT_COLUMNS = [
    "date",
    "game_id",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "side",
    "status",
    "history_games",
    "days_rest",
    "last3_avg_outs",
    "last5_avg_outs",
    "last10_avg_outs",
    "season_avg_outs",
    "season_median_outs",
    "season_std_outs",
    "recent_std_outs",
    "projected_outs",
    "projected_outs_lower_80",
    "projected_outs_upper_80",
    "projected_outs_residual_std",
    "projection_confidence",
    "projection_method",
    "calibration_status",
    "uncertainty_method",
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


def innings_to_outs(value: Any) -> float:
    """Convert MLB innings notation into recorded outs.

    Examples:
        5.0 -> 15
        5.1 -> 16
        5.2 -> 17
        6.0 -> 18
    """
    if value is None or pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if not text:
        return float("nan")

    if "." in text:
        whole_text, fractional_text = text.split(".", 1)
    else:
        whole_text, fractional_text = text, "0"

    try:
        whole_innings = int(whole_text)
        fractional_outs = int(
            fractional_text[:1] or "0"
        )
    except (TypeError, ValueError):
        return float("nan")

    if whole_innings < 0:
        return float("nan")

    if fractional_outs not in {0, 1, 2}:
        return float("nan")

    return float(
        whole_innings * 3
        + fractional_outs
    )


def get_input_paths(
    target_date: date,
) -> tuple[Path, Path]:
    """Return exact current-slate input paths."""
    date_string = target_date.isoformat()

    pitchers_path = (
        PITCHERS_DIRECTORY
        / f"{date_string}.csv"
    )

    logs_path = (
        GAME_LOGS_DIRECTORY
        / f"{date_string}.csv"
    )

    if not pitchers_path.exists():
        raise FileNotFoundError(
            "Current pitcher file was not found: "
            f"{pitchers_path}"
        )

    if not logs_path.exists():
        raise FileNotFoundError(
            "Matching pitcher game-log file was not found: "
            f"{logs_path}"
        )

    return pitchers_path, logs_path


def read_csv_safely(
    path: Path,
    label: str,
) -> pd.DataFrame:
    """Read a required CSV with a useful error."""
    try:
        frame = pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError(
            f"Could not read {label}: {path}"
        ) from exc

    if frame.empty:
        raise ValueError(
            f"{label} is empty: {path}"
        )

    return frame


def require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    label: str,
) -> None:
    """Validate required source columns."""
    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{label} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def clean_text(value: Any) -> Any:
    """Strip text while preserving missing values."""
    if value is None or pd.isna(value):
        return pd.NA

    cleaned = str(value).strip()

    return cleaned if cleaned else pd.NA


def load_and_clean_data(
    target_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate current pitchers and historical logs."""
    pitchers_path, logs_path = get_input_paths(
        target_date
    )

    pitchers = read_csv_safely(
        pitchers_path,
        "current pitcher file",
    )

    logs = read_csv_safely(
        logs_path,
        "pitcher game-log file",
    )

    print(f"Using pitcher file: {pitchers_path}")
    print(f"Using game-log file: {logs_path}")
    print(f"Pitcher rows loaded: {len(pitchers):,}")
    print(f"Game-log rows loaded: {len(logs):,}")

    require_columns(
        pitchers,
        {
            "pitcher_id",
            "pitcher_name",
            "team",
        },
        "Current pitcher file",
    )

    require_columns(
        logs,
        {
            "pitcher_id",
            "pitcher_name",
            "game_date",
            "innings",
        },
        "Pitcher game-log file",
    )

    pitchers = pitchers.copy()
    logs = logs.copy()

    pitchers["pitcher_id"] = pd.to_numeric(
        pitchers["pitcher_id"],
        errors="coerce",
    )

    logs["pitcher_id"] = pd.to_numeric(
        logs["pitcher_id"],
        errors="coerce",
    )

    logs["game_date"] = pd.to_datetime(
        logs["game_date"],
        errors="coerce",
    )

    logs["game_outs"] = logs[
        "innings"
    ].apply(innings_to_outs)

    pitchers = pitchers.dropna(
        subset=[
            "pitcher_id",
            "pitcher_name",
            "team",
        ]
    ).copy()

    logs = logs.dropna(
        subset=[
            "pitcher_id",
            "game_date",
            "game_outs",
        ]
    ).copy()

    pitchers["pitcher_id"] = (
        pitchers["pitcher_id"]
        .astype("int64")
    )

    logs["pitcher_id"] = (
        logs["pitcher_id"]
        .astype("int64")
    )

    logs["game_outs"] = pd.to_numeric(
        logs["game_outs"],
        errors="coerce",
    )

    target_timestamp = pd.Timestamp(
        target_date
    )

    # Prevent current-slate or future data leakage.
    logs = logs.loc[
        logs["game_date"]
        < target_timestamp
    ].copy()

    logs = logs.loc[
        logs["game_outs"].between(
            0,
            27,
            inclusive="both",
        )
    ].copy()

    for column in [
        "pitcher_name",
        "team",
        "opponent",
        "side",
        "status",
    ]:
        if column in pitchers.columns:
            pitchers[column] = pitchers[
                column
            ].apply(clean_text)

    pitchers = pitchers.drop_duplicates(
        subset=[
            "game_id",
            "pitcher_id",
        ]
        if "game_id" in pitchers.columns
        else ["pitcher_id"],
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

    if logs.empty:
        raise ValueError(
            "No pitcher history existed before the target slate."
        )

    return (
        pitchers.reset_index(drop=True),
        logs,
    )


def calculate_days_rest(
    pitcher_logs: pd.DataFrame,
    target_date: date,
) -> float:
    """Calculate days since the pitcher's most recent appearance."""
    if pitcher_logs.empty:
        return float("nan")

    most_recent_date = pitcher_logs[
        "game_date"
    ].max()

    if pd.isna(most_recent_date):
        return float("nan")

    rest_days = (
        pd.Timestamp(target_date)
        - most_recent_date
    ).days

    return float(
        np.clip(
            rest_days,
            0,
            30,
        )
    )


def weighted_outs_projection(
    outs: pd.Series,
) -> float:
    """Blend short-, medium-, and season-level workload."""
    last3_average = float(
        outs.head(3).mean()
    )

    last5_average = float(
        outs.head(5).mean()
    )

    last10_average = float(
        outs.head(10).mean()
    )

    season_average = float(
        outs.mean()
    )

    history_games = len(outs)

    if history_games >= 10:
        projection = (
            0.30 * last3_average
            + 0.30 * last5_average
            + 0.20 * last10_average
            + 0.20 * season_average
        )
    elif history_games >= 5:
        projection = (
            0.40 * last3_average
            + 0.35 * last5_average
            + 0.25 * season_average
        )
    else:
        projection = (
            0.60 * last3_average
            + 0.40 * season_average
        )

    return float(
        np.clip(
            projection,
            MINIMUM_PROJECTED_OUTS,
            MAXIMUM_PROJECTED_OUTS,
        )
    )


def empirical_uncertainty(
    outs: pd.Series,
    projection: float,
) -> tuple[float, float, float]:
    """Estimate a descriptive 80% range from prior game outcomes.

    This is historical dispersion, not model calibration.
    """
    clean_outs = pd.to_numeric(
        outs,
        errors="coerce",
    ).dropna()

    clean_outs = clean_outs.loc[
        np.isfinite(clean_outs)
    ]

    if len(clean_outs) < 3:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    historical_residuals = (
        clean_outs.to_numpy(dtype=float)
        - float(clean_outs.mean())
    )

    lower_residual = float(
        np.quantile(
            historical_residuals,
            0.10,
        )
    )

    upper_residual = float(
        np.quantile(
            historical_residuals,
            0.90,
        )
    )

    standard_deviation = float(
        clean_outs.std(ddof=0)
    )

    lower_bound = float(
        np.clip(
            projection + lower_residual,
            0.0,
            27.0,
        )
    )

    upper_bound = float(
        np.clip(
            projection + upper_residual,
            0.0,
            27.0,
        )
    )

    return (
        lower_bound,
        upper_bound,
        standard_deviation,
    )


def projection_confidence(
    history_games: int,
) -> str:
    """Assign a descriptive history-depth label."""
    if history_games >= 10:
        return "HIGH"

    if history_games >= 5:
        return "MEDIUM"

    return "LOW"


def calculate_pitcher_summaries(
    logs: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    """Build one projection summary per pitcher."""
    summaries: list[dict[str, Any]] = []

    for pitcher_id, pitcher_logs in logs.groupby(
        "pitcher_id",
        sort=False,
    ):
        pitcher_logs = pitcher_logs.sort_values(
            "game_date",
            ascending=False,
        )

        outs = pd.to_numeric(
            pitcher_logs["game_outs"],
            errors="coerce",
        ).dropna()

        outs = outs.loc[
            np.isfinite(outs)
        ]

        history_games = len(outs)

        if history_games < MINIMUM_HISTORY_GAMES:
            continue

        projected_outs = weighted_outs_projection(
            outs
        )

        (
            lower_bound,
            upper_bound,
            historical_std,
        ) = empirical_uncertainty(
            outs,
            projected_outs,
        )

        recent_outs = outs.head(
            min(10, history_games)
        )

        summaries.append(
            {
                "pitcher_id": int(pitcher_id),
                "history_games": history_games,
                "days_rest": calculate_days_rest(
                    pitcher_logs,
                    target_date,
                ),
                "last3_avg_outs": float(
                    outs.head(3).mean()
                ),
                "last5_avg_outs": float(
                    outs.head(5).mean()
                ),
                "last10_avg_outs": float(
                    outs.head(10).mean()
                ),
                "season_avg_outs": float(
                    outs.mean()
                ),
                "season_median_outs": float(
                    outs.median()
                ),
                "season_std_outs": float(
                    outs.std(ddof=0)
                ),
                "recent_std_outs": float(
                    recent_outs.std(ddof=0)
                ),
                "projected_outs": projected_outs,
                "projected_outs_lower_80": lower_bound,
                "projected_outs_upper_80": upper_bound,
                "projected_outs_residual_std": historical_std,
                "projection_confidence": (
                    projection_confidence(
                        history_games
                    )
                ),
                "projection_method": (
                    "weighted_historical_outs"
                ),
                "calibration_status": "UNCALIBRATED",
                "uncertainty_method": (
                    "historical_game_dispersion"
                ),
            }
        )

    return pd.DataFrame(summaries)


def build_output(
    pitchers: pd.DataFrame,
    summaries: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    """Merge current pitchers with historical projections."""
    projections = pitchers.merge(
        summaries,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )

    unmatched = projections.loc[
        projections["history_games"].isna()
    ].copy()

    if not unmatched.empty:
        print(
            "\nPitchers skipped because they lacked sufficient history:"
        )

        display_columns = [
            column
            for column in [
                "pitcher_id",
                "pitcher_name",
                "team",
                "opponent",
            ]
            if column in unmatched.columns
        ]

        print(
            unmatched[
                display_columns
            ].to_string(index=False)
        )

    projections = projections.loc[
        projections["history_games"].notna()
    ].copy()

    if projections.empty:
        raise RuntimeError(
            "No current pitchers had enough history for outs projections."
        )

    projections["date"] = (
        target_date.isoformat()
    )

    for column in OUTPUT_COLUMNS:
        if column not in projections.columns:
            projections[column] = pd.NA

    numeric_columns = [
        "history_games",
        "days_rest",
        "last3_avg_outs",
        "last5_avg_outs",
        "last10_avg_outs",
        "season_avg_outs",
        "season_median_outs",
        "season_std_outs",
        "recent_std_outs",
        "projected_outs",
        "projected_outs_lower_80",
        "projected_outs_upper_80",
        "projected_outs_residual_std",
    ]

    for column in numeric_columns:
        projections[column] = pd.to_numeric(
            projections[column],
            errors="coerce",
        ).round(3)

    projections = projections.sort_values(
        [
            "projected_outs",
            "pitcher_name",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return projections[
        OUTPUT_COLUMNS
    ].copy()


def save_output(
    projections: pd.DataFrame,
) -> None:
    """Save output atomically so stale partial files are impossible."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.csv"
    )

    projections.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )


def project_pitcher_outs() -> pd.DataFrame:
    """Generate and save current pitcher outs projections."""
    target_date = get_target_date()

    print("=" * 72)
    print("GENERATING PITCHER OUTS PROJECTIONS")
    print(f"Slate date: {target_date.isoformat()}")
    print("=" * 72)

    pitchers, logs = load_and_clean_data(
        target_date
    )

    summaries = calculate_pitcher_summaries(
        logs,
        target_date,
    )

    print(
        "Pitchers with sufficient historical data: "
        f"{len(summaries):,}"
    )

    projections = build_output(
        pitchers,
        summaries,
        target_date,
    )

    save_output(projections)

    print(
        f"\nSaved {len(projections):,} pitcher-outs projections "
        f"to {OUTPUT_PATH}"
    )

    print(
        "\nImportant: pitcher outs remain uncalibrated. "
        "The uncertainty field reflects historical variation, "
        "not chronological model residuals."
    )

    preview_columns = [
        "pitcher_name",
        "team",
        "opponent",
        "history_games",
        "days_rest",
        "last3_avg_outs",
        "last5_avg_outs",
        "season_avg_outs",
        "projected_outs",
        "projected_outs_lower_80",
        "projected_outs_upper_80",
        "projection_confidence",
        "calibration_status",
    ]

    print()
    print(
        projections[
            preview_columns
        ]
        .head(40)
        .to_string(index=False)
    )

    return projections


if __name__ == "__main__":
    project_pitcher_outs()
