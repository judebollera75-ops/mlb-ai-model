"""Generate current-slate MLB hitter projections.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD
    MLB_HITTER_LOG_PATH=/optional/custom/path.csv

Inputs:
    data/hitters/<target-date>.csv
    data/hitter_game_logs/<season>.csv
    models/hitters/<market>_model.pkl

Output:
    outputs/hitters/today_hitter_projections.csv

The live feature matrix is created with the same feature-engineering functions
used by build_hitter_training_dataset.py. This prevents training/inference
feature drift.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from sports.mlb.features.features.build_hitter_training_dataset import (
    MINIMUM_PRIOR_GAMES,
    RAW_STAT_COLUMNS,
    add_consistency_features,
    add_derived_targets,
    add_optional_context_features,
    add_opportunity_features,
    add_previous_game_features,
    add_rate_features,
    add_rolling_features,
    add_schedule_features,
    add_season_to_date_features,
    add_trend_features,
    clean_numeric_columns,
    normalize_text_columns,
)


MODEL_DIRECTORY = PROJECT_ROOT / "models" / "hitters"

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "hitters"
OUTPUT_PATH = OUTPUT_DIRECTORY / "today_hitter_projections.csv"
LIVE_FEATURES_PATH = OUTPUT_DIRECTORY / "today_hitter_live_features.csv"

MODEL_MARKETS = {
    "hits": "projected_hits",
    "total_bases": "projected_total_bases",
    "runs": "projected_runs",
    "rbi": "projected_rbi",
    "hits_runs_rbis": "projected_hits_runs_rbis",
    "fantasy_score": "projected_fantasy_score",
}

IDENTITY_COLUMNS = [
    "date",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "side",
    "batting_order",
    "position",
    "home_team",
    "away_team",
    "venue",
]


def get_target_date() -> date:
    """Return the requested MLB slate date."""
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


def get_paths(
    target_date: date,
) -> tuple[Path, Path]:
    """Return current hitter and historical log paths."""
    hitters_path = (
        PROJECT_ROOT
        / "data"
        / "hitters"
        / f"{target_date.isoformat()}.csv"
    )

    custom_logs_path = os.getenv("MLB_HITTER_LOG_PATH")

    if custom_logs_path:
        logs_path = Path(
            custom_logs_path
        ).expanduser().resolve()
    else:
        logs_path = (
            PROJECT_ROOT
            / "data"
            / "hitter_game_logs"
            / f"{target_date.year}.csv"
        )

    return hitters_path, logs_path


def load_model_bundle(
    market: str,
) -> dict[str, Any]:
    """Load and validate one production model bundle."""
    model_path = (
        MODEL_DIRECTORY
        / f"{market}_model.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing hitter model: {model_path}. "
            "Run train_hitter_models.py first."
        )

    bundle = joblib.load(model_path)

    required_keys = {
        "model",
        "features",
        "medians",
        "market",
    }

    missing_keys = required_keys - set(bundle)

    if missing_keys:
        raise ValueError(
            f"{model_path} is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    if bundle["market"] != market:
        raise ValueError(
            f"Model bundle market mismatch for {model_path}. "
            f"Expected {market!r}, found {bundle['market']!r}."
        )

    if not isinstance(bundle["features"], list):
        raise TypeError(
            f"{model_path} contains an invalid feature list."
        )

    if not isinstance(bundle["medians"], dict):
        raise TypeError(
            f"{model_path} contains invalid feature medians."
        )

    return bundle


def load_current_hitters(
    hitters_path: Path,
    target_date: date,
) -> pd.DataFrame:
    """Load and validate current-slate hitters."""
    if not hitters_path.exists():
        raise FileNotFoundError(
            "Current hitter file was not found: "
            f"{hitters_path}"
        )

    try:
        hitters = pd.read_csv(hitters_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read current hitter file: {hitters_path}"
        ) from exc

    required_columns = {
        "game_id",
        "player_id",
        "player_name",
    }

    missing_columns = required_columns - set(hitters.columns)

    if missing_columns:
        raise ValueError(
            "Current hitter file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    hitters["player_id"] = pd.to_numeric(
        hitters["player_id"],
        errors="coerce",
    )

    hitters = hitters.dropna(
        subset=[
            "game_id",
            "player_id",
            "player_name",
        ]
    ).copy()

    hitters["player_id"] = hitters[
        "player_id"
    ].astype("int64")

    hitters["date"] = pd.Timestamp(target_date)

    if "batting_order" in hitters.columns:
        hitters["batting_order"] = pd.to_numeric(
            hitters["batting_order"],
            errors="coerce",
        )

    hitters = normalize_text_columns(hitters)

    hitters = hitters.drop_duplicates(
        subset=["game_id", "player_id"],
        keep="last",
    ).reset_index(drop=True)

    if hitters.empty:
        raise ValueError(
            f"No valid hitters were found in {hitters_path}."
        )

    return hitters


def load_historical_logs(
    logs_path: Path,
    target_date: date,
) -> pd.DataFrame:
    """Load only games completed before the target slate."""
    if not logs_path.exists():
        raise FileNotFoundError(
            f"Hitter game-log file was not found: {logs_path}"
        )

    try:
        logs = pd.read_csv(logs_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read hitter game logs: {logs_path}"
        ) from exc

    required_columns = {
        "date",
        "game_id",
        "player_id",
    }

    missing_columns = required_columns - set(logs.columns)

    if missing_columns:
        raise ValueError(
            "Hitter game logs are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    logs["date"] = pd.to_datetime(
        logs["date"],
        errors="coerce",
    )

    logs["player_id"] = pd.to_numeric(
        logs["player_id"],
        errors="coerce",
    )

    logs = logs.dropna(
        subset=[
            "date",
            "game_id",
            "player_id",
        ]
    ).copy()

    logs["player_id"] = logs[
        "player_id"
    ].astype("int64")

    target_timestamp = pd.Timestamp(target_date)

    logs = logs.loc[
        logs["date"] < target_timestamp
    ].copy()

    logs = normalize_text_columns(logs)

    logs = clean_numeric_columns(
        logs,
        RAW_STAT_COLUMNS,
    )

    logs = logs.sort_values(
        ["player_id", "date", "game_id"]
    ).reset_index(drop=True)

    if logs.empty:
        raise ValueError(
            "No historical hitter games were available before "
            f"{target_date.isoformat()}."
        )

    return logs


def build_projection_rows(
    hitters: pd.DataFrame,
    logs: pd.DataFrame,
) -> pd.DataFrame:
    """Create placeholder rows for today's games.

    All current-game box-score fields are explicitly set to zero so no
    accidental current-game statistics can leak into the projection.
    """
    projection_rows = hitters.copy()
    historical_logs = logs.copy()

    for column in historical_logs.columns:
        if column not in projection_rows.columns:
            projection_rows[column] = pd.NA

    for column in projection_rows.columns:
        if column not in historical_logs.columns:
            historical_logs[column] = pd.NA

    for column in RAW_STAT_COLUMNS:
        projection_rows[column] = 0.0

    projection_rows["__is_projection_row"] = 1
    historical_logs["__is_projection_row"] = 0

    combined_columns = list(
        dict.fromkeys(
            list(historical_logs.columns)
            + list(projection_rows.columns)
        )
    )

    historical_logs = historical_logs.reindex(
        columns=combined_columns
    )

    projection_rows = projection_rows.reindex(
        columns=combined_columns
    )

    combined = pd.concat(
        [historical_logs, projection_rows],
        ignore_index=True,
        sort=False,
    )

    combined = combined.sort_values(
        [
            "player_id",
            "date",
            "game_id",
            "__is_projection_row",
        ]
    ).reset_index(drop=True)

    return combined


def engineer_live_features(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the exact training feature-engineering pipeline."""
    featured = clean_numeric_columns(
        combined,
        RAW_STAT_COLUMNS,
    )

    featured = add_derived_targets(featured)
    featured = add_schedule_features(featured)
    featured = add_previous_game_features(featured)
    featured = add_rolling_features(featured)
    featured = add_season_to_date_features(featured)
    featured = add_rate_features(featured)
    featured = add_trend_features(featured)
    featured = add_consistency_features(featured)
    featured = add_opportunity_features(featured)
    featured = add_optional_context_features(featured)

    numeric_columns = featured.select_dtypes(
        include=[np.number]
    ).columns

    featured[numeric_columns] = featured[
        numeric_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    today_features = featured.loc[
        featured["__is_projection_row"].eq(1)
    ].copy()

    today_features = today_features.loc[
        today_features["prior_games"] >= MINIMUM_PRIOR_GAMES
    ].copy()

    today_features = today_features.drop_duplicates(
        subset=["game_id", "player_id"],
        keep="last",
    ).reset_index(drop=True)

    return today_features


def prepare_model_matrix(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    market: str,
) -> pd.DataFrame:
    """Build a finite feature matrix in saved training order."""
    feature_columns = bundle["features"]
    medians = bundle["medians"]

    missing_columns = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing_columns:
        missing_rate = (
            len(missing_columns)
            / max(len(feature_columns), 1)
        )

        print(
            f"WARNING: {market} is missing "
            f"{len(missing_columns)} of "
            f"{len(feature_columns)} model features."
        )

        print(
            "First missing features: "
            f"{missing_columns[:15]}"
        )

        if missing_rate > 0.10:
            raise ValueError(
                f"{market} live inference is missing "
                f"{missing_rate:.1%} of trained features. "
                "Training and inference feature engineering "
                "are no longer aligned."
            )

    matrix = pd.DataFrame(
        index=frame.index,
    )

    for feature in feature_columns:
        if feature in frame.columns:
            values = pd.to_numeric(
                frame[feature],
                errors="coerce",
            )
        else:
            values = pd.Series(
                np.nan,
                index=frame.index,
                dtype=float,
            )

        median_value = medians.get(feature, 0.0)

        try:
            median_value = float(median_value)
        except (TypeError, ValueError):
            median_value = 0.0

        if not np.isfinite(median_value):
            median_value = 0.0

        matrix[feature] = values.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(median_value)

    return matrix[feature_columns].astype(float)


def residual_interval(
    predictions: np.ndarray,
    residuals: list[Any],
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create empirical prediction ranges from holdout residuals."""
    clean_residuals = pd.to_numeric(
        pd.Series(
            residuals,
            dtype="object",
        ),
        errors="coerce",
    ).dropna()

    clean_residuals = clean_residuals.loc[
        np.isfinite(clean_residuals)
    ]

    if len(clean_residuals) < 50:
        return (
            np.full(
                len(predictions),
                np.nan,
            ),
            np.full(
                len(predictions),
                np.nan,
            ),
            float("nan"),
        )

    lower_residual = float(
        clean_residuals.quantile(
            lower_quantile
        )
    )

    upper_residual = float(
        clean_residuals.quantile(
            upper_quantile
        )
    )

    residual_standard_deviation = float(
        clean_residuals.std(
            ddof=0
        )
    )

    lower_bound = np.clip(
        predictions + lower_residual,
        a_min=0.0,
        a_max=None,
    )

    upper_bound = np.clip(
        predictions + upper_residual,
        a_min=0.0,
        a_max=None,
    )

    return (
        lower_bound,
        upper_bound,
        residual_standard_deviation,
    )


def add_market_prediction(
    frame: pd.DataFrame,
    market: str,
    output_column: str,
) -> pd.DataFrame:
    """Apply one market model and add uncertainty metadata."""
    bundle = load_model_bundle(market)

    matrix = prepare_model_matrix(
        frame,
        bundle,
        market,
    )

    predictions = np.asarray(
        bundle["model"].predict(matrix),
        dtype=float,
    )

    predictions = np.clip(
        predictions,
        a_min=0.0,
        a_max=None,
    )

    lower_bound, upper_bound, residual_std = residual_interval(
        predictions,
        bundle.get(
            "holdout_residuals",
            [],
        ),
    )

    projected = frame.copy()

    projected[output_column] = predictions

    projected[
        f"{output_column}_lower_80"
    ] = lower_bound

    projected[
        f"{output_column}_upper_80"
    ] = upper_bound

    projected[
        f"{output_column}_residual_std"
    ] = residual_std

    projected[
        f"{market}_model_name"
    ] = bundle.get(
        "model_name",
        type(bundle["model"]).__name__,
    )

    projected[
        f"{market}_validation_mae"
    ] = bundle.get(
        "validation_mae",
        np.nan,
    )

    return projected


def select_output_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return stable public output columns."""
    columns: list[str] = []

    for column in IDENTITY_COLUMNS:
        if column in frame.columns:
            columns.append(column)

    for column in [
        "prior_games",
        "days_rest",
        "is_back_to_back",
        "is_short_rest",
        "expected_plate_appearances",
        "expected_at_bats",
    ]:
        if column in frame.columns:
            columns.append(column)

    for market, projection_column in MODEL_MARKETS.items():
        columns.extend(
            [
                projection_column,
                f"{projection_column}_lower_80",
                f"{projection_column}_upper_80",
                f"{projection_column}_residual_std",
                f"{market}_model_name",
                f"{market}_validation_mae",
            ]
        )

    return [
        column
        for column in columns
        if column in frame.columns
    ]


def project_today_hitters() -> pd.DataFrame:
    """Generate and save all current hitter-market projections."""
    target_date = get_target_date()

    hitters_path, logs_path = get_paths(
        target_date
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Generating current MLB hitter projections")
    print(f"Slate date: {target_date.isoformat()}")
    print(f"Current hitters: {hitters_path}")
    print(f"Historical logs: {logs_path}")
    print("=" * 72)

    hitters = load_current_hitters(
        hitters_path,
        target_date,
    )

    logs = load_historical_logs(
        logs_path,
        target_date,
    )

    combined = build_projection_rows(
        hitters,
        logs,
    )

    today = engineer_live_features(
        combined
    )

    projected_player_ids = set(
        today["player_id"].astype(int)
    )

    requested_player_ids = set(
        hitters["player_id"].astype(int)
    )

    skipped_player_ids = (
        requested_player_ids
        - projected_player_ids
    )

    if skipped_player_ids:
        skipped = hitters.loc[
            hitters["player_id"].isin(
                skipped_player_ids
            ),
            [
                "player_id",
                "player_name",
            ],
        ].drop_duplicates()

        print(
            "\nSkipped hitters with fewer than "
            f"{MINIMUM_PRIOR_GAMES} prior games:"
        )

        print(
            skipped.to_string(
                index=False
            )
        )

    if today.empty:
        raise RuntimeError(
            "No current hitters had enough history to project."
        )

    for market, output_column in MODEL_MARKETS.items():
        print(f"\nProjecting market: {market}")

        today = add_market_prediction(
            today,
            market,
            output_column,
        )

    projection_columns = list(
        MODEL_MARKETS.values()
    )

    for column in projection_columns:
        today[column] = pd.to_numeric(
            today[column],
            errors="coerce",
        ).round(3)

        for suffix in [
            "_lower_80",
            "_upper_80",
            "_residual_std",
        ]:
            interval_column = (
                f"{column}{suffix}"
            )

            if interval_column in today.columns:
                today[
                    interval_column
                ] = pd.to_numeric(
                    today[
                        interval_column
                    ],
                    errors="coerce",
                ).round(3)

    today["date"] = target_date.isoformat()

    live_feature_columns = sorted(
        {
            feature
            for market in MODEL_MARKETS
            for feature in load_model_bundle(
                market
            )["features"]
            if feature in today.columns
        }
    )

    live_feature_output_columns = [
        column
        for column in [
            "date",
            "game_id",
            "player_id",
            "player_name",
            "team",
            "opponent",
        ]
        if column in today.columns
    ] + live_feature_columns

    today[
        live_feature_output_columns
    ].to_csv(
        LIVE_FEATURES_PATH,
        index=False,
    )

    output_columns = select_output_columns(
        today
    )

    output = today[
        output_columns
    ].copy()

    sort_columns = [
        column
        for column in [
            "projected_fantasy_score",
            "projected_hits_runs_rbis",
            "projected_total_bases",
            "projected_hits",
        ]
        if column in output.columns
    ]

    if sort_columns:
        output = output.sort_values(
            sort_columns,
            ascending=False,
        )

    output = output.reset_index(
        drop=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nHitter projections completed successfully."
    )

    print(
        f"Projected hitters: {len(output):,}"
    )

    print(
        f"Saved projections: {OUTPUT_PATH}"
    )

    print(
        f"Saved live features: {LIVE_FEATURES_PATH}"
    )

    preview_columns = [
        column
        for column in [
            "player_name",
            "team",
            "opponent",
            "batting_order",
            "expected_plate_appearances",
            "projected_hits",
            "projected_total_bases",
            "projected_runs",
            "projected_rbi",
            "projected_hits_runs_rbis",
            "projected_fantasy_score",
        ]
        if column in output.columns
    ]

    if preview_columns:
        print("\nTop projected hitters:")

        print(
            output[
                preview_columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    return output


if __name__ == "__main__":
    project_today_hitters()
