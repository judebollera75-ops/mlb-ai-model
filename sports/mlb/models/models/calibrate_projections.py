"""Calibrate current MLB pitcher strikeout projections.

Input:
    outputs/pitcher_strikeout_projections.csv

Optional chronological holdout inputs:
    outputs/pitcher_strikeout_test_results.csv
    outputs/strikeout_test_results.csv
    outputs/pitchers/strikeout_test_results.csv
    outputs/pitchers/pitcher_strikeout_test_results.csv

Output:
    outputs/calibrated_strikeout_projections.csv

True calibration requires out-of-sample historical predictions and actual
results. When those are unavailable, this script preserves the raw projection,
applies only realistic safety bounds, and labels the result as uncalibrated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "pitcher_strikeout_projections.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "calibrated_strikeout_projections.csv"
)

HOLDOUT_PATH_CANDIDATES = [
    PROJECT_ROOT
    / "outputs"
    / "pitcher_strikeout_test_results.csv",
    PROJECT_ROOT
    / "outputs"
    / "strikeout_test_results.csv",
    PROJECT_ROOT
    / "outputs"
    / "pitchers"
    / "strikeout_test_results.csv",
    PROJECT_ROOT
    / "outputs"
    / "pitchers"
    / "pitcher_strikeout_test_results.csv",
]

MINIMUM_CALIBRATION_ROWS = 100
MINIMUM_PROJECTED_STRIKEOUTS = 0.0
MAXIMUM_PROJECTED_STRIKEOUTS = 15.0

OUTPUT_COLUMNS = [
    "pitcher_name",
    "team",
    "opponent",
    "game_id",
    "raw_projected_ks",
    "calibrated_projected_ks",
    "projected_ks_lower_80",
    "projected_ks_upper_80",
    "projected_ks_residual_std",
    "calibration_method",
    "calibration_status",
    "calibration_sample_size",
    "validation_mae",
    "validation_rmse",
    "calibration_intercept",
    "calibration_slope",
]


def first_existing_path(
    paths: list[Path],
) -> Path | None:
    """Return the first existing path."""
    for path in paths:
        if path.exists():
            return path

    return None


def choose_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first matching column from a list."""
    return next(
        (
            column
            for column in candidates
            if column in frame.columns
        ),
        None,
    )


def load_current_projections() -> pd.DataFrame:
    """Load and validate current pitcher strikeout projections."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Pitcher strikeout projections were not found: {INPUT_PATH}"
        )

    try:
        projections = pd.read_csv(INPUT_PATH)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError(
            f"Could not read pitcher projections: {INPUT_PATH}"
        ) from exc

    if projections.empty:
        raise ValueError(
            f"Pitcher projection file is empty: {INPUT_PATH}"
        )

    pitcher_column = choose_column(
        projections,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
        ],
    )

    projection_column = choose_column(
        projections,
        [
            "projected_ks",
            "projected_strikeouts",
            "projection",
        ],
    )

    if pitcher_column is None:
        raise ValueError(
            "Pitcher projection file does not contain a recognized "
            "pitcher-name column."
        )

    if projection_column is None:
        raise ValueError(
            "Pitcher projection file does not contain a recognized "
            "strikeout projection column."
        )

    projections = projections.rename(
        columns={
            pitcher_column: "pitcher_name",
            projection_column: "raw_projected_ks",
        }
    )

    projections["pitcher_name"] = (
        projections["pitcher_name"]
        .astype("string")
        .str.strip()
    )

    projections["raw_projected_ks"] = pd.to_numeric(
        projections["raw_projected_ks"],
        errors="coerce",
    )

    projections = projections.dropna(
        subset=[
            "pitcher_name",
            "raw_projected_ks",
        ]
    ).copy()

    projections = projections.loc[
        projections["pitcher_name"].ne("")
        & projections["raw_projected_ks"].ge(0)
    ].copy()

    duplicate_columns = [
        column
        for column in [
            "game_id",
            "pitcher_name",
        ]
        if column in projections.columns
    ]

    if duplicate_columns:
        projections = projections.drop_duplicates(
            subset=duplicate_columns,
            keep="last",
        )

    if projections.empty:
        raise ValueError(
            "No valid pitcher strikeout projections remained after cleaning."
        )

    return projections.reset_index(drop=True)


def load_holdout_results() -> pd.DataFrame:
    """Load chronological holdout predictions and actual strikeouts."""
    holdout_path = first_existing_path(
        HOLDOUT_PATH_CANDIDATES
    )

    if holdout_path is None:
        print(
            "No pitcher strikeout holdout file was found. "
            "True calibration will be skipped."
        )

        return pd.DataFrame()

    try:
        holdout = pd.read_csv(holdout_path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        print(
            f"Could not read holdout file {holdout_path}: {exc}"
        )

        return pd.DataFrame()

    prediction_column = choose_column(
        holdout,
        [
            "prediction",
            "projected_ks",
            "projected_strikeouts",
            "raw_prediction",
            "y_pred",
        ],
    )

    actual_column = choose_column(
        holdout,
        [
            "actual",
            "actual_strikeouts",
            "strikeouts",
            "target_strikeouts",
            "target_ks",
            "y_true",
        ],
    )

    if prediction_column is None or actual_column is None:
        print(
            f"Holdout schema was not recognized in {holdout_path}. "
            "True calibration will be skipped."
        )

        return pd.DataFrame()

    holdout = holdout.rename(
        columns={
            prediction_column: "prediction",
            actual_column: "actual",
        }
    )

    holdout["prediction"] = pd.to_numeric(
        holdout["prediction"],
        errors="coerce",
    )

    holdout["actual"] = pd.to_numeric(
        holdout["actual"],
        errors="coerce",
    )

    holdout = holdout.dropna(
        subset=[
            "prediction",
            "actual",
        ]
    ).copy()

    holdout = holdout.loc[
        np.isfinite(holdout["prediction"])
        & np.isfinite(holdout["actual"])
        & holdout["prediction"].ge(0)
        & holdout["actual"].ge(0)
    ].copy()

    print(
        f"Loaded {len(holdout):,} pitcher calibration rows "
        f"from {holdout_path}."
    )

    return holdout.reset_index(drop=True)


def fit_linear_calibration(
    holdout: pd.DataFrame,
) -> dict[str, Any] | None:
    """Fit an out-of-sample linear projection correction."""
    if len(holdout) < MINIMUM_CALIBRATION_ROWS:
        print(
            "Not enough pitcher holdout rows for calibration. "
            f"Found {len(holdout):,}; "
            f"need at least {MINIMUM_CALIBRATION_ROWS:,}."
        )

        return None

    x = holdout[
        ["prediction"]
    ].to_numpy(dtype=float)

    y = holdout[
        "actual"
    ].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    calibrated_holdout = model.predict(x)

    calibrated_holdout = np.clip(
        calibrated_holdout,
        a_min=MINIMUM_PROJECTED_STRIKEOUTS,
        a_max=MAXIMUM_PROJECTED_STRIKEOUTS,
    )

    residuals = y - calibrated_holdout

    residuals = residuals[
        np.isfinite(residuals)
    ]

    if len(residuals) < MINIMUM_CALIBRATION_ROWS:
        return None

    mae = float(
        mean_absolute_error(
            y,
            calibrated_holdout,
        )
    )

    rmse = float(
        mean_squared_error(
            y,
            calibrated_holdout,
        )
        ** 0.5
    )

    return {
        "model": model,
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "residuals": residuals,
        "residual_std": float(
            np.std(
                residuals,
                ddof=0,
            )
        ),
        "lower_residual": float(
            np.quantile(
                residuals,
                0.10,
            )
        ),
        "upper_residual": float(
            np.quantile(
                residuals,
                0.90,
            )
        ),
        "sample_size": int(len(residuals)),
        "mae": mae,
        "rmse": rmse,
    }


def apply_true_calibration(
    projections: pd.DataFrame,
    calibration: dict[str, Any],
) -> pd.DataFrame:
    """Apply fitted calibration and uncertainty intervals."""
    result = projections.copy()

    raw_values = result[
        "raw_projected_ks"
    ].to_numpy(dtype=float)

    calibrated = calibration[
        "model"
    ].predict(
        raw_values.reshape(-1, 1)
    )

    calibrated = np.clip(
        calibrated,
        a_min=MINIMUM_PROJECTED_STRIKEOUTS,
        a_max=MAXIMUM_PROJECTED_STRIKEOUTS,
    )

    result["calibrated_projected_ks"] = calibrated

    result["projected_ks_lower_80"] = np.clip(
        calibrated
        + calibration["lower_residual"],
        a_min=MINIMUM_PROJECTED_STRIKEOUTS,
        a_max=None,
    )

    result["projected_ks_upper_80"] = np.clip(
        calibrated
        + calibration["upper_residual"],
        a_min=MINIMUM_PROJECTED_STRIKEOUTS,
        a_max=None,
    )

    result["projected_ks_residual_std"] = calibration[
        "residual_std"
    ]

    result["calibration_method"] = (
        "chronological_holdout_linear"
    )

    result["calibration_status"] = "CALIBRATED"

    result["calibration_sample_size"] = calibration[
        "sample_size"
    ]

    result["validation_mae"] = calibration["mae"]
    result["validation_rmse"] = calibration["rmse"]

    result["calibration_intercept"] = calibration[
        "intercept"
    ]

    result["calibration_slope"] = calibration[
        "slope"
    ]

    return result


def apply_safe_fallback(
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Apply range protection without pretending it is calibration."""
    result = projections.copy()

    result["calibrated_projected_ks"] = result[
        "raw_projected_ks"
    ].clip(
        lower=MINIMUM_PROJECTED_STRIKEOUTS,
        upper=MAXIMUM_PROJECTED_STRIKEOUTS,
    )

    result["projected_ks_lower_80"] = np.nan
    result["projected_ks_upper_80"] = np.nan
    result["projected_ks_residual_std"] = np.nan

    result["calibration_method"] = "range_clip_only"
    result["calibration_status"] = "UNCALIBRATED"
    result["calibration_sample_size"] = 0

    result["validation_mae"] = np.nan
    result["validation_rmse"] = np.nan

    result["calibration_intercept"] = 0.0
    result["calibration_slope"] = 1.0

    return result


def select_output_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Keep stable output columns plus useful source metadata."""
    preferred_columns = [
        "pitcher_name",
        "team",
        "opponent",
        "game_id",
        "date",
        "raw_projected_ks",
        "calibrated_projected_ks",
        "projected_ks_lower_80",
        "projected_ks_upper_80",
        "projected_ks_residual_std",
        "calibration_method",
        "calibration_status",
        "calibration_sample_size",
        "validation_mae",
        "validation_rmse",
        "calibration_intercept",
        "calibration_slope",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in frame.columns
    ]

    extra_columns = [
        column
        for column in frame.columns
        if column not in columns
    ]

    return columns + extra_columns


def calibrate_projections() -> pd.DataFrame:
    """Create safe current pitcher strikeout projections."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    projections = load_current_projections()
    holdout = load_holdout_results()

    calibration = fit_linear_calibration(
        holdout
    )

    if calibration is None:
        calibrated = apply_safe_fallback(
            projections
        )
    else:
        calibrated = apply_true_calibration(
            projections,
            calibration,
        )

    calibrated["raw_projected_ks"] = pd.to_numeric(
        calibrated["raw_projected_ks"],
        errors="coerce",
    ).round(3)

    for column in [
        "calibrated_projected_ks",
        "projected_ks_lower_80",
        "projected_ks_upper_80",
        "projected_ks_residual_std",
        "validation_mae",
        "validation_rmse",
        "calibration_intercept",
        "calibration_slope",
    ]:
        calibrated[column] = pd.to_numeric(
            calibrated[column],
            errors="coerce",
        ).round(4)

    calibrated = calibrated.sort_values(
        "calibrated_projected_ks",
        ascending=False,
    ).reset_index(drop=True)

    calibrated = calibrated[
        select_output_columns(
            calibrated
        )
    ]

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.csv"
    )

    calibrated.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )

    print("=" * 72)
    print("PITCHER STRIKEOUT CALIBRATION COMPLETE")
    print("=" * 72)
    print(
        f"Pitchers calibrated: {len(calibrated):,}"
    )
    print(
        "Calibration status: "
        f"{calibrated['calibration_status'].iloc[0]}"
    )
    print(
        "Calibration method: "
        f"{calibrated['calibration_method'].iloc[0]}"
    )
    print(
        "Calibration samples: "
        f"{int(calibrated['calibration_sample_size'].iloc[0])}"
    )
    print(f"Saved to: {OUTPUT_PATH}")

    preview_columns = [
        column
        for column in [
            "pitcher_name",
            "team",
            "opponent",
            "raw_projected_ks",
            "calibrated_projected_ks",
            "projected_ks_lower_80",
            "projected_ks_upper_80",
            "calibration_status",
        ]
        if column in calibrated.columns
    ]

    print()
    print(
        calibrated[
            preview_columns
        ]
        .head(30)
        .to_string(index=False)
    )

    return calibrated


if __name__ == "__main__":
    calibrate_projections()
