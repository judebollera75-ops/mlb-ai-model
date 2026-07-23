"""Train a leakage-safe MLB pitcher strikeout model.

Inputs:
    data/training/strikeout_training_dataset.csv

Outputs:
    models/leakage_free_strikeout_model.pkl
    outputs/leakage_free_strikeout_test_results.csv

The saved model bundle contains:
    model
    features
    feature_medians
    holdout_residuals
    validation_mae
    residual_std
    residual_quantiles
    train_rows
    holdout_rows
    train_end_date
    holdout_start_date
    holdout_end_date

The holdout residuals are used by probability_engine_v2.py to estimate
pitcher-strikeout probabilities empirically instead of relying only on a
Poisson distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


DATA_PATH = Path("data/training/strikeout_training_dataset.csv")
MODEL_PATH = Path("models/leakage_free_strikeout_model.pkl")
RESULTS_PATH = Path("outputs/leakage_free_strikeout_test_results.csv")

HOLDOUT_FRACTION = 0.20
MINIMUM_TRAINING_ROWS = 300
MINIMUM_HOLDOUT_ROWS = 75
RANDOM_STATE = 42

CANDIDATE_FEATURES = [
    "season_k_per_start",
    "avg_k",
    "opp_k_per_game",
    "opp_runs_per_game",
    "opp_hits_per_game",
    "opp_walks_per_game",
    "opp_avg",
    "opp_obp",
    "opp_slg",
    "opp_ops",
    "last3_avg_ks",
    "last3_avg_ip",
    "last5_avg_ks",
    "last5_avg_ip",
    "last5_avg_hits",
    "last5_avg_walks",
    "last5_avg_er",
    "days_rest",
    "park_factor",
]


def read_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset was not found: {DATA_PATH}")

    try:
        frame = pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read training dataset: {DATA_PATH}") from exc

    required_columns = {"date", "actual_strikeouts"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            "Strikeout training dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["actual_strikeouts"] = pd.to_numeric(
        frame["actual_strikeouts"],
        errors="coerce",
    )

    feature_columns = [
        column
        for column in CANDIDATE_FEATURES
        if column in frame.columns
    ]
    if not feature_columns:
        raise ValueError("No valid pregame feature columns were found.")

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(
        subset=["date", "actual_strikeouts"]
    ).sort_values("date").reset_index(drop=True)

    if len(frame) < MINIMUM_TRAINING_ROWS + MINIMUM_HOLDOUT_ROWS:
        raise ValueError(
            "Not enough rows to train and validate the strikeout model. "
            f"Found {len(frame):,}; need at least "
            f"{MINIMUM_TRAINING_ROWS + MINIMUM_HOLDOUT_ROWS:,}."
        )

    frame.attrs["feature_columns"] = feature_columns
    return frame


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * (1.0 - HOLDOUT_FRACTION))
    split_index = min(
        max(split_index, MINIMUM_TRAINING_ROWS),
        len(frame) - MINIMUM_HOLDOUT_ROWS,
    )

    train = frame.iloc[:split_index].copy()
    holdout = frame.iloc[split_index:].copy()

    if train.empty or holdout.empty:
        raise RuntimeError("Chronological train/holdout split produced an empty set.")

    if train["date"].max() >= holdout["date"].min():
        holdout_start_date = holdout["date"].min()
        train = frame.loc[frame["date"] < holdout_start_date].copy()
        holdout = frame.loc[frame["date"] >= holdout_start_date].copy()

    if len(train) < MINIMUM_TRAINING_ROWS or len(holdout) < MINIMUM_HOLDOUT_ROWS:
        raise ValueError(
            "Chronological split does not leave enough rows after keeping "
            "whole dates together."
        )

    return train, holdout


def fit_feature_medians(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    medians = train[feature_columns].median(numeric_only=True)
    medians = medians.reindex(feature_columns)

    global_fallback = float(
        pd.concat(
            [train[column] for column in feature_columns],
            ignore_index=True,
        ).median()
    )
    if not np.isfinite(global_fallback):
        global_fallback = 0.0

    return medians.fillna(global_fallback).astype(float)


def prepare_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    medians: pd.Series,
) -> pd.DataFrame:
    matrix = frame[feature_columns].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    for column in feature_columns:
        matrix[column] = pd.to_numeric(
            matrix[column],
            errors="coerce",
        ).fillna(float(medians[column]))

    return matrix[feature_columns]


def create_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=550,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=8,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.10,
        reg_lambda=1.50,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def residual_summary(residuals: np.ndarray) -> dict[str, float]:
    clean = np.asarray(residuals, dtype=float)
    clean = clean[np.isfinite(clean)]

    if clean.size == 0:
        return {
            "q05": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "q95": np.nan,
        }

    return {
        "q05": float(np.quantile(clean, 0.05)),
        "q10": float(np.quantile(clean, 0.10)),
        "q25": float(np.quantile(clean, 0.25)),
        "q50": float(np.quantile(clean, 0.50)),
        "q75": float(np.quantile(clean, 0.75)),
        "q90": float(np.quantile(clean, 0.90)),
        "q95": float(np.quantile(clean, 0.95)),
    }


def train_model() -> dict[str, Any]:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame = read_training_data()
    feature_columns = list(frame.attrs["feature_columns"])
    train, holdout = chronological_split(frame)

    feature_medians = fit_feature_medians(train, feature_columns)

    x_train = prepare_matrix(train, feature_columns, feature_medians)
    y_train = train["actual_strikeouts"].astype(float)

    x_holdout = prepare_matrix(holdout, feature_columns, feature_medians)
    y_holdout = holdout["actual_strikeouts"].astype(float)

    validation_model = create_model()
    validation_model.fit(
        x_train,
        y_train,
        eval_set=[(x_holdout, y_holdout)],
        verbose=False,
    )

    holdout_predictions = np.asarray(
        validation_model.predict(x_holdout),
        dtype=float,
    )
    holdout_residuals = (
        y_holdout.to_numpy(dtype=float)
        - holdout_predictions
    )

    validation_mae = float(
        mean_absolute_error(
            y_holdout,
            holdout_predictions,
        )
    )
    residual_std = float(
        np.std(
            holdout_residuals,
            ddof=0,
        )
    )

    all_medians = fit_feature_medians(frame, feature_columns)
    x_all = prepare_matrix(frame, feature_columns, all_medians)
    y_all = frame["actual_strikeouts"].astype(float)

    production_model = create_model()
    production_model.fit(
        x_all,
        y_all,
        verbose=False,
    )

    bundle = {
        "model": production_model,
        "features": feature_columns,
        "feature_medians": {
            column: float(all_medians[column])
            for column in feature_columns
        },
        "holdout_residuals": holdout_residuals.astype(float),
        "validation_mae": validation_mae,
        "residual_std": residual_std,
        "residual_quantiles": residual_summary(holdout_residuals),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "all_training_rows": int(len(frame)),
        "train_end_date": train["date"].max().date().isoformat(),
        "holdout_start_date": holdout["date"].min().date().isoformat(),
        "holdout_end_date": holdout["date"].max().date().isoformat(),
        "target_column": "actual_strikeouts",
        "model_version": "strikeout_residual_bundle_v2",
    }

    joblib.dump(bundle, MODEL_PATH)

    identity_columns = [
        column
        for column in [
            "date",
            "game_id",
            "pitcher_id",
            "pitcher_name",
            "team",
            "opponent",
        ]
        if column in holdout.columns
    ]

    results = holdout[identity_columns].copy()
    results["actual_strikeouts"] = y_holdout.to_numpy(dtype=float)
    results["predicted_strikeouts"] = holdout_predictions
    results["residual"] = holdout_residuals
    results["absolute_error"] = np.abs(holdout_residuals)
    results["squared_error"] = np.square(holdout_residuals)
    results.to_csv(RESULTS_PATH, index=False)

    print("=" * 72)
    print("LEAKAGE-SAFE STRIKEOUT MODEL V2 TRAINED")
    print("=" * 72)
    print(f"Features used ({len(feature_columns)}): {feature_columns}")
    print(f"Training rows: {len(train):,}")
    print(f"Holdout rows: {len(holdout):,}")
    print(
        "Holdout dates: "
        f"{holdout['date'].min().date()} to "
        f"{holdout['date'].max().date()}"
    )
    print(f"Holdout MAE: {validation_mae:.3f} strikeouts")
    print(f"Residual standard deviation: {residual_std:.3f}")
    print(f"Model bundle saved to: {MODEL_PATH}")
    print(f"Holdout results saved to: {RESULTS_PATH}")
    print()
    print(results.head(20).to_string(index=False))

    return bundle


if __name__ == "__main__":
    train_model()
