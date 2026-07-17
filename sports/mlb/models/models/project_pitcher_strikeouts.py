"""Generate current-slate pitcher strikeout projections with the trained model.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Inputs:
    data/final/master_dataset.csv
    models/leakage_free_strikeout_model.pkl
    data/training/strikeout_training_dataset.csv  (used only for feature medians)

Output:
    outputs/pitcher_strikeout_projections.csv
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MASTER_PATH = PROJECT_ROOT / "data" / "final" / "master_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "leakage_free_strikeout_model.pkl"
TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "strikeout_training_dataset.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "pitcher_strikeout_projections.csv"

REQUIRED_ID_COLUMNS = {
    "pitcher_name",
    "team",
    "game_id",
    "side",
    "games_started",
    "strikeouts",
}

FALLBACK_FEATURES = [
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


def get_target_date() -> str:
    """Return and validate the slate date."""
    raw_value = os.getenv("MLB_TARGET_DATE", date.today().isoformat())

    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc


def load_model_bundle() -> tuple[Any, list[str]]:
    """Load the serialized estimator and its exact training feature order."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing strikeout model: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        feature_columns = bundle.get("features")
    else:
        model = bundle
        feature_columns = getattr(model, "feature_names_in_", None)

    if model is None or not hasattr(model, "predict"):
        raise TypeError(
            f"{MODEL_PATH} does not contain a usable regression model."
        )

    if feature_columns is None:
        feature_columns = FALLBACK_FEATURES

    feature_columns = [str(column) for column in feature_columns]

    if not feature_columns:
        raise ValueError("The strikeout model has no feature columns.")

    return model, feature_columns


def training_feature_medians(feature_columns: list[str]) -> pd.Series:
    """Calculate stable imputation values from the training data only."""
    medians = pd.Series(0.0, index=feature_columns, dtype="float64")

    if not TRAINING_PATH.exists():
        return medians

    training = pd.read_csv(TRAINING_PATH, usecols=lambda c: c in feature_columns)

    for column in feature_columns:
        if column not in training.columns:
            continue

        values = pd.to_numeric(training[column], errors="coerce")
        median = values.median()

        if pd.notna(median):
            medians.loc[column] = float(median)

    return medians


def prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    """Validate identifiers and create safe season/recent fallback fields."""
    missing_columns = REQUIRED_ID_COLUMNS - set(master.columns)

    if missing_columns:
        raise KeyError(
            f"{MASTER_PATH} is missing columns: {sorted(missing_columns)}"
        )

    prepared = master.copy()

    numeric_columns = {
        "games_started",
        "strikeouts",
        "season_k_per_start",
        "avg_k",
    }

    for column in numeric_columns:
        if column not in prepared.columns:
            prepared[column] = np.nan

        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    calculated_season_rate = (
        prepared["strikeouts"]
        / prepared["games_started"].replace(0, np.nan)
    )

    prepared["season_k_per_start"] = prepared[
        "season_k_per_start"
    ].fillna(calculated_season_rate)

    return prepared


def build_model_matrix(
    master: pd.DataFrame,
    feature_columns: list[str],
    medians: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model matrix in the exact feature order used in training."""
    matrix = pd.DataFrame(index=master.index)
    observed_counts = pd.Series(0, index=master.index, dtype="int64")

    for column in feature_columns:
        if column in master.columns:
            values = pd.to_numeric(master[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=master.index, dtype="float64")

        observed_counts = observed_counts + values.notna().astype("int64")
        matrix[column] = values.fillna(float(medians.get(column, 0.0)))

    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    for column in feature_columns:
        matrix[column] = matrix[column].fillna(float(medians.get(column, 0.0)))

    return matrix[feature_columns], observed_counts


def fallback_projection(master: pd.DataFrame) -> pd.Series:
    """Return a conservative projection only for rows with no model inputs."""
    season = pd.to_numeric(master["season_k_per_start"], errors="coerce")
    recent = pd.to_numeric(master["avg_k"], errors="coerce")

    fallback = pd.Series(np.nan, index=master.index, dtype="float64")
    both = season.notna() & recent.notna()

    fallback.loc[both] = 0.70 * recent.loc[both] + 0.30 * season.loc[both]
    fallback = fallback.fillna(recent).fillna(season)

    median_value = fallback.median()
    if pd.isna(median_value):
        median_value = 5.0

    return fallback.fillna(float(median_value))


def confidence_label(
    games_started: float,
    observed_features: int,
    total_features: int,
    used_model: bool,
) -> str:
    """Assign a data-availability confidence label, not a win probability."""
    if not used_model:
        return "LOW"

    coverage = observed_features / max(total_features, 1)

    if pd.notna(games_started) and games_started >= 10 and coverage >= 0.80:
        return "HIGH"

    if pd.notna(games_started) and games_started >= 5 and coverage >= 0.55:
        return "MEDIUM"

    return "LOW"


def project_strikeouts(target_date: str | None = None) -> pd.DataFrame:
    """Create and save model-based strikeout projections for the slate."""
    if target_date is None:
        target_date = get_target_date()
    else:
        target_date = datetime.strptime(
            target_date,
            "%Y-%m-%d",
        ).date().isoformat()

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master dataset: {MASTER_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    master = prepare_master(pd.read_csv(MASTER_PATH))
    model, feature_columns = load_model_bundle()
    medians = training_feature_medians(feature_columns)
    model_matrix, observed_counts = build_model_matrix(
        master,
        feature_columns,
        medians,
    )

    predictions = np.asarray(model.predict(model_matrix), dtype="float64")

    if predictions.shape[0] != len(master):
        raise RuntimeError(
            "Strikeout model returned an unexpected number of predictions: "
            f"expected {len(master)}, received {predictions.shape[0]}."
        )

    valid_model_prediction = np.isfinite(predictions) & (predictions >= 0)
    fallback = fallback_projection(master).to_numpy(dtype="float64")

    master["projected_ks"] = np.where(
        valid_model_prediction,
        predictions,
        fallback,
    )

    master["projection_source"] = np.where(
        valid_model_prediction,
        "LEAKAGE_FREE_XGBOOST",
        "RECENT_SEASON_FALLBACK",
    )

    master["projection_confidence"] = [
        confidence_label(
            games_started=row_games,
            observed_features=int(row_observed),
            total_features=len(feature_columns),
            used_model=bool(row_used_model),
        )
        for row_games, row_observed, row_used_model in zip(
            master["games_started"],
            observed_counts,
            valid_model_prediction,
        )
    ]

    master["date"] = target_date
    master["model_name"] = MODEL_PATH.stem
    master["model_feature_count"] = len(feature_columns)
    master["observed_model_features"] = observed_counts

    output_columns = [
        "date",
        "game_id",
        "pitcher_name",
        "team",
        "side",
        "games_started",
        "strikeouts",
        "season_k_per_start",
        "avg_k",
        "projected_ks",
        "projection_source",
        "projection_confidence",
        "model_name",
        "model_feature_count",
        "observed_model_features",
    ]

    output = master[output_columns].copy()
    output["projected_ks"] = pd.to_numeric(
        output["projected_ks"],
        errors="coerce",
    ).round(3)

    output = (
        output.drop_duplicates(
            subset=["game_id", "pitcher_name"],
            keep="first",
        )
        .sort_values("projected_ks", ascending=False)
        .reset_index(drop=True)
    )

    if output["projected_ks"].isna().any():
        raise RuntimeError("One or more strikeout projections are missing.")

    output.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Saved {len(output)} model-based strikeout projections "
        f"to {OUTPUT_PATH}"
    )
    print(f"Model: {MODEL_PATH.name}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print()
    print(
        output[
            [
                "pitcher_name",
                "team",
                "season_k_per_start",
                "avg_k",
                "projected_ks",
                "projection_source",
                "projection_confidence",
                "observed_model_features",
            ]
        ].to_string(index=False)
    )

    return output


if __name__ == "__main__":
    project_strikeouts()
