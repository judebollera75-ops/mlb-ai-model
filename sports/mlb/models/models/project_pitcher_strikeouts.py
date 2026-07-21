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


# Conservative production controls.
MIN_MODEL_FEATURE_COVERAGE = 0.55
FULL_MODEL_FEATURE_COVERAGE = 0.85
MIN_GAMES_FOR_FULL_TRUST = 10
MIN_GAMES_FOR_MODEL_USE = 3

MODEL_WEIGHT_LOW = 0.35
MODEL_WEIGHT_MEDIUM = 0.60
MODEL_WEIGHT_HIGH = 0.78

MIN_PROJECTED_KS = 1.5
MAX_PROJECTED_KS = 11.0
MAX_DEVIATION_FROM_BASELINE = 2.25


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


def calculate_feature_coverage(
    observed_features: int,
    total_features: int,
) -> float:
    """Return the fraction of trained features available for this row."""
    return float(
        np.clip(
            observed_features / max(total_features, 1),
            0.0,
            1.0,
        )
    )


def calculate_reliability_score(
    games_started: float,
    feature_coverage: float,
    used_model: bool,
) -> float:
    """Return a 0-100 data reliability score, not a win probability."""
    if not used_model:
        return 20.0

    games_value = 0.0 if pd.isna(games_started) else float(games_started)
    experience_score = min(games_value / MIN_GAMES_FOR_FULL_TRUST, 1.0)

    score = (
        0.65 * feature_coverage
        + 0.35 * experience_score
    ) * 100.0

    return float(np.clip(score, 0.0, 100.0))


def confidence_label(
    reliability_score: float,
    used_model: bool,
) -> str:
    """Map the numerical reliability score to an audit-friendly label."""
    if not used_model or reliability_score < 50:
        return "LOW"

    if reliability_score >= 80:
        return "HIGH"

    return "MEDIUM"


def model_blend_weight(
    games_started: float,
    feature_coverage: float,
    used_model: bool,
) -> float:
    """Choose how much of the raw model projection to trust."""
    if not used_model:
        return 0.0

    games_value = 0.0 if pd.isna(games_started) else float(games_started)

    if (
        games_value >= MIN_GAMES_FOR_FULL_TRUST
        and feature_coverage >= FULL_MODEL_FEATURE_COVERAGE
    ):
        return MODEL_WEIGHT_HIGH

    if (
        games_value >= 5
        and feature_coverage >= MIN_MODEL_FEATURE_COVERAGE
    ):
        return MODEL_WEIGHT_MEDIUM

    return MODEL_WEIGHT_LOW


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

    fallback = fallback_projection(master).to_numpy(dtype="float64")

    feature_coverage = (
        observed_counts.astype(float)
        / max(len(feature_columns), 1)
    ).clip(lower=0.0, upper=1.0)

    games_started = pd.to_numeric(
        master["games_started"],
        errors="coerce",
    ).fillna(0.0)

    raw_model_valid = np.isfinite(predictions) & (predictions >= 0)

    sufficient_inputs = (
        feature_coverage.ge(MIN_MODEL_FEATURE_COVERAGE)
        & games_started.ge(MIN_GAMES_FOR_MODEL_USE)
    )

    use_model = raw_model_valid & sufficient_inputs.to_numpy(dtype=bool)

    blend_weights = np.array(
        [
            model_blend_weight(
                games_started=float(row_games),
                feature_coverage=float(row_coverage),
                used_model=bool(row_used_model),
            )
            for row_games, row_coverage, row_used_model in zip(
                games_started,
                feature_coverage,
                use_model,
            )
        ],
        dtype="float64",
    )

    conservative_model_projection = np.clip(
        predictions,
        fallback - MAX_DEVIATION_FROM_BASELINE,
        fallback + MAX_DEVIATION_FROM_BASELINE,
    )

    blended_projection = (
        blend_weights * conservative_model_projection
        + (1.0 - blend_weights) * fallback
    )

    final_projection = np.where(
        use_model,
        blended_projection,
        fallback,
    )

    final_projection = np.clip(
        final_projection,
        MIN_PROJECTED_KS,
        MAX_PROJECTED_KS,
    )

    master["raw_model_projected_ks"] = predictions
    master["baseline_projected_ks"] = fallback
    master["model_blend_weight"] = blend_weights
    master["feature_coverage"] = feature_coverage
    master["projected_ks"] = final_projection

    master["projection_source"] = np.select(
        [
            use_model & (blend_weights >= MODEL_WEIGHT_HIGH),
            use_model,
        ],
        [
            "BLENDED_MODEL_HIGH_TRUST",
            "BLENDED_MODEL_CONSERVATIVE",
        ],
        default="RECENT_SEASON_FALLBACK",
    )

    master["projection_reliability_score"] = [
        calculate_reliability_score(
            games_started=float(row_games),
            feature_coverage=float(row_coverage),
            used_model=bool(row_used_model),
        )
        for row_games, row_coverage, row_used_model in zip(
            games_started,
            feature_coverage,
            use_model,
        )
    ]

    master["projection_confidence"] = [
        confidence_label(
            reliability_score=float(row_score),
            used_model=bool(row_used_model),
        )
        for row_score, row_used_model in zip(
            master["projection_reliability_score"],
            use_model,
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
        "raw_model_projected_ks",
        "baseline_projected_ks",
        "model_blend_weight",
        "feature_coverage",
        "projected_ks",
        "projection_source",
        "projection_reliability_score",
        "projection_confidence",
        "model_name",
        "model_feature_count",
        "observed_model_features",
    ]

    audit_columns = [
        "opponent",
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
        "days_rest",
        "park_factor",
    ]

    for column in audit_columns:
        if column in master.columns and column not in output_columns:
            output_columns.append(column)

    output = master[output_columns].copy()
    for column in [
        "raw_model_projected_ks",
        "baseline_projected_ks",
        "model_blend_weight",
        "feature_coverage",
        "projected_ks",
        "projection_reliability_score",
    ]:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column],
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
                "raw_model_projected_ks",
                "baseline_projected_ks",
                "projected_ks",
                "model_blend_weight",
                "feature_coverage",
                "projection_reliability_score",
                "projection_source",
                "projection_confidence",
                "observed_model_features",
            ]
        ].to_string(index=False)
    )

    return output


if __name__ == "__main__":
    project_strikeouts()
