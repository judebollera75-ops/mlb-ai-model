"""Generate leakage-safe MLB starting-pitcher outs projections.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Inputs:
    data/pitchers/<target-date>.csv
    data/game_logs/<target-date>.csv
    data/historical/pitcher_game_logs.csv

Output:
    outputs/pitcher_outs_projections.csv

The model is trained only on features that would have been known before each
historical start. A chronological holdout is used to estimate prediction error.
Current projections blend the trained model with a stable historical workload
estimate, with heavier shrinkage for pitchers who have limited history.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PITCHERS_DIRECTORY = PROJECT_ROOT / "data" / "pitchers"
GAME_LOGS_DIRECTORY = PROJECT_ROOT / "data" / "game_logs"
HISTORICAL_LOGS_PATH = PROJECT_ROOT / "data" / "historical" / "pitcher_game_logs.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "pitcher_outs_projections.csv"
CALIBRATION_PATH = PROJECT_ROOT / "models" / "pitcher_outs_calibration.pkl"

MINIMUM_HISTORY_GAMES = 3
MINIMUM_TRAINING_HISTORY_GAMES = 5
MINIMUM_TRAINING_ROWS = 400
MINIMUM_PROJECTED_OUTS = 3.0
MAXIMUM_PROJECTED_OUTS = 24.0
HOLDOUT_FRACTION = 0.20
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "history_games",
    "days_rest",
    "last1_outs",
    "last3_avg_outs",
    "last5_avg_outs",
    "last10_avg_outs",
    "season_avg_outs",
    "season_median_outs",
    "season_std_outs",
    "recent_std_outs",
    "outs_trend",
    "last3_avg_pitches",
    "last5_avg_pitches",
    "last3_avg_batters_faced",
    "last5_avg_batters_faced",
    "last3_avg_earned_runs",
    "last5_avg_earned_runs",
    "last3_avg_hits",
    "last5_avg_hits",
    "last3_avg_walks",
    "last5_avg_walks",
    "last3_avg_whip",
    "last5_avg_whip",
    "is_home",
]

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
    "baseline_projected_outs",
    "model_projected_outs",
    "projected_pitch_count",
    "projected_pitches_per_out",
    "model_blend_weight",
    "projection_reliability_score",
    "projection_confidence",
    "projection_method",
    "calibration_status",
    "uncertainty_method",
]


def get_target_date() -> date:
    raw_value = os.getenv("MLB_TARGET_DATE", date.today().isoformat())
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc


def innings_to_outs(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    if "." in text:
        whole_text, fraction_text = text.split(".", 1)
    else:
        whole_text, fraction_text = text, "0"

    try:
        whole_innings = int(whole_text)
        fractional_outs = int(fraction_text[:1] or "0")
    except (TypeError, ValueError):
        return float("nan")

    if whole_innings < 0 or fractional_outs not in {0, 1, 2}:
        return float("nan")

    return float(whole_innings * 3 + fractional_outs)


def read_required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} was not found: {path}")

    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {label}: {path}") from exc

    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")

    return frame


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def clean_text(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def prepare_logs(frame: pd.DataFrame, target_date: date) -> pd.DataFrame:
    logs = frame.copy()
    require_columns(
        logs,
        {"pitcher_id", "game_date"},
        "Pitcher game-log file",
    )

    logs["pitcher_id"] = pd.to_numeric(logs["pitcher_id"], errors="coerce")
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")

    if "outs_recorded" in logs.columns:
        logs["game_outs"] = pd.to_numeric(logs["outs_recorded"], errors="coerce")
    elif "innings" in logs.columns:
        logs["game_outs"] = logs["innings"].apply(innings_to_outs)
    else:
        raise ValueError("Pitcher game-log file needs outs_recorded or innings.")

    logs = logs.dropna(subset=["pitcher_id", "game_date", "game_outs"]).copy()
    logs["pitcher_id"] = logs["pitcher_id"].astype("int64")
    logs = logs.loc[logs["game_date"] < pd.Timestamp(target_date)].copy()
    logs = logs.loc[logs["game_outs"].between(0, 27, inclusive="both")].copy()

    numeric_defaults = {
        "pitches_thrown": np.nan,
        "batters_faced": np.nan,
        "earned_runs": np.nan,
        "hits": np.nan,
        "walks": np.nan,
        "whip": np.nan,
        "is_home": np.nan,
    }
    for column, default in numeric_defaults.items():
        if column not in logs.columns:
            logs[column] = default
        logs[column] = pd.to_numeric(logs[column], errors="coerce")

    return logs.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def load_current_data(target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_string = target_date.isoformat()
    pitchers_path = PITCHERS_DIRECTORY / f"{date_string}.csv"
    logs_path = GAME_LOGS_DIRECTORY / f"{date_string}.csv"

    pitchers = read_required_csv(pitchers_path, "current pitcher file")
    current_logs = read_required_csv(logs_path, "current pitcher game logs")

    require_columns(
        pitchers,
        {"pitcher_id", "pitcher_name", "team"},
        "Current pitcher file",
    )

    pitchers = pitchers.copy()
    pitchers["pitcher_id"] = pd.to_numeric(pitchers["pitcher_id"], errors="coerce")
    pitchers = pitchers.dropna(subset=["pitcher_id", "pitcher_name", "team"]).copy()
    pitchers["pitcher_id"] = pitchers["pitcher_id"].astype("int64")

    if "status" not in pitchers.columns and "game_status" in pitchers.columns:
        pitchers["status"] = pitchers["game_status"]

    for column in ["pitcher_name", "team", "opponent", "side", "status"]:
        if column in pitchers.columns:
            pitchers[column] = pitchers[column].apply(clean_text)

    dedupe_columns = ["pitcher_id"]
    if "game_id" in pitchers.columns:
        dedupe_columns.insert(0, "game_id")
    pitchers = pitchers.drop_duplicates(subset=dedupe_columns, keep="last")

    logs = prepare_logs(current_logs, target_date)

    print(f"Using pitcher file: {pitchers_path}")
    print(f"Using current game-log file: {logs_path}")
    print(f"Current pitcher rows loaded: {len(pitchers):,}")
    print(f"Current historical log rows loaded: {len(logs):,}")

    return pitchers.reset_index(drop=True), logs


def safe_mean(frame: pd.DataFrame, column: str, window: int) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").tail(window)
    return float(values.mean()) if values.notna().any() else float("nan")


def create_feature_row(
    history: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    is_home: Any = np.nan,
) -> dict[str, float]:
    history = history.sort_values("game_date")
    outs = pd.to_numeric(history["game_outs"], errors="coerce").dropna()

    if outs.empty:
        return {column: float("nan") for column in FEATURE_COLUMNS}

    last_date = history["game_date"].max()
    days_rest = float((target_timestamp - last_date).days) if pd.notna(last_date) else np.nan
    days_rest = float(np.clip(days_rest, 0, 30)) if np.isfinite(days_rest) else np.nan

    season_average = float(outs.mean())
    last3_average = float(outs.tail(3).mean())
    recent_outs = outs.tail(min(10, len(outs)))

    row = {
        "history_games": float(len(outs)),
        "days_rest": days_rest,
        "last1_outs": float(outs.iloc[-1]),
        "last3_avg_outs": last3_average,
        "last5_avg_outs": float(outs.tail(5).mean()),
        "last10_avg_outs": float(outs.tail(10).mean()),
        "season_avg_outs": season_average,
        "season_median_outs": float(outs.median()),
        "season_std_outs": float(outs.std(ddof=0)),
        "recent_std_outs": float(recent_outs.std(ddof=0)),
        "outs_trend": last3_average - season_average,
        "last3_avg_pitches": safe_mean(history, "pitches_thrown", 3),
        "last5_avg_pitches": safe_mean(history, "pitches_thrown", 5),
        "last3_avg_batters_faced": safe_mean(history, "batters_faced", 3),
        "last5_avg_batters_faced": safe_mean(history, "batters_faced", 5),
        "last3_avg_earned_runs": safe_mean(history, "earned_runs", 3),
        "last5_avg_earned_runs": safe_mean(history, "earned_runs", 5),
        "last3_avg_hits": safe_mean(history, "hits", 3),
        "last5_avg_hits": safe_mean(history, "hits", 5),
        "last3_avg_walks": safe_mean(history, "walks", 3),
        "last5_avg_walks": safe_mean(history, "walks", 5),
        "last3_avg_whip": safe_mean(history, "whip", 3),
        "last5_avg_whip": safe_mean(history, "whip", 5),
        "is_home": pd.to_numeric(pd.Series([is_home]), errors="coerce").iloc[0],
    }
    return row


def build_training_dataset(logs: pd.DataFrame) -> pd.DataFrame:
    """Create chronological lag features without row-by-row lookback loops."""
    frame = logs.sort_values(["pitcher_id", "game_date"]).copy()
    group = frame.groupby("pitcher_id", sort=False, group_keys=False)

    frame["history_games"] = group.cumcount().astype(float)
    frame["days_rest"] = group["game_date"].diff().dt.days.clip(0, 30)

    shifted_outs = group["game_outs"].shift(1)
    frame["last1_outs"] = shifted_outs
    frame["last3_avg_outs"] = shifted_outs.groupby(frame["pitcher_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    frame["last5_avg_outs"] = shifted_outs.groupby(frame["pitcher_id"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    frame["last10_avg_outs"] = shifted_outs.groupby(frame["pitcher_id"]).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    frame["season_avg_outs"] = shifted_outs.groupby(frame["pitcher_id"]).expanding().mean().reset_index(level=0, drop=True)
    frame["season_median_outs"] = shifted_outs.groupby(frame["pitcher_id"]).expanding().median().reset_index(level=0, drop=True)
    frame["season_std_outs"] = shifted_outs.groupby(frame["pitcher_id"]).expanding().std(ddof=0).reset_index(level=0, drop=True)
    frame["recent_std_outs"] = shifted_outs.groupby(frame["pitcher_id"]).rolling(10, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
    frame["outs_trend"] = frame["last3_avg_outs"] - frame["season_avg_outs"]

    rolling_specs = {
        "pitches_thrown": ("last3_avg_pitches", "last5_avg_pitches"),
        "batters_faced": ("last3_avg_batters_faced", "last5_avg_batters_faced"),
        "earned_runs": ("last3_avg_earned_runs", "last5_avg_earned_runs"),
        "hits": ("last3_avg_hits", "last5_avg_hits"),
        "walks": ("last3_avg_walks", "last5_avg_walks"),
        "whip": ("last3_avg_whip", "last5_avg_whip"),
    }
    for source, (name3, name5) in rolling_specs.items():
        shifted = group[source].shift(1)
        frame[name3] = shifted.groupby(frame["pitcher_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        frame[name5] = shifted.groupby(frame["pitcher_id"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    frame["is_home"] = pd.to_numeric(frame["is_home"], errors="coerce")
    frame["target_outs"] = frame["game_outs"].astype(float)

    training = frame.loc[frame["history_games"] >= MINIMUM_TRAINING_HISTORY_GAMES].copy()
    return training[["pitcher_id", "game_date", "target_outs", *FEATURE_COLUMNS]].sort_values("game_date").reset_index(drop=True)



def fit_leakage_safe_model(
    training: pd.DataFrame,
) -> tuple[dict[str, Any] | None, float, np.ndarray, int]:
    """Fit a chronological two-model ensemble and retain empirical residuals."""
    if len(training) < MINIMUM_TRAINING_ROWS:
        print(
            "Not enough chronological training rows for the outs model; "
            "using the workload-based historical fallback."
        )
        return None, float("nan"), np.array([], dtype=float), 0

    split_index = int(len(training) * (1.0 - HOLDOUT_FRACTION))
    split_index = min(max(split_index, 1), len(training) - 1)

    train = training.iloc[:split_index].copy()
    holdout = training.iloc[split_index:].copy()

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURE_COLUMNS])
    x_holdout = imputer.transform(holdout[FEATURE_COLUMNS])

    rf = RandomForestRegressor(
        n_estimators=350,
        max_depth=9,
        min_samples_leaf=10,
        max_features=0.70,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    et = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=8,
        max_features=0.80,
        random_state=RANDOM_STATE + 1,
        n_jobs=-1,
    )

    rf.fit(x_train, train["target_outs"])
    et.fit(x_train, train["target_outs"])

    holdout_predictions = (
        0.55 * rf.predict(x_holdout)
        + 0.45 * et.predict(x_holdout)
    )
    residuals = (
        holdout["target_outs"].to_numpy(dtype=float)
        - holdout_predictions
    )
    holdout_mae = float(
        mean_absolute_error(
            holdout["target_outs"],
            holdout_predictions,
        )
    )

    x_all = imputer.fit_transform(training[FEATURE_COLUMNS])
    rf.fit(x_all, training["target_outs"])
    et.fit(x_all, training["target_outs"])

    bundle = {
        "imputer": imputer,
        "rf": rf,
        "et": et,
    }

    print(f"Chronological training rows: {len(train):,}")
    print(f"Chronological holdout rows: {len(holdout):,}")
    print(f"Ensemble holdout MAE: {holdout_mae:.3f} outs")
    print(
        "Empirical 80% residual interval: "
        f"[{np.quantile(residuals, 0.10):.3f}, "
        f"{np.quantile(residuals, 0.90):.3f}]"
    )

    return bundle, holdout_mae, residuals, len(holdout)


def stable_historical_projection(features: dict[str, float]) -> float:
    """Estimate expected outs from workload, efficiency, and recent results."""
    history_games = int(features["history_games"])

    recent_result_projection = np.nanmean(
        [
            features["last3_avg_outs"],
            features["last5_avg_outs"],
            features["last10_avg_outs"],
            features["season_avg_outs"],
            features["season_median_outs"],
        ]
    )

    projected_pitch_count = np.nanmean(
        [
            features["last3_avg_pitches"],
            features["last5_avg_pitches"],
        ]
    )

    recent_outs = np.nanmean(
        [
            features["last3_avg_outs"],
            features["last5_avg_outs"],
        ]
    )

    pitches_per_out = (
        projected_pitch_count / recent_outs
        if np.isfinite(projected_pitch_count)
        and np.isfinite(recent_outs)
        and recent_outs > 0
        else np.nan
    )

    workload_projection = (
        projected_pitch_count / pitches_per_out
        if np.isfinite(projected_pitch_count)
        and np.isfinite(pitches_per_out)
        and pitches_per_out > 0
        else np.nan
    )

    if np.isfinite(workload_projection):
        projection = (
            0.55 * recent_result_projection
            + 0.45 * workload_projection
        )
    else:
        projection = recent_result_projection

    # Reduce trust for pitchers with very little history.
    if history_games < 5:
        projection = 0.75 * projection + 0.25 * 15.0

    return float(
        np.clip(
            projection,
            MINIMUM_PROJECTED_OUTS,
            MAXIMUM_PROJECTED_OUTS,
        )
    )


def model_blend_weight(history_games: int, disagreement: float = 0.0) -> float:
    """Trust the model less when history is limited or models disagree."""
    if history_games >= 15:
        base = 0.78
    elif history_games >= 10:
        base = 0.70
    elif history_games >= 6:
        base = 0.58
    else:
        base = 0.40

    disagreement_penalty = min(max(disagreement, 0.0) / 4.0, 0.25)
    return float(np.clip(base - disagreement_penalty, 0.30, 0.78))


def confidence_label(reliability_score: float, calibrated: bool) -> str:
    if not calibrated or reliability_score < 55:
        return "LOW"
    if reliability_score >= 78:
        return "HIGH"
    return "MEDIUM"

def create_current_feature_table(
    pitchers: pd.DataFrame,
    logs: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_timestamp = pd.Timestamp(target_date)

    for _, pitcher in pitchers.iterrows():
        pitcher_id = int(pitcher["pitcher_id"])
        history = logs.loc[logs["pitcher_id"] == pitcher_id].copy()
        if len(history) < MINIMUM_HISTORY_GAMES:
            continue

        features = create_feature_row(
            history,
            target_timestamp,
            pitcher.get("is_home", np.nan),
        )
        features["pitcher_id"] = pitcher_id
        rows.append(features)

    return pd.DataFrame(rows)



def generate_summaries(
    current_features: pd.DataFrame,
    model: dict[str, Any] | None,
    residuals: np.ndarray,
    holdout_size: int,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    calibrated = model is not None and residuals.size >= 50 and holdout_size > 0

    if calibrated:
        residual_q10 = float(np.quantile(residuals, 0.10))
        residual_q90 = float(np.quantile(residuals, 0.90))
        residual_std = float(np.std(residuals, ddof=0))
    else:
        residual_q10 = float("nan")
        residual_q90 = float("nan")
        residual_std = float("nan")

    for _, row in current_features.iterrows():
        features = {column: float(row[column]) for column in FEATURE_COLUMNS}
        history_games = int(features["history_games"])
        historical_projection = stable_historical_projection(features)

        projected_pitch_count = float(
            np.nanmean(
                [
                    features["last3_avg_pitches"],
                    features["last5_avg_pitches"],
                ]
            )
        )
        recent_outs = float(
            np.nanmean(
                [
                    features["last3_avg_outs"],
                    features["last5_avg_outs"],
                ]
            )
        )
        pitches_per_out = (
            projected_pitch_count / recent_outs
            if np.isfinite(projected_pitch_count)
            and np.isfinite(recent_outs)
            and recent_outs > 0
            else float("nan")
        )

        if model is not None:
            feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            x = model["imputer"].transform(feature_frame)
            rf_projection = float(model["rf"].predict(x)[0])
            et_projection = float(model["et"].predict(x)[0])
            ml_projection = 0.55 * rf_projection + 0.45 * et_projection
            disagreement = abs(rf_projection - et_projection)
            weight = model_blend_weight(history_games, disagreement)
            projection = (
                weight * ml_projection
                + (1.0 - weight) * historical_projection
            )
            method = "chronological_rf_extratrees_workload_blend"
        else:
            ml_projection = float("nan")
            disagreement = 3.0
            weight = 0.0
            projection = historical_projection
            method = "workload_historical_fallback"

        projection = float(
            np.clip(
                projection,
                MINIMUM_PROJECTED_OUTS,
                MAXIMUM_PROJECTED_OUTS,
            )
        )

        history_score = min(history_games / 15.0, 1.0)
        stability = features["recent_std_outs"]
        stability_score = (
            float(np.clip(1.0 - stability / 7.0, 0.0, 1.0))
            if np.isfinite(stability)
            else 0.35
        )
        agreement_score = float(
            np.clip(1.0 - disagreement / 4.0, 0.0, 1.0)
        )
        reliability_score = 100.0 * (
            0.45 * history_score
            + 0.30 * stability_score
            + 0.25 * agreement_score
        )

        if calibrated:
            lower = float(np.clip(projection + residual_q10, 0.0, 27.0))
            upper = float(np.clip(projection + residual_q90, 0.0, 27.0))
            uncertainty_method = "empirical_chronological_holdout_quantiles"
            calibration_status = "CHRONOLOGICAL_HOLDOUT"
            output_residual_std = residual_std
        else:
            output_residual_std = features["season_std_outs"]
            interval_radius = 1.2815515655446004 * output_residual_std
            lower = float(np.clip(projection - interval_radius, 0.0, 27.0))
            upper = float(np.clip(projection + interval_radius, 0.0, 27.0))
            uncertainty_method = "historical_game_dispersion"
            calibration_status = "UNCALIBRATED_FALLBACK"

        summaries.append(
            {
                "pitcher_id": int(row["pitcher_id"]),
                "history_games": history_games,
                "days_rest": features["days_rest"],
                "last3_avg_outs": features["last3_avg_outs"],
                "last5_avg_outs": features["last5_avg_outs"],
                "last10_avg_outs": features["last10_avg_outs"],
                "season_avg_outs": features["season_avg_outs"],
                "season_median_outs": features["season_median_outs"],
                "season_std_outs": features["season_std_outs"],
                "recent_std_outs": features["recent_std_outs"],
                "projected_outs": projection,
                "projected_outs_lower_80": lower,
                "projected_outs_upper_80": upper,
                "projected_outs_residual_std": output_residual_std,
                "baseline_projected_outs": historical_projection,
                "model_projected_outs": ml_projection,
                "projected_pitch_count": projected_pitch_count,
                "projected_pitches_per_out": pitches_per_out,
                "model_blend_weight": weight,
                "projection_reliability_score": reliability_score,
                "projection_confidence": confidence_label(
                    reliability_score,
                    calibrated,
                ),
                "projection_method": method,
                "calibration_status": calibration_status,
                "uncertainty_method": uncertainty_method,
            }
        )

    return pd.DataFrame(summaries)

def build_output(
    pitchers: pd.DataFrame,
    summaries: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    projections = pitchers.merge(
        summaries,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )

    unmatched = projections.loc[projections["history_games"].isna()]
    if not unmatched.empty:
        print("\nPitchers skipped because they lacked sufficient history:")
        columns = [c for c in ["pitcher_id", "pitcher_name", "team", "opponent"] if c in unmatched]
        print(unmatched[columns].to_string(index=False))

    projections = projections.loc[projections["history_games"].notna()].copy()
    if projections.empty:
        raise RuntimeError("No current pitchers had enough history for outs projections.")

    projections["date"] = target_date.isoformat()
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
    "baseline_projected_outs",
    "model_projected_outs",
    "projected_pitch_count",
    "projected_pitches_per_out",
    "model_blend_weight",
    "projection_reliability_score",
    ]
    for column in numeric_columns:
        projections[column] = pd.to_numeric(projections[column], errors="coerce").round(3)

    projections = projections.sort_values(
        ["projected_outs", "pitcher_name"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return projections[OUTPUT_COLUMNS].copy()


def save_output(projections: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".tmp.csv")
    projections.to_csv(temporary_path, index=False)
    temporary_path.replace(OUTPUT_PATH)



def save_calibration_bundle(
    residuals: np.ndarray,
    holdout_mae: float,
    holdout_size: int,
    training_rows: int,
    target_date: date,
) -> None:
    """Persist empirical holdout error data for the probability engine."""
    clean_residuals = np.asarray(residuals, dtype=float)
    clean_residuals = clean_residuals[np.isfinite(clean_residuals)]

    if clean_residuals.size == 0 or holdout_size <= 0:
        print(
            "Calibration bundle was not saved because no valid chronological "
            "holdout residuals were available."
        )
        return

    quantiles = {
        "q05": float(np.quantile(clean_residuals, 0.05)),
        "q10": float(np.quantile(clean_residuals, 0.10)),
        "q25": float(np.quantile(clean_residuals, 0.25)),
        "q50": float(np.quantile(clean_residuals, 0.50)),
        "q75": float(np.quantile(clean_residuals, 0.75)),
        "q90": float(np.quantile(clean_residuals, 0.90)),
        "q95": float(np.quantile(clean_residuals, 0.95)),
    }

    bundle = {
        "market": "pitcher_outs",
        "target_column": "target_outs",
        "holdout_residuals": clean_residuals.astype(float),
        "validation_mae": float(holdout_mae),
        "residual_std": float(np.std(clean_residuals, ddof=0)),
        "residual_mean": float(np.mean(clean_residuals)),
        "residual_median": float(np.median(clean_residuals)),
        "residual_quantiles": quantiles,
        "holdout_size": int(holdout_size),
        "training_rows": int(training_rows),
        "feature_columns": FEATURE_COLUMNS.copy(),
        "generated_before_slate": target_date.isoformat(),
        "model_version": "pitcher_outs_empirical_residuals_v1",
        "uncertainty_method": "chronological_holdout_residuals",
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = CALIBRATION_PATH.with_suffix(".tmp.pkl")
    joblib.dump(bundle, temporary_path)
    temporary_path.replace(CALIBRATION_PATH)

    print(f"Saved pitcher-outs calibration bundle to {CALIBRATION_PATH}")


def project_pitcher_outs() -> pd.DataFrame:
    target_date = get_target_date()

    print("=" * 72)
    print("GENERATING LEAKAGE-SAFE PITCHER OUTS PROJECTIONS")
    print(f"Slate date: {target_date.isoformat()}")
    print("=" * 72)

    pitchers, current_logs = load_current_data(target_date)

    historical_raw = read_required_csv(HISTORICAL_LOGS_PATH, "historical pitcher game logs")
    historical_logs = prepare_logs(historical_raw, target_date)
    print(f"Global historical rows available before slate: {len(historical_logs):,}")

    training = build_training_dataset(historical_logs)
    print(f"Leakage-safe feature rows created: {len(training):,}")

    model, holdout_mae, residuals, holdout_size = fit_leakage_safe_model(training)

    save_calibration_bundle(
        residuals=residuals,
        holdout_mae=holdout_mae,
        holdout_size=holdout_size,
        training_rows=len(training),
        target_date=target_date,
    )

    current_features = create_current_feature_table(pitchers, current_logs, target_date)
    print(f"Current pitchers with sufficient history: {len(current_features):,}")

    summaries = generate_summaries(current_features, model, residuals, holdout_size)
    projections = build_output(pitchers, summaries, target_date)
    save_output(projections)

    print(f"\nSaved {len(projections):,} pitcher-outs projections to {OUTPUT_PATH}")
    if np.isfinite(holdout_mae):
        print(f"Chronological validation MAE used for review: {holdout_mae:.3f} outs")

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
    print(projections[preview_columns].head(40).to_string(index=False))

    return projections


if __name__ == "__main__":
    project_pitcher_outs()
