"""Train focused MLB hitter models for HRR and fantasy score.

This production script intentionally retrains only the two priority hitter markets:
    - hits_runs_rbis
    - fantasy_score

Existing model files for hits, total bases, runs, and RBI are left untouched so the
current daily inference pipeline remains backward compatible.

Inputs:
    data/training/hitter_training_dataset.csv
    data/training/hitter_feature_manifest.csv

Outputs:
    models/hitters/hits_runs_rbis_model.pkl
    models/hitters/fantasy_score_model.pkl
    outputs/hitters/<market>_test_results.csv
    outputs/hitters/<market>_feature_importance.csv
    outputs/hitters/model_summary.csv
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = PROJECT_ROOT / "data" / "training" / "hitter_training_dataset.csv"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "data" / "training" / "hitter_feature_manifest.csv"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "hitters"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "hitters"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.80
MINIMUM_TRAIN_ROWS = 500
MINIMUM_TEST_ROWS = 100
MODEL_THREADS = int(os.getenv("MLB_MODEL_THREADS", "2"))
MAX_FEATURES_PER_MARKET = int(os.getenv("MLB_HITTER_MAX_FEATURES", "140"))

TARGETS = {
    "hits_runs_rbis": "target_hits_runs_rbis",
    "fantasy_score": "target_fantasy_score",
}

RESULT_ID_COLUMNS = [
    "date",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
]

# Strong features are placed first. Remaining eligible features are scored by
# market-specific keywords and then capped to reduce noise and overfitting.
MARKET_PRIORITY_FEATURES = {
    "hits_runs_rbis": [
        "expected_plate_appearances",
        "expected_at_bats",
        "expected_hits_runs_rbis_from_opportunity",
        "last3_hits_runs_rbis_avg",
        "last5_hits_runs_rbis_avg",
        "last10_hits_runs_rbis_avg",
        "last20_hits_runs_rbis_avg",
        "season_avg_hits_runs_rbis",
        "last5_hits_runs_rbis_per_pa",
        "last10_hits_runs_rbis_per_pa",
        "season_hits_runs_rbis_per_pa",
        "last5_hits_runs_rbis_std",
        "last10_hits_runs_rbis_std",
        "last5_hits_runs_rbis_coefficient_variation",
        "trend_last3_vs_last10_hits_runs_rbis",
        "trend_last5_vs_last20_hits_runs_rbis",
        "trend_last10_vs_season_hits_runs_rbis",
        "last5_hits_avg",
        "last10_hits_avg",
        "last5_runs_avg",
        "last10_runs_avg",
        "last5_rbi_avg",
        "last10_rbi_avg",
        "batting_order",
        "top_of_order",
        "middle_of_order",
        "is_home",
        "days_rest",
        "is_back_to_back",
        "opponent_pitcher_match_available",
        "opponent_pitcher_history_available",
        "opponent_pitcher_last5_avg_era",
        "opponent_pitcher_last5_avg_whip",
        "opponent_pitcher_last5_avg_strikeout_rate_bf",
        "opponent_pitcher_last5_avg_walk_rate_bf",
        "opponent_pitcher_last5_avg_home_runs_per_9",
        "opponent_pitcher_last5_avg_fip_component",
    ],
    "fantasy_score": [
        "expected_plate_appearances",
        "expected_at_bats",
        "expected_fantasy_from_opportunity",
        "last3_fantasy_score_avg",
        "last5_fantasy_score_avg",
        "last10_fantasy_score_avg",
        "last20_fantasy_score_avg",
        "season_avg_fantasy_score",
        "last5_fantasy_score_per_pa",
        "last10_fantasy_score_per_pa",
        "season_fantasy_score_per_pa",
        "last5_fantasy_score_std",
        "last10_fantasy_score_std",
        "last5_fantasy_score_coefficient_variation",
        "trend_last3_vs_last10_fantasy_score",
        "trend_last5_vs_last20_fantasy_score",
        "trend_last10_vs_season_fantasy_score",
        "last5_total_bases_avg",
        "last10_total_bases_avg",
        "last5_home_runs_avg",
        "last10_home_runs_avg",
        "last5_walks_avg",
        "last10_walks_avg",
        "last5_stolen_bases_avg",
        "last10_stolen_bases_avg",
        "last5_runs_avg",
        "last5_rbi_avg",
        "batting_order",
        "top_of_order",
        "middle_of_order",
        "is_home",
        "days_rest",
        "is_back_to_back",
        "opponent_pitcher_match_available",
        "opponent_pitcher_history_available",
        "opponent_pitcher_last5_avg_era",
        "opponent_pitcher_last5_avg_whip",
        "opponent_pitcher_last5_avg_strikeout_rate_bf",
        "opponent_pitcher_last5_avg_walk_rate_bf",
        "opponent_pitcher_last5_avg_home_runs_per_9",
        "opponent_pitcher_last5_avg_fip_component",
    ],
}

MARKET_KEYWORDS = {
    "hits_runs_rbis": {
        "hits_runs_rbis": 12,
        "expected_plate": 8,
        "expected_at_bats": 7,
        "hits": 5,
        "runs": 5,
        "rbi": 5,
        "plate_appearances": 4,
        "at_bats": 3,
        "batting_order": 6,
        "top_of_order": 5,
        "middle_of_order": 4,
        "opponent_pitcher": 5,
        "location_split": 3,
        "opponent_split": 3,
        "trend": 3,
        "season": 2,
        "last5": 4,
        "last10": 3,
        "last20": 2,
        "std": 1,
        "coefficient_variation": 2,
    },
    "fantasy_score": {
        "fantasy_score": 12,
        "expected_fantasy": 10,
        "expected_plate": 8,
        "total_bases": 6,
        "home_runs": 6,
        "stolen_bases": 6,
        "walks": 4,
        "hit_by_pitch": 3,
        "runs": 4,
        "rbi": 4,
        "plate_appearances": 4,
        "batting_order": 6,
        "top_of_order": 5,
        "middle_of_order": 4,
        "opponent_pitcher": 5,
        "location_split": 3,
        "trend": 3,
        "season": 2,
        "last5": 4,
        "last10": 3,
        "last20": 2,
        "std": 1,
        "coefficient_variation": 2,
    },
}


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    model: Any
    predictions: np.ndarray
    mae: float
    rmse: float
    median_absolute_error: float
    p90_absolute_error: float
    selection_score: float
    fit_seconds: float


def log(message: str = "") -> None:
    print(message, flush=True)


def build_xgboost_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=325,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=10,
        subsample=0.82,
        colsample_bytree=0.78,
        reg_alpha=0.20,
        reg_lambda=3.0,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        max_bin=128,
        random_state=RANDOM_STATE,
        n_jobs=MODEL_THREADS,
        verbosity=0,
    )


def build_extra_trees_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=16,
        min_samples_leaf=7,
        max_features=0.70,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=MODEL_THREADS,
    )



MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "xgboost": build_xgboost_model,
    "extra_trees": build_extra_trees_model,
}


def load_training_frame() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing training dataset: {DATA_PATH}. Run build_hitter_training_dataset.py first."
        )

    frame = pd.read_csv(DATA_PATH)
    required = {"date", "game_id", "player_id", *TARGETS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training dataset is missing columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "game_id", "player_id"]).copy()
    frame = frame.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)

    if FEATURE_MANIFEST_PATH.exists():
        manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
        if {"feature_order", "feature_name"}.issubset(manifest.columns):
            manifest["feature_order"] = pd.to_numeric(manifest["feature_order"], errors="coerce")
            manifest = manifest.dropna(subset=["feature_order", "feature_name"]).sort_values("feature_order")
            features = [str(x).strip() for x in manifest["feature_name"] if str(x).strip() in frame.columns]
        else:
            features = []
    else:
        features = []

    if not features:
        excluded = {
            "date", "game_id", "player_id", "player_name", "team", "opponent",
            "opponent_pitcher_name", "target_hits", "target_total_bases",
            "target_home_runs", "target_runs", "target_rbi",
            "target_hits_runs_rbis", "target_fantasy_score", "hits", "runs",
            "rbi", "hits_runs_rbis", "fantasy_score", "total_bases",
            "home_runs", "walks", "stolen_bases", "plate_appearances", "at_bats",
        }
        features = [
            column for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        ]

    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    for target in TARGETS.values():
        frame[target] = pd.to_numeric(frame[target], errors="coerce")

    log(f"Loaded {len(frame):,} rows and {len(features):,} eligible features.")
    return frame, features


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = np.array(sorted(frame["date"].dropna().unique()))
    if len(unique_dates) < 10:
        raise ValueError("At least 10 unique dates are required.")
    split_index = max(1, min(int(len(unique_dates) * TRAIN_FRACTION), len(unique_dates) - 1))
    split_date = pd.Timestamp(unique_dates[split_index])
    train = frame.loc[frame["date"] < split_date].copy()
    test = frame.loc[frame["date"] >= split_date].copy()
    if len(train) < MINIMUM_TRAIN_ROWS or len(test) < MINIMUM_TEST_ROWS:
        raise ValueError(f"Chronological split too small: train={len(train)}, test={len(test)}")
    return train, test, split_date


def feature_score(market: str, feature: str) -> int:
    name = feature.casefold()
    score = 0
    for keyword, weight in MARKET_KEYWORDS[market].items():
        if keyword in name:
            score += weight
    if "previous_game" in name:
        score -= 1
    if name in {"month", "day_of_week", "prior_games", "days_rest", "is_home"}:
        score += 2
    return score


def select_market_features(
    market: str,
    train: pd.DataFrame,
    eligible_features: list[str],
) -> list[str]:
    usable = []
    for feature in eligible_features:
        series = pd.to_numeric(train[feature], errors="coerce")
        if series.notna().sum() == 0 or series.nunique(dropna=True) <= 1:
            continue
        usable.append(feature)

    selected: list[str] = []
    for feature in MARKET_PRIORITY_FEATURES[market]:
        if feature in usable and feature not in selected:
            selected.append(feature)

    ranked = sorted(
        (feature for feature in usable if feature not in selected),
        key=lambda feature: (-feature_score(market, feature), feature),
    )
    selected.extend(ranked[: max(0, MAX_FEATURES_PER_MARKET - len(selected))])

    if len(selected) < 10:
        raise ValueError(f"Only {len(selected)} usable features selected for {market}.")
    return selected[:MAX_FEATURES_PER_MARKET]


def calculate_medians(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    return frame[features].median(numeric_only=True).reindex(features).fillna(0.0)


def prepare_matrix(frame: pd.DataFrame, features: list[str], medians: pd.Series) -> pd.DataFrame:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(medians)
    return matrix.astype(np.float32)


def evaluate_candidate(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> CandidateResult:
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    predictions = np.clip(np.asarray(model.predict(x_test), dtype=float), 0.0, None)
    actual = y_test.to_numpy(dtype=float)
    absolute_errors = np.abs(actual - predictions)
    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(mean_squared_error(actual, predictions) ** 0.5)
    median_error = float(np.median(absolute_errors))
    p90_error = float(np.quantile(absolute_errors, 0.90))
    # Favor low typical error while penalizing damaging tail misses.
    selection_score = mae + 0.12 * p90_error + 0.03 * rmse
    return CandidateResult(
        model_name=model_name,
        model=model,
        predictions=predictions,
        mae=mae,
        rmse=rmse,
        median_absolute_error=median_error,
        p90_absolute_error=p90_error,
        selection_score=selection_score,
        fit_seconds=fit_seconds,
    )


def baseline_feature(market: str, frame: pd.DataFrame) -> str | None:
    candidates = {
        "hits_runs_rbis": ["last5_hits_runs_rbis_avg", "last5_avg_hits_runs_rbis"],
        "fantasy_score": ["last5_fantasy_score_avg", "last5_avg_fantasy_score"],
    }[market]
    return next((column for column in candidates if column in frame.columns), None)


def feature_importance(model: Any, features: list[str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "importance", "importance_rank"])
    output = pd.DataFrame({"feature": features, "importance": np.asarray(values, dtype=float)})
    output = output.sort_values("importance", ascending=False).reset_index(drop=True)
    output["importance_rank"] = np.arange(1, len(output) + 1)
    return output


def train_hitter_models() -> pd.DataFrame:
    started = time.perf_counter()
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("FOCUSED HITTER TRAINING: HRR + FANTASY SCORE")
    log(f"Threads per model: {MODEL_THREADS}")
    log("=" * 72)

    frame, eligible_features = load_training_frame()
    train, test, split_date = chronological_split(frame)
    summary_rows: list[dict[str, Any]] = []

    for market, target_column in TARGETS.items():
        market_started = time.perf_counter()
        features = select_market_features(market, train, eligible_features)
        log("")
        log("-" * 72)
        log(f"Training {market} with {len(features)} focused features")
        log(f"Holdout begins: {split_date.date()}")
        log("-" * 72)

        y_train_all = pd.to_numeric(train[target_column], errors="coerce")
        y_test_all = pd.to_numeric(test[target_column], errors="coerce")
        train_mask = y_train_all.notna()
        test_mask = y_test_all.notna()
        if int(train_mask.sum()) < MINIMUM_TRAIN_ROWS or int(test_mask.sum()) < MINIMUM_TEST_ROWS:
            raise RuntimeError(f"Insufficient valid rows for {market}.")

        medians = calculate_medians(train.loc[train_mask], features)
        x_train = prepare_matrix(train.loc[train_mask], features, medians)
        x_test = prepare_matrix(test.loc[test_mask], features, medians)
        y_train = y_train_all.loc[train_mask]
        y_test = y_test_all.loc[test_mask]

        candidates: list[CandidateResult] = []
        for name, factory in MODEL_FACTORIES.items():
            log(f"Training candidate: {name}")
            result = evaluate_candidate(name, factory(), x_train, y_train, x_test, y_test)
            candidates.append(result)
            log(
                f"{name}: MAE={result.mae:.4f} RMSE={result.rmse:.4f} "
                f"P90={result.p90_absolute_error:.4f} Score={result.selection_score:.4f}"
            )

        winner = min(candidates, key=lambda item: (item.selection_score, item.mae))
        log(f"Selected: {winner.model_name}")

        ids = [column for column in RESULT_ID_COLUMNS if column in test.columns]
        results = test.loc[test_mask, ids + [target_column]].copy().rename(columns={target_column: "actual"})
        results["market"] = market
        results["selected_model"] = winner.model_name
        results["prediction"] = winner.predictions
        results["residual"] = results["actual"] - results["prediction"]
        results["absolute_error"] = results["residual"].abs()
        for candidate in candidates:
            results[f"prediction_{candidate.model_name}"] = candidate.predictions

        base_feature = baseline_feature(market, test)
        baseline_mae = np.nan
        if base_feature:
            fallback = float(pd.to_numeric(train[base_feature], errors="coerce").median())
            if not np.isfinite(fallback):
                fallback = 0.0
            baseline_predictions = pd.to_numeric(
                test.loc[test_mask, base_feature], errors="coerce"
            ).fillna(fallback).clip(lower=0.0)
            results["baseline_last5"] = baseline_predictions.to_numpy()
            baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))

        results.to_csv(OUTPUT_DIRECTORY / f"{market}_test_results.csv", index=False)
        residuals = pd.to_numeric(results["residual"], errors="coerce").dropna()
        residuals = residuals.loc[np.isfinite(residuals)].to_numpy(dtype=float)
        if len(residuals) < 100:
            raise RuntimeError(f"Only {len(residuals)} residuals available for {market}.")

        # Refit the winning architecture on all historical rows.
        full_target = pd.to_numeric(frame[target_column], errors="coerce")
        full_mask = full_target.notna()
        full_medians = calculate_medians(frame.loc[full_mask], features)
        full_matrix = prepare_matrix(frame.loc[full_mask], features, full_medians)
        production_model = MODEL_FACTORIES[winner.model_name]()
        production_model.fit(full_matrix, full_target.loc[full_mask])

        importance = feature_importance(production_model, features)
        importance.to_csv(OUTPUT_DIRECTORY / f"{market}_feature_importance.csv", index=False)

        bundle = {
            "model": production_model,
            "model_name": winner.model_name,
            "features": features,
            "medians": full_medians.to_dict(),
            "target": target_column,
            "market": market,
            "training_rows": int(full_mask.sum()),
            "training_start_date": frame["date"].min().date().isoformat(),
            "training_end_date": frame["date"].max().date().isoformat(),
            "validation_split_date": split_date.date().isoformat(),
            "validation_rows": int(test_mask.sum()),
            "validation_mae": winner.mae,
            "validation_rmse": winner.rmse,
            "validation_median_absolute_error": winner.median_absolute_error,
            "validation_p90_absolute_error": winner.p90_absolute_error,
            "validation_selection_score": winner.selection_score,
            "baseline_feature": base_feature,
            "baseline_mae": None if np.isnan(baseline_mae) else baseline_mae,
            "holdout_residuals": residuals.tolist(),
            "holdout_residual_count": len(residuals),
            "candidate_metrics": {
                result.model_name: {
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "median_absolute_error": result.median_absolute_error,
                    "p90_absolute_error": result.p90_absolute_error,
                    "selection_score": result.selection_score,
                    "fit_seconds": result.fit_seconds,
                }
                for result in candidates
            },
            "training_focus": "priority_hrr_and_fantasy_score_v1",
        }
        model_path = MODEL_DIRECTORY / f"{market}_model.pkl"
        joblib.dump(bundle, model_path, compress=3)

        summary_rows.append({
            "market": market,
            "selected_model": winner.model_name,
            "feature_count": len(features),
            "training_rows": int(full_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "validation_split_date": split_date.date().isoformat(),
            "mae": round(winner.mae, 4),
            "rmse": round(winner.rmse, 4),
            "median_absolute_error": round(winner.median_absolute_error, 4),
            "p90_absolute_error": round(winner.p90_absolute_error, 4),
            "selection_score": round(winner.selection_score, 4),
            "baseline_last5_mae": np.nan if np.isnan(baseline_mae) else round(baseline_mae, 4),
            "model_vs_baseline_mae": np.nan if np.isnan(baseline_mae) else round(baseline_mae - winner.mae, 4),
            "residual_count": len(residuals),
            "market_total_seconds": round(time.perf_counter() - market_started, 2),
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        })
        log(f"Saved: {model_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIRECTORY / "model_summary.csv", index=False)
    log("")
    log("=" * 72)
    log("FOCUSED HITTER TRAINING COMPLETE")
    log(f"Runtime: {(time.perf_counter() - started) / 60:.2f} minutes")
    log(summary[["market", "selected_model", "mae", "baseline_last5_mae", "feature_count"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    train_hitter_models()
