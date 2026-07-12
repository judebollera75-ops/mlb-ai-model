"""Train production MLB hitter models with chronological validation.

Inputs:
    data/training/hitter_training_dataset.csv
    data/training/hitter_feature_manifest.csv

Outputs:
    models/hitters/<market>_model.pkl
    outputs/hitters/<market>_test_results.csv
    outputs/hitters/<market>_feature_importance.csv
    outputs/hitters/model_summary.csv

Each market compares multiple regression algorithms on a chronological
holdout period. The winning algorithm is then refit on all available history
before being saved for production inference.
"""

from __future__ import annotations

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

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "hitter_training_dataset.csv"
)

FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "hitter_feature_manifest.csv"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "hitters"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "hitters"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.80
MINIMUM_TRAIN_ROWS = 500
MINIMUM_TEST_ROWS = 100

TARGETS = {
    "hits": "target_hits",
    "total_bases": "target_total_bases",
    "runs": "target_runs",
    "rbi": "target_rbi",
    "hits_runs_rbis": "target_hits_runs_rbis",
    "fantasy_score": "target_fantasy_score",
}

BASELINE_FEATURES = {
    "hits": "last5_hits_avg",
    "total_bases": "last5_total_bases_avg",
    "runs": "last5_runs_avg",
    "rbi": "last5_rbi_avg",
    "hits_runs_rbis": "last5_hits_runs_rbis_avg",
    "fantasy_score": "last5_fantasy_score_avg",
}

RESULT_ID_COLUMNS = [
    "date",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
]


@dataclass(frozen=True)
class CandidateResult:
    """Validation result for one candidate model."""

    model_name: str
    model: Any
    predictions: np.ndarray
    mae: float
    rmse: float
    median_absolute_error: float
    p90_absolute_error: float
    selection_score: float


def build_xgboost_model() -> XGBRegressor:
    """Return the production XGBoost candidate."""
    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_extra_trees_model() -> ExtraTreesRegressor:
    """Return the production Extra Trees candidate."""
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=14,
        min_samples_split=12,
        min_samples_leaf=5,
        max_features=0.70,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "xgboost": build_xgboost_model,
    "extra_trees": build_extra_trees_model,
}


def load_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Load the ordered feature list created by the dataset builder."""
    if not FEATURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Feature manifest was not found. Run "
            "build_hitter_training_dataset.py first. "
            f"Expected: {FEATURE_MANIFEST_PATH}"
        )

    manifest = pd.read_csv(FEATURE_MANIFEST_PATH)

    required_manifest_columns = {
        "feature_order",
        "feature_name",
    }

    missing_manifest_columns = (
        required_manifest_columns - set(manifest.columns)
    )

    if missing_manifest_columns:
        raise ValueError(
            "Feature manifest is missing columns: "
            f"{sorted(missing_manifest_columns)}"
        )

    manifest["feature_order"] = pd.to_numeric(
        manifest["feature_order"],
        errors="coerce",
    )

    manifest = manifest.dropna(
        subset=["feature_order", "feature_name"]
    )

    manifest = manifest.sort_values("feature_order")

    manifest_features = (
        manifest["feature_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    available_features = [
        feature
        for feature in manifest_features
        if feature in frame.columns
    ]

    missing_features = [
        feature
        for feature in manifest_features
        if feature not in frame.columns
    ]

    if missing_features:
        print(
            "WARNING: Features listed in the manifest but absent "
            f"from the dataset: {len(missing_features)}"
        )
        print(missing_features[:20])

    if not available_features:
        raise ValueError(
            "No manifest features were available in the training dataset."
        )

    return available_features


def prepare_training_frame() -> tuple[pd.DataFrame, list[str]]:
    """Load and validate the hitter training dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Hitter training dataset was not found. Run "
            "build_hitter_training_dataset.py first. "
            f"Expected: {DATA_PATH}"
        )

    frame = pd.read_csv(DATA_PATH)

    required_columns = {
        "date",
        "game_id",
        "player_id",
        *TARGETS.values(),
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=["date", "game_id", "player_id"]
    ).copy()

    frame = frame.sort_values(
        ["date", "game_id", "player_id"]
    ).reset_index(drop=True)

    feature_columns = load_feature_columns(frame)

    for feature in feature_columns:
        frame[feature] = pd.to_numeric(
            frame[feature],
            errors="coerce",
        )

    for target in TARGETS.values():
        frame[target] = pd.to_numeric(
            frame[target],
            errors="coerce",
        )

    return frame, feature_columns


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split complete slates chronologically into train and test sets."""
    unique_dates = np.array(
        sorted(frame["date"].dropna().unique())
    )

    if len(unique_dates) < 10:
        raise ValueError(
            "At least 10 unique game dates are required for "
            "chronological validation."
        )

    split_index = int(len(unique_dates) * TRAIN_FRACTION)
    split_index = max(1, min(split_index, len(unique_dates) - 1))

    split_date = pd.Timestamp(unique_dates[split_index])

    train = frame.loc[
        frame["date"] < split_date
    ].copy()

    test = frame.loc[
        frame["date"] >= split_date
    ].copy()

    if len(train) < MINIMUM_TRAIN_ROWS:
        raise ValueError(
            f"Training split has only {len(train)} rows. "
            f"At least {MINIMUM_TRAIN_ROWS} are required."
        )

    if len(test) < MINIMUM_TEST_ROWS:
        raise ValueError(
            f"Test split has only {len(test)} rows. "
            f"At least {MINIMUM_TEST_ROWS} are required."
        )

    return train, test, split_date


def remove_unusable_features(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """Remove features that contain no useful training variation."""
    usable_features: list[str] = []

    for feature in feature_columns:
        series = pd.to_numeric(
            train[feature],
            errors="coerce",
        )

        non_missing_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))

        if non_missing_count == 0:
            continue

        if unique_count <= 1:
            continue

        usable_features.append(feature)

    if not usable_features:
        raise ValueError(
            "All candidate features were empty or constant."
        )

    removed_count = len(feature_columns) - len(usable_features)

    if removed_count:
        print(
            f"Removed {removed_count} empty or constant features."
        )

    return usable_features


def calculate_training_medians(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Calculate imputations using training data only."""
    medians = train[feature_columns].median(
        numeric_only=True
    )

    medians = medians.reindex(feature_columns).fillna(0.0)

    return medians


def prepare_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    medians: pd.Series,
) -> pd.DataFrame:
    """Return a finite numeric model matrix."""
    matrix = frame[feature_columns].copy()

    for feature in feature_columns:
        matrix[feature] = pd.to_numeric(
            matrix[feature],
            errors="coerce",
        )

    matrix = matrix.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    matrix = matrix.fillna(medians)

    return matrix.astype(float)


def evaluate_candidate(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> CandidateResult:
    """Fit and evaluate one candidate model."""
    model.fit(x_train, y_train)

    predictions = np.asarray(
        model.predict(x_test),
        dtype=float,
    )

    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    residuals = y_test.to_numpy(dtype=float) - predictions
    absolute_errors = np.abs(residuals)

    mae = float(
        mean_absolute_error(y_test, predictions)
    )

    rmse = float(
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    median_absolute_error = float(
        np.median(absolute_errors)
    )

    p90_absolute_error = float(
        np.quantile(absolute_errors, 0.90)
    )

    # MAE is the primary metric. A small tail-error penalty discourages
    # models that occasionally create extremely poor projections.
    selection_score = float(
        mae + (0.10 * p90_absolute_error)
    )

    return CandidateResult(
        model_name=model_name,
        model=model,
        predictions=predictions,
        mae=mae,
        rmse=rmse,
        median_absolute_error=median_absolute_error,
        p90_absolute_error=p90_absolute_error,
        selection_score=selection_score,
    )


def calculate_baseline_metrics(
    market: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_mask: pd.Series,
    target_column: str,
) -> dict[str, float | None]:
    """Evaluate the corresponding last-five-game average baseline."""
    baseline_feature = BASELINE_FEATURES.get(market)

    if (
        baseline_feature is None
        or baseline_feature not in train.columns
        or baseline_feature not in test.columns
    ):
        return {
            "baseline_mae": None,
            "baseline_rmse": None,
        }

    training_baseline = pd.to_numeric(
        train[baseline_feature],
        errors="coerce",
    )

    fallback_value = float(
        training_baseline.median()
    )

    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    baseline_predictions = pd.to_numeric(
        test.loc[test_mask, baseline_feature],
        errors="coerce",
    ).fillna(fallback_value)

    baseline_predictions = baseline_predictions.clip(lower=0.0)

    actual_values = pd.to_numeric(
        test.loc[test_mask, target_column],
        errors="coerce",
    )

    return {
        "baseline_mae": float(
            mean_absolute_error(
                actual_values,
                baseline_predictions,
            )
        ),
        "baseline_rmse": float(
            mean_squared_error(
                actual_values,
                baseline_predictions,
            )
            ** 0.5
        ),
    }


def extract_feature_importance(
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return a sorted feature-importance table when supported."""
    importance_values = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importance_values is None:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "importance_rank",
            ]
        )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": np.asarray(
                importance_values,
                dtype=float,
            ),
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance["importance_rank"] = (
        np.arange(len(importance)) + 1
    )

    return importance


def refit_production_model(
    model_name: str,
    full_frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[Any, dict[str, float], int]:
    """Refit the winning algorithm using all available history."""
    target = pd.to_numeric(
        full_frame[target_column],
        errors="coerce",
    )

    valid_mask = target.notna()

    production_frame = full_frame.loc[valid_mask].copy()
    production_target = target.loc[valid_mask]

    production_medians = (
        production_frame[feature_columns]
        .median(numeric_only=True)
        .reindex(feature_columns)
        .fillna(0.0)
    )

    production_matrix = prepare_matrix(
        production_frame,
        feature_columns,
        production_medians,
    )

    production_model = MODEL_FACTORIES[model_name]()
    production_model.fit(
        production_matrix,
        production_target,
    )

    return (
        production_model,
        production_medians.to_dict(),
        int(valid_mask.sum()),
    )


def save_market_results(
    market: str,
    target_column: str,
    test: pd.DataFrame,
    test_mask: pd.Series,
    winner: CandidateResult,
    candidate_results: list[CandidateResult],
    baseline_feature: str | None,
) -> pd.DataFrame:
    """Save holdout predictions and residuals for calibration."""
    available_id_columns = [
        column
        for column in RESULT_ID_COLUMNS
        if column in test.columns
    ]

    results = test.loc[
        test_mask,
        available_id_columns + [target_column],
    ].copy()

    results = results.rename(
        columns={target_column: "actual"}
    )

    results["market"] = market
    results["selected_model"] = winner.model_name
    results["prediction"] = winner.predictions
    results["residual"] = (
        results["actual"] - results["prediction"]
    )
    results["absolute_error"] = results["residual"].abs()

    for candidate in candidate_results:
        results[
            f"prediction_{candidate.model_name}"
        ] = candidate.predictions

    if (
        baseline_feature
        and baseline_feature in test.columns
    ):
        results["baseline_last5"] = pd.to_numeric(
            test.loc[test_mask, baseline_feature],
            errors="coerce",
        )

        results["baseline_last5_error"] = (
            results["actual"] - results["baseline_last5"]
        ).abs()

    output_path = (
        OUTPUT_DIRECTORY
        / f"{market}_test_results.csv"
    )

    results.to_csv(output_path, index=False)

    return results


def train_hitter_models() -> pd.DataFrame:
    """Train, validate, select, and save all hitter-market models."""
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame, manifest_features = prepare_training_frame()

    train, test, split_date = chronological_split(frame)

    feature_columns = remove_unusable_features(
        train,
        manifest_features,
    )

    validation_medians = calculate_training_medians(
        train,
        feature_columns,
    )

    x_train_all = prepare_matrix(
        train,
        feature_columns,
        validation_medians,
    )

    x_test_all = prepare_matrix(
        test,
        feature_columns,
        validation_medians,
    )

    print("=" * 72)
    print("Training production hitter models")
    print(f"Pregame features: {len(feature_columns):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print(
        "Training range: "
        f"{train['date'].min().date()} to "
        f"{train['date'].max().date()}"
    )
    print(
        "Testing range: "
        f"{test['date'].min().date()} to "
        f"{test['date'].max().date()}"
    )
    print(f"Holdout begins: {split_date.date()}")
    print("=" * 72)

    summary_rows: list[dict[str, Any]] = []

    for market, target_column in TARGETS.items():
        print(f"\n{'-' * 72}")
        print(f"Market: {market}")
        print(f"Target: {target_column}")

        y_train = pd.to_numeric(
            train[target_column],
            errors="coerce",
        )

        y_test = pd.to_numeric(
            test[target_column],
            errors="coerce",
        )

        train_mask = y_train.notna()
        test_mask = y_test.notna()

        train_row_count = int(train_mask.sum())
        test_row_count = int(test_mask.sum())

        if train_row_count < MINIMUM_TRAIN_ROWS:
            print(
                f"Skipping {market}: only {train_row_count} "
                "valid training rows."
            )
            continue

        if test_row_count < MINIMUM_TEST_ROWS:
            print(
                f"Skipping {market}: only {test_row_count} "
                "valid testing rows."
            )
            continue

        x_train = x_train_all.loc[train_mask]
        x_test = x_test_all.loc[test_mask]

        market_y_train = y_train.loc[train_mask]
        market_y_test = y_test.loc[test_mask]

        candidate_results: list[CandidateResult] = []

        for model_name, factory in MODEL_FACTORIES.items():
            print(f"Training candidate: {model_name}")

            candidate = evaluate_candidate(
                model_name=model_name,
                model=factory(),
                x_train=x_train,
                y_train=market_y_train,
                x_test=x_test,
                y_test=market_y_test,
            )

            candidate_results.append(candidate)

            print(
                f"  MAE: {candidate.mae:.4f} | "
                f"RMSE: {candidate.rmse:.4f} | "
                f"P90 error: {candidate.p90_absolute_error:.4f} | "
                f"Selection score: {candidate.selection_score:.4f}"
            )

        winner = min(
            candidate_results,
            key=lambda result: (
                result.selection_score,
                result.mae,
            ),
        )

        baseline_metrics = calculate_baseline_metrics(
            market=market,
            train=train,
            test=test,
            test_mask=test_mask,
            target_column=target_column,
        )

        baseline_feature = BASELINE_FEATURES.get(market)

        holdout_results = save_market_results(
            market=market,
            target_column=target_column,
            test=test,
            test_mask=test_mask,
            winner=winner,
            candidate_results=candidate_results,
            baseline_feature=baseline_feature,
        )

        residuals = (
            holdout_results["actual"]
            - holdout_results["prediction"]
        ).to_numpy(dtype=float)

        residuals = residuals[
            np.isfinite(residuals)
        ]

        (
            production_model,
            production_medians,
            production_rows,
        ) = refit_production_model(
            model_name=winner.model_name,
            full_frame=frame,
            target_column=target_column,
            feature_columns=feature_columns,
        )

        feature_importance = extract_feature_importance(
            production_model,
            feature_columns,
        )

        feature_importance.to_csv(
            OUTPUT_DIRECTORY
            / f"{market}_feature_importance.csv",
            index=False,
        )

        model_bundle = {
            "model": production_model,
            "model_name": winner.model_name,
            "features": feature_columns,
            "medians": production_medians,
            "target": target_column,
            "market": market,
            "training_rows": production_rows,
            "training_start_date": (
                frame["date"].min().date().isoformat()
            ),
            "training_end_date": (
                frame["date"].max().date().isoformat()
            ),
            "validation_split_date": (
                split_date.date().isoformat()
            ),
            "validation_rows": test_row_count,
            "validation_mae": winner.mae,
            "validation_rmse": winner.rmse,
            "validation_median_absolute_error": (
                winner.median_absolute_error
            ),
            "validation_p90_absolute_error": (
                winner.p90_absolute_error
            ),
            "validation_selection_score": (
                winner.selection_score
            ),
            "baseline_feature": baseline_feature,
            "baseline_mae": baseline_metrics["baseline_mae"],
            "baseline_rmse": baseline_metrics["baseline_rmse"],
            "holdout_residuals": residuals.tolist(),
            "candidate_metrics": {
                candidate.model_name: {
                    "mae": candidate.mae,
                    "rmse": candidate.rmse,
                    "median_absolute_error": (
                        candidate.median_absolute_error
                    ),
                    "p90_absolute_error": (
                        candidate.p90_absolute_error
                    ),
                    "selection_score": (
                        candidate.selection_score
                    ),
                }
                for candidate in candidate_results
            },
        }

        model_path = (
            MODEL_DIRECTORY
            / f"{market}_model.pkl"
        )

        joblib.dump(model_bundle, model_path)

        baseline_mae = baseline_metrics["baseline_mae"]

        summary_row: dict[str, Any] = {
            "market": market,
            "selected_model": winner.model_name,
            "feature_count": len(feature_columns),
            "training_rows": production_rows,
            "test_rows": test_row_count,
            "validation_split_date": split_date.date().isoformat(),
            "mae": round(winner.mae, 4),
            "rmse": round(winner.rmse, 4),
            "median_absolute_error": round(
                winner.median_absolute_error,
                4,
            ),
            "p90_absolute_error": round(
                winner.p90_absolute_error,
                4,
            ),
            "selection_score": round(
                winner.selection_score,
                4,
            ),
            "baseline_last5_mae": (
                round(baseline_mae, 4)
                if baseline_mae is not None
                else np.nan
            ),
            "model_vs_baseline_mae": (
                round(baseline_mae - winner.mae, 4)
                if baseline_mae is not None
                else np.nan
            ),
            "residual_count": len(residuals),
            "model_path": str(
                model_path.relative_to(PROJECT_ROOT)
            ),
        }

        for candidate in candidate_results:
            summary_row[
                f"{candidate.model_name}_mae"
            ] = round(candidate.mae, 4)

            summary_row[
                f"{candidate.model_name}_selection_score"
            ] = round(candidate.selection_score, 4)

        summary_rows.append(summary_row)

        print(f"\nSelected model: {winner.model_name}")
        print(f"Holdout MAE: {winner.mae:.4f}")

        if baseline_mae is not None:
            print(f"Last-five baseline MAE: {baseline_mae:.4f}")
            print(
                "MAE improvement versus baseline: "
                f"{baseline_mae - winner.mae:.4f}"
            )

        print(f"Residuals saved: {len(residuals):,}")
        print(f"Production rows: {production_rows:,}")
        print(f"Saved model: {model_path}")

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        raise RuntimeError(
            "No hitter models were successfully trained."
        )

    expected_markets = set(TARGETS)
    trained_markets = set(summary["market"])

    missing_markets = expected_markets - trained_markets

    if missing_markets:
        raise RuntimeError(
            "The following hitter models were not trained: "
            f"{sorted(missing_markets)}"
        )

    summary_path = (
        OUTPUT_DIRECTORY
        / "model_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("HITTER MODEL SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")

    return summary


if __name__ == "__main__":
    train_hitter_models()
