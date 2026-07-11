import os
import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


DATA_PATH = "data/training/hitter_training_dataset.csv"
MODEL_DIR = "models/hitters"
OUTPUT_DIR = "outputs/hitters"

TARGETS = {
    "hits": "target_hits",
    "total_bases": "target_total_bases",
    "runs": "target_runs",
    "rbi": "target_rbi",
    "hits_runs_rbis": "target_hits_runs_rbis",
    "fantasy_score": "target_fantasy_score",
}


def train_hitter_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Only pregame information
    feature_cols = [
        column
        for column in df.columns
        if (
            column.startswith("last3_avg_")
            or column.startswith("last5_avg_")
            or column.startswith("last10_avg_")
            or column.startswith("last3_")
            or column.startswith("last5_")
            or column.startswith("last10_")
            or column.startswith("previous_game_")
            or column in ["days_rest", "prior_games"]
        )
    ]

    if not feature_cols:
        raise ValueError("No rolling pregame features were found.")

    for column in feature_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    # Split by unique date so a single slate does not get split across train/test
    unique_dates = sorted(df["date"].dropna().unique())

    if len(unique_dates) < 2:
        raise ValueError("Not enough unique dates to create a train/test split.")

    split_idx = max(1, int(len(unique_dates) * 0.80))
    if split_idx >= len(unique_dates):
        split_idx = len(unique_dates) - 1

    split_date = unique_dates[split_idx]

    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split failed. One side is empty.")

    # Calculate medians using training data only
    medians = train[feature_cols].median()

    X_train = train[feature_cols].fillna(medians)
    X_test = test[feature_cols].fillna(medians)

    print("Pregame features:", len(feature_cols))
    print("Train rows:", len(train))
    print("Test rows:", len(test))
    print("Train range:", train["date"].min(), "to", train["date"].max())
    print("Test range:", test["date"].min(), "to", test["date"].max())
    print()

    summary = []

    for market, target_col in TARGETS.items():
        if target_col not in df.columns:
            print(f"Skipping {market}: {target_col} is missing")
            continue

        y_train = pd.to_numeric(
            train[target_col],
            errors="coerce"
        )

        y_test = pd.to_numeric(
            test[target_col],
            errors="coerce"
        )

        train_mask = y_train.notna()
        test_mask = y_test.notna()

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"Skipping {market}: no train/test rows available")
            continue

        model = XGBRegressor(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train.loc[train_mask],
            y_train.loc[train_mask],
        )

        predictions = model.predict(
            X_test.loc[test_mask]
        )

        # These stats cannot be negative
        predictions = predictions.clip(min=0)

        mae = mean_absolute_error(
            y_test.loc[test_mask],
            predictions,
        )

        # Simple baseline: last5 average of the same stat if available
        baseline_col = f"last5_avg_{market}"
        baseline_mae = None

        if baseline_col in test.columns:
            baseline_preds = pd.to_numeric(
                test.loc[test_mask, baseline_col],
                errors="coerce"
            ).fillna(
                pd.to_numeric(train[baseline_col], errors="coerce").median()
            )

            baseline_mae = mean_absolute_error(
                y_test.loc[test_mask],
                baseline_preds,
            )

        model_path = f"{MODEL_DIR}/{market}_model.pkl"

        joblib.dump(
            {
                "model": model,
                "features": feature_cols,
                "medians": medians.to_dict(),
                "target": target_col,
                "market": market,
                "split_date": str(split_date),
            },
            model_path,
        )

        results = test.loc[
            test_mask,
            [
                "date",
                "game_id",
                "player_id",
                "player_name",
                target_col,
            ],
        ].copy()

        results["prediction"] = predictions
        results["absolute_error"] = (
            results[target_col] - results["prediction"]
        ).abs()

        if baseline_col in test.columns:
            results["baseline_last5"] = pd.to_numeric(
                test.loc[test_mask, baseline_col],
                errors="coerce"
            )
            results["baseline_last5_error"] = (
                results[target_col] - results["baseline_last5"]
            ).abs()

        results.to_csv(
            f"{OUTPUT_DIR}/{market}_test_results.csv",
            index=False,
        )

        summary_row = {
            "market": market,
            "mae": round(mae, 3),
            "test_rows": int(test_mask.sum()),
        }

        if baseline_mae is not None:
            summary_row["baseline_last5_mae"] = round(baseline_mae, 3)
            summary_row["model_vs_baseline"] = round(
                baseline_mae - mae, 3
            )

        summary.append(summary_row)

        print(f"{market.upper()} model trained")
        print("MAE:", round(mae, 3))
        if baseline_mae is not None:
            print("Baseline last5 MAE:", round(baseline_mae, 3))
            print("Model improvement:", round(baseline_mae - mae, 3))
        print("Saved:", model_path)
        print()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        f"{OUTPUT_DIR}/model_summary.csv",
        index=False,
    )

    print("MODEL SUMMARY")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    train_hitter_models()
