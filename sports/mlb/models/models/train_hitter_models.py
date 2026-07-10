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
}


def train_hitter_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Only pregame information.
    feature_cols = [
        column
        for column in df.columns
        if (
            column.startswith("last3_avg_")
            or column.startswith("last5_avg_")
            or column.startswith("last10_avg_")
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

    # Use the first 80% chronologically for training.
    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    # Calculate medians using training data only.
    medians = train[feature_cols].median()

    X_train = train[feature_cols].fillna(medians)
    X_test = test[feature_cols].fillna(medians)

    print("Pregame features:", len(feature_cols))
    print("Train rows:", len(train))
    print("Test rows:", len(test))
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

        # These stats cannot be negative.
        predictions = predictions.clip(min=0)

        mae = mean_absolute_error(
            y_test.loc[test_mask],
            predictions,
        )

        model_path = f"{MODEL_DIR}/{market}_model.pkl"

        joblib.dump(
            {
                "model": model,
                "features": feature_cols,
                "medians": medians.to_dict(),
                "target": target_col,
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

        results.to_csv(
            f"{OUTPUT_DIR}/{market}_test_results.csv",
            index=False,
        )

        summary.append({
            "market": market,
            "mae": round(mae, 3),
            "test_rows": int(test_mask.sum()),
        })

        print(f"{market.upper()} model trained")
        print("MAE:", round(mae, 3))
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
