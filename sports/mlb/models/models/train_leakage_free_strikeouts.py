import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/training/strikeout_training_dataset.csv"
MODEL_PATH = "models/leakage_free_strikeout_model.pkl"
RESULTS_PATH = "outputs/leakage_free_strikeout_test_results.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["actual_strikeouts"] = pd.to_numeric(
    df["actual_strikeouts"],
    errors="coerce"
)

# Only information that should be available before the game
candidate_features = [
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

# Use only columns that actually exist
feature_cols = [
    col for col in candidate_features
    if col in df.columns
]

if not feature_cols:
    raise ValueError("No valid pregame feature columns were found.")

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "actual_strikeouts"])
df = df.sort_values("date").reset_index(drop=True)

# Fill missing values using medians from the available data
df[feature_cols] = df[feature_cols].fillna(
    df[feature_cols].median()
)

split_index = int(len(df) * 0.75)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

X_train = train[feature_cols]
y_train = train["actual_strikeouts"]

X_test = test[feature_cols]
y_test = test["actual_strikeouts"]

model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

joblib.dump(
    {
        "model": model,
        "features": feature_cols,
    },
    MODEL_PATH
)

results = test[[
    "date",
    "pitcher_name",
    "actual_strikeouts"
]].copy()

results["predicted_strikeouts"] = predictions
results["absolute_error"] = (
    results["actual_strikeouts"]
    - results["predicted_strikeouts"]
).abs()

results.to_csv(RESULTS_PATH, index=False)

print("Leakage-free strikeout model trained")
print("Features used:", feature_cols)
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Test dates:", test["date"].min(), "to", test["date"].max())
print("MAE:", round(mae, 3))
print()
print(results.head(20).to_string(index=False))
