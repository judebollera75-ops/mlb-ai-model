import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/final/master_dataset_ballpark.csv")

df = df[df["status"] == "Final"].copy()

feature_cols = [
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
    "park_factor",
]

df = df.dropna(subset=feature_cols + ["pitcher_id"])

# Use actual strikeouts from historical results
results = pd.read_csv("data/historical/strikeout_results.csv")

df = df.merge(
    results[["date", "game_id", "pitcher_id", "actual_strikeouts"]],
    on=["date", "game_id", "pitcher_id"],
    how="left"
)

df = df.dropna(subset=["actual_strikeouts"])

X = df[feature_cols]
y = df["actual_strikeouts"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print()
print("Ballpark model trained")
print("MAE:", round(mae, 3))
print()
print(importance.head(20))
