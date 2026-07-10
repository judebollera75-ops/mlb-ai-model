import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

results = pd.read_csv("data/historical/strikeout_results.csv")
features = pd.read_csv("data/final/master_dataset_rest.csv")

data = results.merge(
    features,
    on=["date", "game_id", "pitcher_id", "pitcher_name", "team", "opponent", "side"],
    how="left"
)

feature_cols = [
    "actual_ip",
    "actual_walks",
    "actual_hits",
    "actual_earned_runs",
    "opp_k_per_game",
    "opp_runs_per_game",
    "opp_hits_per_game",
    "opp_walks_per_game",
    "opp_avg",
    "opp_obp",
    "opp_slg",
    "opp_ops",
    "last5_avg_ks",
    "last5_avg_ip",
    "last5_avg_hits",
    "last5_avg_walks",
    "last5_avg_er",
    "last3_avg_ks",
    "last3_avg_ip",
    "days_rest",
    "short_rest",
    "normal_rest",
    "extra_rest",
]

data = data.dropna(subset=["actual_strikeouts"])

for col in feature_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())

X = data[feature_cols]
y = pd.to_numeric(data["actual_strikeouts"], errors="coerce")

valid = y.notna()
X = X[valid]
y = y[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.035,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

joblib.dump(model, "models/xgboost_rest_strikeout_model.pkl")

out = pd.DataFrame({
    "actual_ks": y_test.values,
    "predicted_ks": preds
})

out.to_csv("outputs/xgboost_rest_test_results.csv", index=False)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv("outputs/xgboost_rest_feature_importance.csv", index=False)

print("XGBoost rest model trained successfully")
print("MAE:", round(mae, 3))
print()
print("Top feature importance:")
print(importance.head(20))
print()
print(out.head(20))
