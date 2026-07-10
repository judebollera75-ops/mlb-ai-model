import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/training/strikeout_training_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

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
]

df = df.dropna(subset=["actual_strikeouts"])

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

split_index = int(len(df) * 0.75)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[feature_cols]
y_train = train["actual_strikeouts"]

X_test = test[feature_cols]
y_test = test["actual_strikeouts"]

model = XGBRegressor(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

joblib.dump(model, "models/time_split_xgboost_strikeout_model.pkl")

print("Time-split XGBoost trained")
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Test date range:", test["date"].min(), "to", test["date"].max())
print("MAE:", round(mae, 3))
