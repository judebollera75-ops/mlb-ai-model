import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

results = pd.read_csv("data/historical/strikeout_results.csv")

# Basic cleanup
results = results.dropna(subset=["actual_strikeouts", "actual_ip"])
results["actual_strikeouts"] = pd.to_numeric(results["actual_strikeouts"], errors="coerce")
results["actual_ip"] = pd.to_numeric(results["actual_ip"], errors="coerce")
results["actual_walks"] = pd.to_numeric(results["actual_walks"], errors="coerce")
results["actual_hits"] = pd.to_numeric(results["actual_hits"], errors="coerce")
results["actual_earned_runs"] = pd.to_numeric(results["actual_earned_runs"], errors="coerce")
results = results.dropna()

# Simple starter features
features = results[[
    "actual_ip",
    "actual_walks",
    "actual_hits",
    "actual_earned_runs"
]]

target = results["actual_strikeouts"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

joblib.dump(model, "models/strikeout_model.pkl")

summary = pd.DataFrame({
    "actual_ks": y_test.values,
    "predicted_ks": preds
})

summary.to_csv("outputs/strikeout_model_test_results.csv", index=False)

print("Model trained successfully")
print("Mean absolute error:", round(mae, 3))
print(summary.head(20))
