import os
import pandas as pd

def build_pitcher_features(target_date="2026-07-09"):
    os.makedirs("data/features", exist_ok=True)

    logs = pd.read_csv(f"data/game_logs/{target_date}.csv")

    features = (
        logs.groupby(["pitcher_id", "pitcher_name", "team"])
        .agg(
            avg_ip=("innings", "mean"),
            avg_k=("strikeouts", "mean"),
            avg_bb=("walks", "mean"),
            avg_hits=("hits", "mean"),
            avg_er=("earned_runs", "mean"),
            avg_hr=("home_runs", "mean"),
            avg_era=("era", "mean"),
        )
        .reset_index()
    )

    features.to_csv("data/features/pitcher_features.csv", index=False)
    return features

if __name__ == "__main__":
    print(build_pitcher_features())
