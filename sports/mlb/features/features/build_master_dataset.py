import os
import pandas as pd

def build_master_dataset(target_date="2026-07-10"):
    os.makedirs("data/final", exist_ok=True)

    pitchers = pd.read_csv(f"data/pitchers/{target_date}.csv")
    stats = pd.read_csv(f"data/pitcher_stats/{target_date}.csv")
    features = pd.read_csv("data/features/pitcher_features.csv")

    master = (
        pitchers
        .merge(stats, on="pitcher_id", how="left", suffixes=("", "_season"))
        .merge(features, on="pitcher_id", how="left", suffixes=("", "_recent"))
    )

    master.to_csv("data/final/master_dataset.csv", index=False)
    return master

if __name__ == "__main__":
    print(build_master_dataset())
