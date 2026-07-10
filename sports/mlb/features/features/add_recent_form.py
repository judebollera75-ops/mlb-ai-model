import pandas as pd
import os

def add_recent_form():

    os.makedirs("data/final", exist_ok=True)

    master = pd.read_csv("data/final/master_dataset_with_opponent.csv")

    logs = pd.read_csv("data/game_logs/2026-07-10.csv")

    logs = logs.sort_values(["pitcher_id", "game_date"])

    recent = (
        logs.groupby("pitcher_id")
            .tail(5)
            .groupby("pitcher_id")
            .agg(
                last5_avg_ks=("strikeouts", "mean"),
                last5_avg_ip=("innings", "mean"),
                last5_avg_hits=("hits", "mean"),
                last5_avg_walks=("walks", "mean"),
                last5_avg_er=("earned_runs", "mean"),
            )
            .reset_index()
    )

    last3 = (
        logs.groupby("pitcher_id")
            .tail(3)
            .groupby("pitcher_id")
            .agg(
                last3_avg_ks=("strikeouts", "mean"),
                last3_avg_ip=("innings", "mean"),
            )
            .reset_index()
    )

    master = master.merge(recent, on="pitcher_id", how="left")
    master = master.merge(last3, on="pitcher_id", how="left")

    master.to_csv(
        "data/final/master_dataset_recent.csv",
        index=False
    )

    return master


if __name__ == "__main__":
    print(add_recent_form())
