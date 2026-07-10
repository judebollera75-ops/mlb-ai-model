import os
import pandas as pd

def add_opponent_features():
    os.makedirs("data/final", exist_ok=True)

    starts = pd.read_csv("data/historical/pitcher_starts.csv")
    team_stats = pd.read_csv("data/team_stats/team_batting_2026.csv")

    keep_cols = [
        "team",
        "team_k_per_game",
        "runs_per_game",
        "hits_per_game",
        "walks_per_game",
        "avg",
        "obp",
        "slg",
        "ops"
    ]

    opponent_stats = team_stats[keep_cols].copy()
    opponent_stats = opponent_stats.rename(columns={
        "team": "opponent",
        "team_k_per_game": "opp_k_per_game",
        "runs_per_game": "opp_runs_per_game",
        "hits_per_game": "opp_hits_per_game",
        "walks_per_game": "opp_walks_per_game",
        "avg": "opp_avg",
        "obp": "opp_obp",
        "slg": "opp_slg",
        "ops": "opp_ops"
    })

    enhanced = starts.merge(opponent_stats, on="opponent", how="left")
    enhanced.to_csv("data/final/master_dataset_with_opponent.csv", index=False)

    return enhanced

if __name__ == "__main__":
    print(add_opponent_features())
