import os
import pandas as pd


MASTER_PATH = "data/final/master_dataset.csv"
TEAM_STATS_PATH = "data/team_stats/team_batting_2026.csv"


def add_opponent_features():

    os.makedirs("data/final", exist_ok=True)

    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(MASTER_PATH)

    if not os.path.exists(TEAM_STATS_PATH):
        raise FileNotFoundError(TEAM_STATS_PATH)

    master = pd.read_csv(MASTER_PATH)
    team_stats = pd.read_csv(TEAM_STATS_PATH)

    keep_cols = [
        "team",
        "team_k_per_game",
        "runs_per_game",
        "hits_per_game",
        "walks_per_game",
        "avg",
        "obp",
        "slg",
        "ops",
    ]

    opponent_stats = (
        team_stats[keep_cols]
        .copy()
        .rename(
            columns={
                "team": "opponent",
                "team_k_per_game": "opp_k_per_game",
                "runs_per_game": "opp_runs_per_game",
                "hits_per_game": "opp_hits_per_game",
                "walks_per_game": "opp_walks_per_game",
                "avg": "opp_avg",
                "obp": "opp_obp",
                "slg": "opp_slg",
                "ops": "opp_ops",
            }
        )
    )

    master = master.drop(
        columns=[
            c
            for c in opponent_stats.columns
            if c != "opponent" and c in master.columns
        ],
        errors="ignore",
    )

    master = master.merge(
        opponent_stats,
        on="opponent",
        how="left",
    )

    master.to_csv(
        MASTER_PATH,
        index=False,
    )

    print(
        f"Added opponent features to {len(master)} pitchers."
    )

    return master


if __name__ == "__main__":
    add_opponent_features()
