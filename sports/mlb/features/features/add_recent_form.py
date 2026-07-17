import os
import glob
import pandas as pd


MASTER_PATH = "data/final/master_dataset.csv"
GAME_LOG_FOLDER = "data/game_logs"


def get_latest_game_log():

    files = glob.glob(os.path.join(GAME_LOG_FOLDER, "*.csv"))

    if not files:
        raise FileNotFoundError(
            "No game log CSVs found."
        )

    return max(files, key=os.path.getmtime)


def add_recent_form():

    os.makedirs("data/final", exist_ok=True)

    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(MASTER_PATH)

    master = pd.read_csv(MASTER_PATH)

    latest_log = get_latest_game_log()

    logs = pd.read_csv(latest_log)

    logs["game_date"] = pd.to_datetime(logs["game_date"])

    logs = logs.sort_values(
        ["pitcher_id", "game_date"]
    )

    recent5 = (
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

    recent3 = (
        logs.groupby("pitcher_id")
        .tail(3)
        .groupby("pitcher_id")
        .agg(
            last3_avg_ks=("strikeouts", "mean"),
            last3_avg_ip=("innings", "mean"),
        )
        .reset_index()
    )

    master = master.drop(
        columns=[
            c
            for c in list(recent5.columns) + list(recent3.columns)
            if c != "pitcher_id" and c in master.columns
        ],
        errors="ignore",
    )

    master = master.merge(
        recent5,
        on="pitcher_id",
        how="left",
    )

    master = master.merge(
        recent3,
        on="pitcher_id",
        how="left",
    )

    master.to_csv(
        MASTER_PATH,
        index=False,
    )

    print(
        f"Updated recent form for {len(master)} pitchers."
    )

    print(f"Using game log: {latest_log}")

    return master


if __name__ == "__main__":
    add_recent_form()
