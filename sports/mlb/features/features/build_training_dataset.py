import os
import pandas as pd

def build_training_dataset():
    os.makedirs("data/training", exist_ok=True)

    results = pd.read_csv("data/historical/strikeout_results.csv")
    features = pd.read_csv("data/final/master_dataset_recent.csv")

    data = results.merge(
        features,
        on=["date", "game_id", "pitcher_id", "pitcher_name", "team", "opponent", "side"],
        how="left"
    )

    data.to_csv("data/training/strikeout_training_dataset.csv", index=False)

    print("Training dataset created")
    print(data.shape)
    print(data.head())

if __name__ == "__main__":
    build_training_dataset()
