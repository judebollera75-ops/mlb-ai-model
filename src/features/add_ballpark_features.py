
import pandas as pd

def add_ballpark():

    master = pd.read_csv("data/final/master_dataset_recent.csv")
    parks = pd.read_csv("data/ballpark/ballpark_factors.csv")

    master = master.merge(
        parks,
        left_on="team",
        right_on="team",
        how="left"
    )

    master.to_csv(
        "data/final/master_dataset_ballpark.csv",
        index=False
    )

    return master

if __name__ == "__main__":
    print(add_ballpark())
