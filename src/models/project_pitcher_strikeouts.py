import os
import pandas as pd

def project_strikeouts():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/final/master_dataset.csv")

    # Simple first projection:
    # 70% recent K average + 30% season K per start
    df["season_k_per_start"] = df["strikeouts"] / df["games_started"]
    df["projected_ks"] = (df["avg_k"] * 0.70) + (df["season_k_per_start"] * 0.30)

    output = df[[
        "pitcher_name",
        "team",
        "games_started",
        "strikeouts",
        "season_k_per_start",
        "avg_k",
        "projected_ks"
    ]].copy()

    output = output.sort_values("projected_ks", ascending=False)

    output.to_csv("outputs/pitcher_strikeout_projections.csv", index=False)

    return output

if __name__ == "__main__":
    print(project_strikeouts())
