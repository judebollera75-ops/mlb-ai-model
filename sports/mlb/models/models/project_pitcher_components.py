import os
import pandas as pd

def project_pitcher_components():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/final/master_dataset_recent.csv")

    # Keep today's/upcoming pitcher rows
    df = df[df["status"].isin(["Pre-Game", "Scheduled"])].copy()

    # First smarter version: use recent form instead of flat assumptions
    df["projected_ip"] = df["last5_avg_ip"].fillna(df["last3_avg_ip"]).fillna(5.5)
    df["projected_hits_allowed"] = df["last5_avg_hits"].fillna(5.0)
    df["projected_walks"] = df["last5_avg_walks"].fillna(2.0)
    df["projected_er"] = df["last5_avg_er"].fillna(2.5)

    df["projected_win_prob"] = 0.35

    df.to_csv("outputs/pitcher_component_projections.csv", index=False)

    print(df[[
        "pitcher_name",
        "team",
        "projected_ip",
        "projected_hits_allowed",
        "projected_walks",
        "projected_er",
        "projected_win_prob"
    ]].head(30))

if __name__ == "__main__":
    project_pitcher_components()
