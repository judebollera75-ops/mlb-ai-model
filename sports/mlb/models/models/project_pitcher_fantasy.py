import os
import pandas as pd

def project_pitcher_fantasy():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("outputs/pitcher_strikeout_projections.csv")

    # Simple first version assumptions
    df["projected_ip"] = 5.5
    df["projected_er"] = 2.5
    df["projected_hits_allowed"] = 5.0
    df["projected_walks"] = 2.0
    df["projected_win"] = 0.35

    # DraftKings MLB pitcher scoring
    df["draftkings_pitcher_points"] = (
        df["projected_ip"] * 2.25
        + df["projected_ks"] * 2
        - df["projected_er"] * 2
        - df["projected_hits_allowed"] * 0.6
        - df["projected_walks"] * 0.6
        + df["projected_win"] * 4
    )

    # FanDuel MLB pitcher scoring
    df["fanduel_pitcher_points"] = (
        df["projected_ip"] * 3
        + df["projected_ks"] * 3
        - df["projected_er"] * 3
        + df["projected_win"] * 6
    )

    df = df.sort_values("draftkings_pitcher_points", ascending=False)

    df.to_csv("outputs/pitcher_fantasy_projections.csv", index=False)

    print(df[[
        "pitcher_name",
        "team",
        "projected_ks",
        "draftkings_pitcher_points",
        "fanduel_pitcher_points"
    ]].head(20))

if __name__ == "__main__":
    project_pitcher_fantasy()
