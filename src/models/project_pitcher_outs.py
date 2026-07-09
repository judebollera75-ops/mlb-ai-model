import os
import pandas as pd

def innings_to_outs(ip):
    """
    Converts baseball innings format:
    5.0 = 15 outs
    5.1 = 16 outs
    5.2 = 17 outs
    """
    if pd.isna(ip):
        return None

    whole = int(float(ip))
    decimal = round(float(ip) - whole, 1)

    extra_outs = 0
    if decimal == 0.1:
        extra_outs = 1
    elif decimal == 0.2:
        extra_outs = 2

    return whole * 3 + extra_outs

def project_pitcher_outs():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/final/master_dataset_recent.csv")

    df["projected_ip"] = df["last5_avg_ip"].fillna(df["last3_avg_ip"]).fillna(5.5)
    df["projected_outs"] = df["projected_ip"].apply(innings_to_outs)

    df.to_csv("outputs/pitcher_outs_projections.csv", index=False)

    print(df[[
        "pitcher_name",
        "team",
        "projected_ip",
        "projected_outs"
    ]].head(30))

if __name__ == "__main__":
    project_pitcher_outs()
