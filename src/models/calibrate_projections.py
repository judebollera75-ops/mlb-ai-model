import os
import pandas as pd

def calibrate_projections():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("outputs/pitcher_strikeout_projections.csv")

    df["raw_projected_ks"] = df["projected_ks"]

    # Keep projections in a realistic MLB starter range
    df["calibrated_projected_ks"] = df["projected_ks"].clip(lower=2.0, upper=11.5)

    df = df.sort_values("calibrated_projected_ks", ascending=False)

    df.to_csv("outputs/calibrated_strikeout_projections.csv", index=False)

    print(df[[
        "pitcher_name",
        "team",
        "raw_projected_ks",
        "calibrated_projected_ks"
    ]])

if __name__ == "__main__":
    calibrate_projections()
