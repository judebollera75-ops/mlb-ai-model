import os
import pandas as pd

def rank_platform_props():
    os.makedirs("outputs", exist_ok=True)

    props = pd.read_csv("data/platform_props_template.csv")
    projections = pd.read_csv("outputs/calibrated_strikeout_projections.csv")

    merged = props.merge(
        projections,
        left_on="player",
        right_on="pitcher_name",
        how="left"
    )

    merged["projection"] = merged["calibrated_projected_ks"]
    merged["edge"] = merged["projection"] - merged["line"]

    merged["pick"] = merged.apply(
        lambda r: "MORE/YES" if r["edge"] > 0 else "LESS/NO",
        axis=1
    )

    ranked = merged.sort_values("edge", key=lambda x: x.abs(), ascending=False)

    ranked.to_csv("outputs/best_platform_props.csv", index=False)

    print(ranked[[
        "platform",
        "player",
        "market",
        "line",
        "projection",
        "edge",
        "pick"
    ]])

if __name__ == "__main__":
    rank_platform_props()
