import os
import sys
import pandas as pd

sys.path.append(".")

from shared.betting.edge_engine import rank_props

def rank_platform_props():
    os.makedirs("outputs", exist_ok=True)

    props = pd.read_csv("data/platform_props_template.csv")

    k = pd.read_csv("outputs/calibrated_strikeout_projections.csv")
    k = k.rename(columns={
        "pitcher_name": "player",
        "calibrated_projected_ks": "projection"
    })
    k["market"] = "pitcher_strikeouts"

    f = pd.read_csv("outputs/pitcher_fantasy_projections.csv")
    f = f.rename(columns={
        "pitcher_name": "player",
        "draftkings_pitcher_points": "projection"
    })
    f["market"] = "pitcher_fantasy_score"

    o = pd.read_csv("outputs/pitcher_outs_projections.csv")
    o = o.rename(columns={
        "pitcher_name": "player",
        "projected_outs": "projection"
    })
    o["market"] = "pitcher_outs"

    projections = pd.concat([
        k[["player", "market", "projection"]],
        f[["player", "market", "projection"]],
        o[["player", "market", "projection"]],
    ])

    ranked = rank_props(props, projections)

    ranked.to_csv("outputs/best_platform_props.csv", index=False)

    print(ranked[[
        "grade",
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
