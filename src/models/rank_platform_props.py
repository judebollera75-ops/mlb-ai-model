import os
import pandas as pd

def grade_edge(edge):
    edge_abs = abs(edge)

    if edge_abs >= 2.0:
        return "A+"
    elif edge_abs >= 1.5:
        return "A"
    elif edge_abs >= 1.0:
        return "B+"
    elif edge_abs >= 0.5:
        return "B"
    else:
        return "C / PASS"

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

    projections = pd.concat([
        k[["player", "market", "projection"]],
        f[["player", "market", "projection"]]
    ])

    merged = props.merge(
        projections,
        on=["player", "market"],
        how="left"
    )

    merged["edge"] = merged["projection"] - merged["line"]

    merged["pick"] = merged.apply(
        lambda r: "MORE/YES" if r["edge"] > 0 else "LESS/NO",
        axis=1
    )

    merged["grade"] = merged["edge"].apply(grade_edge)

    merged = merged.sort_values("edge", key=lambda x: x.abs(), ascending=False)

    merged.to_csv("outputs/best_platform_props.csv", index=False)

    print(merged[[
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
