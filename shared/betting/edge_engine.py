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

def rank_props(props, projections):
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

    return merged.sort_values(
        "edge",
        key=lambda x: x.abs(),
        ascending=False
    )
