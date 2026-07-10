import pandas as pd

def implied_prob(american_odds):
    odds = float(american_odds)

    if odds < 0:
        return abs(odds) / (abs(odds) + 100)

    return 100 / (odds + 100)

def calculate_edges():
    probs = pd.read_csv("outputs/probability_table.csv")
    lines = pd.read_csv("data/sportsbook_lines_template.csv")

    merged = lines.merge(
        probs,
        left_on=["pitcher", "line"],
        right_on=["pitcher", "line"],
        how="left"
    )

    merged["model_prob"] = merged.apply(
        lambda r: r["over_prob"] if r["side"] == "OVER" else r["under_prob"],
        axis=1
    )

    merged["book_implied_prob"] = merged["sportsbook_odds"].apply(implied_prob)

    merged["edge"] = merged["model_prob"] - merged["book_implied_prob"]

    merged = merged.sort_values("edge", ascending=False)

    merged.to_csv("outputs/edge_finder_results.csv", index=False)

    print(merged[[
        "pitcher",
        "line",
        "side",
        "sportsbook_odds",
        "model_prob",
        "book_implied_prob",
        "edge"
    ]])

if __name__ == "__main__":
    calculate_edges()
