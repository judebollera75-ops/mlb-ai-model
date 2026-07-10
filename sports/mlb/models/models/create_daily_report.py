import os
import pandas as pd

def create_daily_report():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("outputs/best_platform_props.csv")

    report = df[[
        "grade",
        "platform",
        "player",
        "market",
        "line",
        "projection",
        "edge",
        "pick"
    ]].copy()

    report = report.sort_values("edge", key=lambda x: x.abs(), ascending=False)

    report.to_csv("outputs/daily_best_bets_report.csv", index=False)

    print("\nTODAY'S BEST PROP EDGES\n")
    print(report.to_string(index=False))

if __name__ == "__main__":
    create_daily_report()
