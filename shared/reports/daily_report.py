import os
import pandas as pd

def build_daily_report():
    os.makedirs("outputs", exist_ok=True)

    reports = []

    mlb_path = "outputs/best_platform_props.csv"

    if os.path.exists(mlb_path):
        mlb = pd.read_csv(mlb_path)
        mlb["sport"] = "MLB"
        reports.append(mlb)

    if not reports:
        print("No reports found yet.")
        return

    final = pd.concat(reports, ignore_index=True)

    final = final.sort_values(
        "edge",
        key=lambda x: x.abs(),
        ascending=False
    )

    final.to_csv("outputs/universal_daily_report.csv", index=False)

    print("\nUNIVERSAL DAILY PROP REPORT\n")
    print(final[[
        "sport",
        "grade",
        "platform",
        "player",
        "market",
        "line",
        "projection",
        "edge",
        "pick"
    ]].to_string(index=False))

if __name__ == "__main__":
    build_daily_report()
