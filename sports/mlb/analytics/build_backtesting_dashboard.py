from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

HISTORY_FILE = ROOT / "outputs" / "history" / "mlb_bet_results.csv"
OUTPUT_DIR = ROOT / "outputs" / "analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pct(x):
    return round(x * 100, 2)


def main():
    if not HISTORY_FILE.exists():
        print("History file not found.")
        return

    df = pd.read_csv(HISTORY_FILE)

    if df.empty:
        print("History file is empty.")
        return

    df = df[df["grading_status"] == "GRADED"].copy()

    if df.empty:
        print("No graded bets.")
        return

    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(1)

    wins = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()
    pushes = (df["outcome"] == "PUSH").sum()

    total = len(df)

    total_profit = df["profit"].sum()
    total_stake = df["stake"].sum()

    roi = total_profit / total_stake if total_stake else 0

    summary = pd.DataFrame(
        [
            {
                "bets": total,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": pct(wins / total),
                "units": round(total_profit, 2),
                "roi": pct(roi),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "overall_summary.csv",
        index=False,
    )

    market = (
        df.groupby("market")
        .agg(
            bets=("market", "count"),
            wins=("outcome", lambda x: (x == "WIN").sum()),
            units=("profit", "sum"),
        )
        .reset_index()
    )

    market["win_rate"] = (
        market["wins"] / market["bets"] * 100
    ).round(2)

    market.to_csv(
        OUTPUT_DIR / "market_summary.csv",
        index=False,
    )

    grade = (
        df.groupby("grade")
        .agg(
            bets=("grade", "count"),
            wins=("outcome", lambda x: (x == "WIN").sum()),
            units=("profit", "sum"),
        )
        .reset_index()
    )

    grade["win_rate"] = (
        grade["wins"] / grade["bets"] * 100
    ).round(2)

    grade.to_csv(
        OUTPUT_DIR / "grade_summary.csv",
        index=False,
    )

    sportsbook = (
        df.groupby("platform")
        .agg(
            bets=("platform", "count"),
            wins=("outcome", lambda x: (x == "WIN").sum()),
            units=("profit", "sum"),
        )
        .reset_index()
    )

    sportsbook["win_rate"] = (
        sportsbook["wins"] / sportsbook["bets"] * 100
    ).round(2)

    sportsbook.to_csv(
        OUTPUT_DIR / "sportsbook_summary.csv",
        index=False,
    )

    daily = (
        df.groupby("event_date")
        .agg(
            bets=("event_date", "count"),
            units=("profit", "sum"),
        )
        .reset_index()
    )

    daily.to_csv(
        OUTPUT_DIR / "daily_summary.csv",
        index=False,
    )

    print()
    print("========== MLB MODEL REPORT ==========")
    print(f"Record : {wins}-{losses}-{pushes}")
    print(f"Win %  : {pct(wins / total):.2f}%")
    print(f"Units  : {total_profit:.2f}")
    print(f"ROI    : {pct(roi):.2f}%")
    print("======================================")
    print()

    print("Saved:")
    print("overall_summary.csv")
    print("market_summary.csv")
    print("grade_summary.csv")
    print("sportsbook_summary.csv")
    print("daily_summary.csv")


if __name__ == "__main__":
    main()
