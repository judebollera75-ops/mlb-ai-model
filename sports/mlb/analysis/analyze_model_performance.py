"""
Analyze MLB betting model performance.

Reads:
    outputs/history/mlb_bet_results.csv

Produces:
    - Overall performance
    - Market performance
    - Confidence tier performance
    - Probability calibration
    - Sportsbook performance
    - CSV summary tables

Version: 1.0
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "history"
    / "mlb_bet_results.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
)


def load_results() -> pd.DataFrame:
    """Load graded betting history."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{RESULTS_PATH}"
        )

    df = pd.read_csv(RESULTS_PATH)

    df = df[df["grading_status"] == "GRADED"].copy()

    df["profit"] = pd.to_numeric(
        df["profit"],
        errors="coerce",
    )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    df["sportsbook_odds"] = pd.to_numeric(
        df["sportsbook_odds"],
        errors="coerce",
    )

    df["projection"] = pd.to_numeric(
        df["projection"],
        errors="coerce",
    )

    return df


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall model performance."""

    wins = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()

    pushes = len(df) - wins - losses

    total_profit = df["profit"].sum()

    roi = (
        total_profit
        / len(df)
        if len(df)
        else np.nan
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total Bets",
                "Wins",
                "Losses",
                "Pushes",
                "Hit Rate",
                "ROI",
                "Profit (Units)",
                "Average Probability",
            ],
            "Value": [
                len(df),
                wins,
                losses,
                pushes,
                round(wins / (wins + losses), 4)
                if wins + losses
                else np.nan,
                round(roi, 4),
                round(total_profit, 2),
                round(df["probability"].mean(), 4),
            ],
        }
    )

    return summary


def market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Performance by betting market."""

    rows = []

    for market, group in df.groupby("market"):

        wins = (group["outcome"] == "WIN").sum()
        losses = (group["outcome"] == "LOSS").sum()

        hit_rate = (
            wins / (wins + losses)
            if wins + losses
            else np.nan
        )

        roi = (
            group["profit"].sum()
            / len(group)
        )

        rows.append(
            {
                "market": market,
                "bets": len(group),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(hit_rate, 4),
                "roi": round(roi, 4),
                "profit": round(group["profit"].sum(), 2),
                "avg_probability": round(
                    group["probability"].mean(),
                    4,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "roi",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def confidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Performance by confidence tier."""

    rows = []

    for tier, group in df.groupby("confidence_tier"):

        wins = (group["outcome"] == "WIN").sum()
        losses = (group["outcome"] == "LOSS").sum()

        rows.append(
            {
                "confidence": tier,
                "bets": len(group),
                "hit_rate": round(
                    wins / (wins + losses),
                    4,
                )
                if wins + losses
                else np.nan,
                "roi": round(
                    group["profit"].sum()
                    / len(group),
                    4,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "roi",
            ascending=False,
        )
    )


def save_tables(
    overall: pd.DataFrame,
    market: pd.DataFrame,
    confidence: pd.DataFrame,
    calibration: pd.DataFrame,
    sportsbook: pd.DataFrame,
):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall.to_csv(
        OUTPUT_DIRECTORY
        / "overall_summary.csv",
        index=False,
    )

    market.to_csv(
        OUTPUT_DIRECTORY
        / "market_summary.csv",
        index=False,
    )

    confidence.to_csv(
        OUTPUT_DIRECTORY
        / "confidence_summary.csv",
        index=False,
    )
calibration.to_csv(
    OUTPUT_DIRECTORY
    / "probability_calibration.csv",
    index=False,
)

sportsbook.to_csv(
    OUTPUT_DIRECTORY
    / "sportsbook_summary.csv",
    index=False,
)

def main():

    df = load_results()

    overall = overall_summary(df)
    market = market_summary(df)
    confidence = confidence_summary(df)

    save_tables(
        overall,
        market,
        confidence,
    )

    print("\n========================")
    print("OVERALL PERFORMANCE")
    print("========================")
    print(overall)

    print("\n========================")
    print("MARKET PERFORMANCE")
    print("========================")
    print(market)

    print("\n========================")
    print("CONFIDENCE TIERS")
    print("========================")
    print(confidence)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
