"""
Analyze MLB betting model performance.

Reads:
    outputs/history/mlb_bet_results.csv

Outputs:
    outputs/analysis/

Version: 2.0
"""

from __future__ import annotations

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


NUMERIC_COLUMNS = [
    "profit",
    "probability",
    "projection",
    "sportsbook_odds",
    "line",
]


def load_results() -> pd.DataFrame:
    """
    Load graded betting history.
    """

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results not found:\n{RESULTS_PATH}"
        )

    df = pd.read_csv(RESULTS_PATH)

    if "grading_status" in df.columns:
        df = df[
            df["grading_status"] == "GRADED"
        ].copy()

    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def hit_rate(group: pd.DataFrame) -> float:

    wins = (
        group["outcome"] == "WIN"
    ).sum()

    losses = (
        group["outcome"] == "LOSS"
    ).sum()

    if wins + losses == 0:
        return np.nan

    return wins / (wins + losses)


def roi(group: pd.DataFrame) -> float:

    if len(group) == 0:
        return np.nan

    return (
        group["profit"].sum()
        / len(group)
    )


def total_profit(group: pd.DataFrame) -> float:

    return float(
        group["profit"].sum()
    )


def average_probability(
    group: pd.DataFrame,
) -> float:

    if "probability" not in group.columns:
        return np.nan

    return float(
        group["probability"].mean()
    )


def ensure_output_directory():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def export_table(
    dataframe: pd.DataFrame,
    filename: str,
):

    ensure_output_directory()

    dataframe.to_csv(
        OUTPUT_DIRECTORY / filename,
        index=False,
    )
    def overall_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Overall betting performance.
    """

    wins = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()

    pushes = (
        (df["outcome"] == "PUSH").sum()
        if "PUSH" in df["outcome"].values
        else 0
    )

    total = wins + losses

    summary = pd.DataFrame(
        [
            {
                "bets": total,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": round(
                    wins / total,
                    4,
                )
                if total
                else np.nan,
                "profit": round(
                    df["profit"].sum(),
                    2,
                ),
                "roi": round(
                    roi(df),
                    4,
                ),
                "average_probability": round(
                    average_probability(df),
                    4,
                ),
            }
        ]
    )

    return summary


def market_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance by betting market.
    """

    rows = []

    for market, group in df.groupby("market"):

        wins = (
            group["outcome"] == "WIN"
        ).sum()

        losses = (
            group["outcome"] == "LOSS"
        ).sum()

        rows.append(
            {
                "market": market,
                "bets": len(group),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(
                    hit_rate(group),
                    4,
                ),
                "roi": round(
                    roi(group),
                    4,
                ),
                "profit": round(
                    total_profit(group),
                    2,
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


def confidence_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance by confidence tier.
    """

    if "confidence" not in df.columns:
        return pd.DataFrame()

    rows = []

    for confidence, group in df.groupby(
        "confidence"
    ):

        wins = (
            group["outcome"] == "WIN"
        ).sum()

        losses = (
            group["outcome"] == "LOSS"
        ).sum()

        rows.append(
            {
                "confidence": confidence,
                "bets": len(group),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(
                    hit_rate(group),
                    4,
                ),
                "roi": round(
                    roi(group),
                    4,
                ),
                "profit": round(
                    total_profit(group),
                    2,
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


def probability_bucket_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calibration report.
    """

    if "probability" not in df.columns:
        return pd.DataFrame()

    data = df.dropna(
        subset=["probability"]
    ).copy()

    bins = np.arange(
        0.50,
        1.01,
        0.05,
    )

    labels = [
        f"{int(a*100)}-{int(b*100)}%"
        for a, b in zip(
            bins[:-1],
            bins[1:],
        )
    ]

    data["bucket"] = pd.cut(
        data["probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rows = []

    for bucket, group in data.groupby(
        "bucket",
        observed=False,
    ):

        if len(group) == 0:
            continue

        rows.append(
            {
                "bucket": bucket,
                "bets": len(group),
                "predicted_probability": round(
                    group["probability"].mean(),
                    4,
                ),
                "actual_hit_rate": round(
                    hit_rate(group),
                    4,
                ),
                "difference": round(
                    hit_rate(group)
                    - group["probability"].mean(),
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


def sportsbook_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance by sportsbook.
    """

    if "platform" not in df.columns:
        return pd.DataFrame()

    rows = []

    for platform, group in df.groupby(
        "platform"
    ):

        wins = (
            group["outcome"] == "WIN"
        ).sum()

        losses = (
            group["outcome"] == "LOSS"
        ).sum()

        rows.append(
            {
                "sportsbook": platform,
                "bets": len(group),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(
                    hit_rate(group),
                    4,
                ),
                "roi": round(
                    roi(group),
                    4,
                ),
                "profit": round(
                    total_profit(group),
                    2,
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


def player_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performance by player.
    """

    player_column = None

    for candidate in [
        "player",
        "player_name",
    ]:
        if candidate in df.columns:
            player_column = candidate
            break

    if player_column is None:
        return pd.DataFrame()

    rows = []

    for player, group in df.groupby(
        player_column
    ):

        if len(group) < 3:
            continue

        rows.append(
            {
                "player": player,
                "bets": len(group),
                "hit_rate": round(
                    hit_rate(group),
                    4,
                ),
                "roi": round(
                    roi(group),
                    4,
                ),
                "profit": round(
                    total_profit(group),
                    2,
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
    def save_tables(
    overall: pd.DataFrame,
    market: pd.DataFrame,
    confidence: pd.DataFrame,
    calibration: pd.DataFrame,
    sportsbook: pd.DataFrame,
    players: pd.DataFrame,
) -> None:
    """
    Save all analysis tables.
    """

    export_table(
        overall,
        "overall_summary.csv",
    )

    export_table(
        market,
        "market_summary.csv",
    )

    export_table(
        confidence,
        "confidence_summary.csv",
    )

    export_table(
        calibration,
        "probability_calibration.csv",
    )

    export_table(
        sportsbook,
        "sportsbook_summary.csv",
    )

    export_table(
        players,
        "player_summary.csv",
    )


def print_report(
    overall: pd.DataFrame,
    market: pd.DataFrame,
    confidence: pd.DataFrame,
    calibration: pd.DataFrame,
    sportsbook: pd.DataFrame,
    players: pd.DataFrame,
) -> None:
    """
    Print a readable console report.
    """

    print("\n" + "=" * 70)
    print("MLB MODEL PERFORMANCE REPORT")
    print("=" * 70)

    print("\nOVERALL")
    print(overall.to_string(index=False))

    print("\nTOP MARKETS")
    print(
        market.head(10).to_string(index=False)
    )

    if not confidence.empty:
        print("\nCONFIDENCE TIERS")
        print(
            confidence.to_string(index=False)
        )

    if not calibration.empty:
        print("\nCALIBRATION")
        print(
            calibration.to_string(index=False)
        )

    if not sportsbook.empty:
        print("\nSPORTSBOOK PERFORMANCE")
        print(
            sportsbook.to_string(index=False)
        )

    if not players.empty:
        print("\nBEST PLAYERS")
        print(
            players.head(15).to_string(index=False)
        )

    print("\nRECOMMENDATIONS")

    if not market.empty:

        best_market = market.iloc[0]
        worst_market = market.iloc[-1]

        print(
            f"\nBest Market : {best_market['market']} "
            f"(ROI {best_market['roi']:.3f})"
        )

        print(
            f"Worst Market: {worst_market['market']} "
            f"(ROI {worst_market['roi']:.3f})"
        )

    if not calibration.empty:

        worst_bucket = calibration.iloc[
            calibration["difference"]
            .abs()
            .idxmax()
        ]

        print(
            "\nLargest calibration error:"
        )
        print(
            worst_bucket.to_string()
        )

    print("\nCSV files written to:")
    print(OUTPUT_DIRECTORY)

    print("=" * 70)


def main():

    df = load_results()

    overall = overall_summary(df)

    market = market_summary(df)

    confidence = confidence_summary(df)

    calibration = probability_bucket_summary(
        df
    )

    sportsbook = sportsbook_summary(df)

    players = player_summary(df)

    save_tables(
        overall,
        market,
        confidence,
        calibration,
        sportsbook,
        players,
    )

    print_report(
        overall,
        market,
        confidence,
        calibration,
        sportsbook,
        players,
    )


if __name__ == "__main__":
    main()
