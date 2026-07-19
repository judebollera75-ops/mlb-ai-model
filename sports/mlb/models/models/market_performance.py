"""Analyze historical MLB betting performance by market.

Input:
    outputs/history/mlb_bet_results.csv

Output:
    outputs/market_performance.csv

The output summarizes graded results for each market and creates a conservative
market weight that can later be used by the daily-card confidence system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "history"
    / "mlb_bet_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "market_performance.csv"
)

MINIMUM_SAMPLE_FOR_FULL_WEIGHT = 100
MINIMUM_WEIGHT = 0.85
MAXIMUM_WEIGHT = 1.15

OUTPUT_COLUMNS = [
    "market",
    "bets",
    "wins",
    "losses",
    "pushes",
    "hit_rate",
    "roi",
    "average_profit_per_bet",
    "units",
    "sample_reliability",
    "market_weight",
]


def normalize_outcome(value: Any) -> str:
    """Normalize different result labels into WIN, LOSS, PUSH, or empty."""
    if value is None or pd.isna(value):
        return ""

    outcome = str(value).strip().upper()

    if outcome in {
        "WIN",
        "W",
        "WON",
        "SUCCESS",
        "HIT",
    }:
        return "WIN"

    if outcome in {
        "LOSS",
        "L",
        "LOST",
        "FAIL",
        "MISS",
    }:
        return "LOSS"

    if outcome in {
        "PUSH",
        "P",
        "TIE",
        "VOID",
    }:
        return "PUSH"

    return ""


def american_odds_profit(
    odds: Any,
    stake: Any = 1.0,
) -> float:
    """Return profit for a winning American-odds wager."""
    try:
        numeric_odds = float(odds)
        numeric_stake = float(stake)
    except (TypeError, ValueError):
        return float("nan")

    if (
        not np.isfinite(numeric_odds)
        or not np.isfinite(numeric_stake)
        or numeric_odds == 0
        or numeric_stake < 0
    ):
        return float("nan")

    if numeric_odds > 0:
        return numeric_stake * numeric_odds / 100.0

    return numeric_stake * 100.0 / abs(numeric_odds)


def calculate_profit(row: pd.Series) -> float:
    """Use saved profit when available, otherwise calculate it."""
    saved_profit = pd.to_numeric(
        row.get("profit"),
        errors="coerce",
    )

    if pd.notna(saved_profit):
        return float(saved_profit)

    outcome = row.get("normalized_outcome", "")

    stake = pd.to_numeric(
        row.get("stake", 1.0),
        errors="coerce",
    )

    if pd.isna(stake) or float(stake) <= 0:
        stake = 1.0

    stake = float(stake)

    if outcome == "LOSS":
        return -stake

    if outcome == "PUSH":
        return 0.0

    if outcome == "WIN":
        winning_profit = american_odds_profit(
            row.get("sportsbook_odds"),
            stake,
        )

        if np.isfinite(winning_profit):
            return float(winning_profit)

        # Use even-money profit when historical odds are unavailable.
        return stake

    return float("nan")


def calculate_market_weight(
    hit_rate: float,
    roi: float,
    sample_size: int,
) -> tuple[float, float]:
    """Create a conservative sample-adjusted market weight."""
    if (
        not np.isfinite(hit_rate)
        or not np.isfinite(roi)
        or sample_size <= 0
    ):
        return 0.0, 1.0

    sample_reliability = min(
        1.0,
        sample_size / MINIMUM_SAMPLE_FOR_FULL_WEIGHT,
    )

    # Compare performance to a neutral 50% hit rate and zero ROI.
    hit_rate_component = (hit_rate - 0.50) * 0.80
    roi_component = np.clip(roi, -0.30, 0.30) * 0.40

    raw_adjustment = (
        hit_rate_component
        + roi_component
    )

    adjusted_weight = (
        1.0
        + raw_adjustment * sample_reliability
    )

    adjusted_weight = float(
        np.clip(
            adjusted_weight,
            MINIMUM_WEIGHT,
            MAXIMUM_WEIGHT,
        )
    )

    return (
        float(sample_reliability),
        adjusted_weight,
    )


def load_history() -> pd.DataFrame:
    """Load betting history and validate required columns."""
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Bet history was not found: {HISTORY_PATH}"
        )

    try:
        history = pd.read_csv(HISTORY_PATH)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(
            f"Could not read bet history: {HISTORY_PATH}"
        ) from exc

    required_columns = {
        "market",
        "outcome",
    }

    missing_columns = (
        required_columns - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "Bet history is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return history


def build_market_performance() -> pd.DataFrame:
    """Build and save historical performance by market."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = load_history()

    history["market"] = (
        history["market"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    history["normalized_outcome"] = (
        history["outcome"]
        .apply(normalize_outcome)
    )

    # Only completed bets are included.
    graded = history.loc[
        history["market"].ne("")
        & history["normalized_outcome"].isin(
            {"WIN", "LOSS", "PUSH"}
        )
    ].copy()

    if "sportsbook_odds" not in graded.columns:
        graded["sportsbook_odds"] = np.nan

    graded["sportsbook_odds"] = pd.to_numeric(
        graded["sportsbook_odds"],
        errors="coerce",
    )

    # Exclude extreme longshot prices from market-weight calculations.
    # These bets remain in the full history but do not dominate the
    # confidence adjustment.
    graded = graded.loc[
        graded["sportsbook_odds"].isna()
        | graded["sportsbook_odds"].between(-300, 300)
    ].copy()

    if graded.empty:
        empty_output = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

        empty_output.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print("=" * 72)
        print("MARKET PERFORMANCE COMPLETE")
        print("No graded historical bets were available.")
        print(f"Saved empty report to: {OUTPUT_PATH}")
        print("=" * 72)

        return empty_output

    if "stake" not in graded.columns:
        graded["stake"] = 1.0

    graded["stake"] = pd.to_numeric(
        graded["stake"],
        errors="coerce",
    ).fillna(1.0)

    graded.loc[
        graded["stake"].le(0),
        "stake",
    ] = 1.0

    graded["calculated_profit"] = graded.apply(
        calculate_profit,
        axis=1,
    )

    rows: list[dict[str, Any]] = []

    for market, group in graded.groupby(
        "market",
        sort=True,
    ):
        wins = int(
            group["normalized_outcome"]
            .eq("WIN")
            .sum()
        )

        losses = int(
            group["normalized_outcome"]
            .eq("LOSS")
            .sum()
        )

        pushes = int(
            group["normalized_outcome"]
            .eq("PUSH")
            .sum()
        )

        bets = int(len(group))
        decisions = wins + losses

        hit_rate = (
            wins / decisions
            if decisions > 0
            else float("nan")
        )

        units = float(
            group["calculated_profit"]
            .fillna(0.0)
            .sum()
        )

        total_staked = float(
            group["stake"].sum()
        )

        roi = (
            units / total_staked
            if total_staked > 0
            else float("nan")
        )

        average_profit_per_bet = (
            units / bets
            if bets > 0
            else float("nan")
        )

        (
            sample_reliability,
            market_weight,
        ) = calculate_market_weight(
            hit_rate=hit_rate,
            roi=roi,
            sample_size=decisions,
        )

        rows.append(
            {
                "market": market,
                "bets": bets,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": hit_rate,
                "roi": roi,
                "average_profit_per_bet": (
                    average_profit_per_bet
                ),
                "units": units,
                "sample_reliability": (
                    sample_reliability
                ),
                "market_weight": market_weight,
            }
        )

    output = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    output = output.sort_values(
        [
            "market_weight",
            "bets",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    for column in [
        "hit_rate",
        "roi",
        "average_profit_per_bet",
        "units",
        "sample_reliability",
        "market_weight",
    ]:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).round(4)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 72)
    print("MARKET PERFORMANCE COMPLETE")
    print(f"Graded bets analyzed: {len(graded):,}")
    print(f"Markets analyzed: {len(output):,}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 72)

    print(
        output.to_string(
            index=False
        )
    )

    return output


if __name__ == "__main__":
    build_market_performance()
