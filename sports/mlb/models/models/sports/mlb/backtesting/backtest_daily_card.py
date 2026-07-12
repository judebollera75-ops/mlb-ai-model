"""Backtest graded MLB daily-card recommendations.

This script evaluates historical recommendations after results have been
recorded. It calculates hit rate, ROI, calibration, drawdown, Sharpe ratio,
and performance by market, platform, grade, and confidence tier.

Expected input:
    outputs/history/mlb_bet_results.csv

Expected columns:
    event_date
    player
    market
    direction
    line
    sportsbook_odds
    probability
    expected_value
    grade
    confidence_tier
    platform
    actual_result

Optional columns:
    closing_line
    closing_odds
    recommended_bankroll_fraction
    stake

Outputs:
    outputs/backtesting/mlb_backtest_summary.csv
    outputs/backtesting/mlb_backtest_by_market.csv
    outputs/backtesting/mlb_backtest_by_platform.csv
    outputs/backtesting/mlb_backtest_by_grade.csv
    outputs/backtesting/mlb_backtest_by_confidence.csv
    outputs/backtesting/mlb_backtest_by_probability_bucket.csv
    outputs/backtesting/mlb_backtest_bankroll_curve.csv
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "history"
    / "mlb_bet_results.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "mlb_backtest_summary.csv"
)

BY_MARKET_PATH = (
    OUTPUT_DIRECTORY
    / "mlb_backtest_by_market.csv"
)

BY_PLATFORM_PATH = (
    OUTPUT_DIRECTORY
    / "mlb_backtest_by_platform.csv"
)

BY_GRADE_PATH = (
    OUTPUT_DIRECTORY
    / "mlb_backtest_by_grade.csv"
)

BY_CONFIDENCE_PATH = (
    OUTPUT_DIRECTORY
    / "mlb_backtest_by_confidence.csv"
)

BY_PROBABILITY_BUCKET_PATH = (
    OUTPUT_DIRECTORY
    / "mlb_backtest_by_probability_bucket.csv"
)

BANKROLL_CURVE_PATH = (
    OUTPUT_DIRECTORY
    / "mlb_backtest_bankroll_curve.csv"
)

DEFAULT_STAKE = 1.0
MINIMUM_SAMPLE_FOR_REPORTING = 10

REQUIRED_COLUMNS = {
    "event_date",
    "player",
    "market",
    "direction",
    "line",
    "sportsbook_odds",
    "probability",
    "platform",
    "actual_result",
}


def american_odds_profit_per_unit(odds: Any) -> float:
    """Return net profit on a one-unit winning wager."""
    try:
        numeric_odds = float(odds)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(numeric_odds) or numeric_odds == 0:
        return float("nan")

    if numeric_odds > 0:
        return numeric_odds / 100.0

    return 100.0 / abs(numeric_odds)


def normalize_direction(value: Any) -> str:
    """Normalize betting direction names."""
    cleaned = str(value).strip().casefold()

    if cleaned in {"over", "more", "yes", "higher"}:
        return "Over"

    if cleaned in {"under", "less", "no", "lower"}:
        return "Under"

    return ""


def grade_outcome(
    direction: str,
    line: float,
    actual_result: float,
) -> str:
    """Return WIN, LOSS, or PUSH."""
    if not np.isfinite(line) or not np.isfinite(actual_result):
        return "UNRESOLVED"

    if direction == "Over":
        if actual_result > line:
            return "WIN"
        if actual_result < line:
            return "LOSS"
        return "PUSH"

    if direction == "Under":
        if actual_result < line:
            return "WIN"
        if actual_result > line:
            return "LOSS"
        return "PUSH"

    return "UNRESOLVED"


def calculate_profit(
    outcome: str,
    sportsbook_odds: float,
    stake: float,
) -> float:
    """Calculate realized profit for one wager."""
    if outcome == "PUSH":
        return 0.0

    if outcome == "LOSS":
        return -stake

    if outcome != "WIN":
        return float("nan")

    profit_multiple = american_odds_profit_per_unit(
        sportsbook_odds
    )

    if not np.isfinite(profit_multiple):
        return float("nan")

    return stake * profit_multiple


def calculate_clv(row: pd.Series) -> float:
    """Calculate simple line-based CLV from the bettor's perspective."""
    if "closing_line" not in row.index:
        return float("nan")

    placed_line = pd.to_numeric(
        row.get("line"),
        errors="coerce",
    )

    closing_line = pd.to_numeric(
        row.get("closing_line"),
        errors="coerce",
    )

    if pd.isna(placed_line) or pd.isna(closing_line):
        return float("nan")

    direction = row.get("direction")

    if direction == "Over":
        return float(closing_line - placed_line)

    if direction == "Under":
        return float(placed_line - closing_line)

    return float("nan")


def calculate_brier_score(
    probabilities: pd.Series,
    outcomes: pd.Series,
) -> float:
    """Calculate Brier score for binary settled bets."""
    mask = (
        probabilities.notna()
        & outcomes.notna()
    )

    if not mask.any():
        return float("nan")

    return float(
        np.mean(
            (
                probabilities.loc[mask].astype(float)
                - outcomes.loc[mask].astype(float)
            )
            ** 2
        )
    )


def calculate_log_loss(
    probabilities: pd.Series,
    outcomes: pd.Series,
) -> float:
    """Calculate binary log loss with clipping."""
    mask = (
        probabilities.notna()
        & outcomes.notna()
    )

    if not mask.any():
        return float("nan")

    probability_values = np.clip(
        probabilities.loc[mask].astype(float),
        1e-6,
        1 - 1e-6,
    )

    outcome_values = outcomes.loc[mask].astype(float)

    return float(
        -np.mean(
            outcome_values * np.log(probability_values)
            + (1.0 - outcome_values)
            * np.log(1.0 - probability_values)
        )
    )


def calculate_max_drawdown(
    bankroll_values: pd.Series,
) -> float:
    """Calculate maximum percentage drawdown."""
    if bankroll_values.empty:
        return float("nan")

    running_maximum = bankroll_values.cummax()

    drawdowns = (
        bankroll_values - running_maximum
    ) / running_maximum.replace(0, np.nan)

    return float(drawdowns.min())


def calculate_sharpe_ratio(
    profits: pd.Series,
) -> float:
    """Calculate a per-bet Sharpe-style ratio."""
    profits = pd.to_numeric(
        profits,
        errors="coerce",
    ).dropna()

    if len(profits) < 2:
        return float("nan")

    standard_deviation = float(
        profits.std(ddof=1)
    )

    if standard_deviation == 0:
        return float("nan")

    return float(
        profits.mean()
        / standard_deviation
        * math.sqrt(len(profits))
    )


def prepare_results() -> pd.DataFrame:
    """Load and grade historical bet results."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Historical bet-results file was not found: {INPUT_PATH}"
        )

    frame = pd.read_csv(INPUT_PATH)

    missing_columns = (
        REQUIRED_COLUMNS - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Historical results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame["event_date"] = pd.to_datetime(
        frame["event_date"],
        errors="coerce",
    )

    frame["direction"] = frame[
        "direction"
    ].apply(normalize_direction)

    for column in [
        "line",
        "sportsbook_odds",
        "probability",
        "expected_value",
        "actual_result",
        "recommended_bankroll_fraction",
        "stake",
        "closing_line",
        "closing_odds",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    if "stake" not in frame.columns:
        frame["stake"] = DEFAULT_STAKE

    frame["stake"] = frame[
        "stake"
    ].fillna(DEFAULT_STAKE)

    frame = frame.dropna(
        subset=[
            "event_date",
            "player",
            "market",
            "direction",
            "line",
            "sportsbook_odds",
            "actual_result",
        ]
    ).copy()

    frame = frame.loc[
        frame["direction"].isin(
            {"Over", "Under"}
        )
    ].copy()

    frame["outcome"] = frame.apply(
        lambda row: grade_outcome(
            row["direction"],
            float(row["line"]),
            float(row["actual_result"]),
        ),
        axis=1,
    )

    frame = frame.loc[
        frame["outcome"].isin(
            {"WIN", "LOSS", "PUSH"}
        )
    ].copy()

    frame["profit"] = frame.apply(
        lambda row: calculate_profit(
            row["outcome"],
            row["sportsbook_odds"],
            row["stake"],
        ),
        axis=1,
    )

    frame["binary_outcome"] = frame[
        "outcome"
    ].map(
        {
            "WIN": 1.0,
            "LOSS": 0.0,
            "PUSH": np.nan,
        }
    )

    frame["clv"] = frame.apply(
        calculate_clv,
        axis=1,
    )

    frame = frame.sort_values(
        [
            "event_date",
            "player",
            "market",
        ]
    ).reset_index(drop=True)

    return frame


def summarize_group(
    group: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate core backtest metrics for one group."""
    settled = group.loc[
        group["outcome"].isin(
            {"WIN", "LOSS"}
        )
    ].copy()

    total_bets = len(group)
    settled_bets = len(settled)
    wins = int(
        settled["outcome"].eq("WIN").sum()
    )
    losses = int(
        settled["outcome"].eq("LOSS").sum()
    )
    pushes = int(
        group["outcome"].eq("PUSH").sum()
    )

    total_staked = float(
        group["stake"].sum()
    )

    total_profit = float(
        group["profit"].sum()
    )

    hit_rate = (
        wins / settled_bets
        if settled_bets > 0
        else float("nan")
    )

    roi = (
        total_profit / total_staked
        if total_staked > 0
        else float("nan")
    )

    average_probability = float(
        settled["probability"].mean()
    ) if (
        "probability" in settled.columns
        and not settled.empty
    ) else float("nan")

    average_expected_value = float(
        settled["expected_value"].mean()
    ) if (
        "expected_value" in settled.columns
        and not settled.empty
    ) else float("nan")

    average_clv = float(
        group["clv"].mean()
    ) if "clv" in group.columns else float("nan")

    brier_score = calculate_brier_score(
        settled.get(
            "probability",
            pd.Series(dtype=float),
        ),
        settled.get(
            "binary_outcome",
            pd.Series(dtype=float),
        ),
    )

    log_loss = calculate_log_loss(
        settled.get(
            "probability",
            pd.Series(dtype=float),
        ),
        settled.get(
            "binary_outcome",
            pd.Series(dtype=float),
        ),
    )

    return {
        "total_bets": total_bets,
        "settled_bets": settled_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": hit_rate,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "average_probability": average_probability,
        "average_expected_value": average_expected_value,
        "average_clv": average_clv,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "sharpe_ratio": calculate_sharpe_ratio(
            settled["profit"]
        ),
    }


def build_group_report(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Build metrics by one categorical field."""
    if group_column not in frame.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for group_value, group in frame.groupby(
        group_column,
        dropna=False,
    ):
        metrics = summarize_group(group)
        metrics[group_column] = group_value
        rows.append(metrics)

    report = pd.DataFrame(rows)

    if report.empty:
        return report

    report = report.loc[
        report["total_bets"]
        >= MINIMUM_SAMPLE_FOR_REPORTING
    ].copy()

    return report.sort_values(
        ["roi", "total_bets"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_probability_bucket_report(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Measure calibration and performance by probability bucket."""
    settled = frame.loc[
        frame["outcome"].isin(
            {"WIN", "LOSS"}
        )
        & frame["probability"].notna()
    ].copy()

    if settled.empty:
        return pd.DataFrame()

    bucket_edges = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        1.01,
    ]

    bucket_labels = [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80%+",
    ]

    settled["probability_bucket"] = pd.cut(
        settled["probability"],
        bins=bucket_edges,
        labels=bucket_labels,
        right=False,
        include_lowest=True,
    )

    report = build_group_report(
        settled,
        "probability_bucket",
    )

    if report.empty:
        return report

    report["calibration_gap"] = (
        report["hit_rate"]
        - report["average_probability"]
    )

    return report


def build_bankroll_curve(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build cumulative bankroll and drawdown history."""
    curve = frame[
        [
            "event_date",
            "player",
            "market",
            "platform",
            "outcome",
            "stake",
            "profit",
        ]
    ].copy()

    curve["cumulative_profit"] = curve[
        "profit"
    ].cumsum()

    curve["bankroll"] = (
        curve["cumulative_profit"] + 100.0
    )

    curve["running_peak"] = curve[
        "bankroll"
    ].cummax()

    curve["drawdown"] = (
        curve["bankroll"]
        - curve["running_peak"]
    ) / curve["running_peak"].replace(
        0,
        np.nan,
    )

    return curve


def round_report(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Round report metrics for CSV readability."""
    rounded = frame.copy()

    for column in [
        "hit_rate",
        "roi",
        "average_probability",
        "average_expected_value",
        "average_clv",
        "brier_score",
        "log_loss",
        "sharpe_ratio",
        "calibration_gap",
        "total_staked",
        "total_profit",
    ]:
        if column in rounded.columns:
            rounded[column] = pd.to_numeric(
                rounded[column],
                errors="coerce",
            ).round(4)

    return rounded


def backtest_daily_card() -> pd.DataFrame:
    """Run the complete MLB historical-card backtest."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = prepare_results()

    if frame.empty:
        raise RuntimeError(
            "No settled MLB recommendations were available."
        )

    summary_metrics = summarize_group(frame)

    bankroll_curve = build_bankroll_curve(frame)

    summary_metrics["maximum_drawdown"] = (
        calculate_max_drawdown(
            bankroll_curve["bankroll"]
        )
    )

    summary_metrics["start_date"] = (
        frame["event_date"].min().date().isoformat()
    )

    summary_metrics["end_date"] = (
        frame["event_date"].max().date().isoformat()
    )

    summary = pd.DataFrame(
        [summary_metrics]
    )

    by_market = build_group_report(
        frame,
        "market",
    )

    by_platform = build_group_report(
        frame,
        "platform",
    )

    by_grade = build_group_report(
        frame,
        "grade",
    )

    by_confidence = build_group_report(
        frame,
        "confidence_tier",
    )

    by_probability_bucket = (
        build_probability_bucket_report(frame)
    )

    round_report(summary).to_csv(
        SUMMARY_PATH,
        index=False,
    )

    round_report(by_market).to_csv(
        BY_MARKET_PATH,
        index=False,
    )

    round_report(by_platform).to_csv(
        BY_PLATFORM_PATH,
        index=False,
    )

    round_report(by_grade).to_csv(
        BY_GRADE_PATH,
        index=False,
    )

    round_report(by_confidence).to_csv(
        BY_CONFIDENCE_PATH,
        index=False,
    )

    round_report(
        by_probability_bucket
    ).to_csv(
        BY_PROBABILITY_BUCKET_PATH,
        index=False,
    )

    bankroll_curve.to_csv(
        BANKROLL_CURVE_PATH,
        index=False,
    )

    print("=" * 72)
    print("MLB BACKTEST COMPLETE")
    print("=" * 72)
    print(
        round_report(summary)
        .to_string(index=False)
    )

    print("\nPerformance by market:")
    print(
        round_report(by_market)
        .to_string(index=False)
    )

    print("\nCalibration by probability bucket:")
    print(
        round_report(
            by_probability_bucket
        ).to_string(index=False)
    )

    print(f"\nSaved reports to: {OUTPUT_DIRECTORY}")

    return summary


if __name__ == "__main__":
    backtest_daily_card()
