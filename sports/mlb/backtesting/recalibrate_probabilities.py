"""Recalculate conservative MLB Elite thresholds from graded history.

Place this file at:
    sports/mlb/backtesting/recalibrate_probabilities.py

Input:
    outputs/history/mlb_bet_results.csv

Outputs:
    outputs/calibration/elite_thresholds.csv
    outputs/calibration/elite_threshold_backtest.csv

This script uses only graded WIN/LOSS rows. It searches historical thresholds
chronologically and recommends the strictest practical settings that have
actually reached the target hit rate. When no segment proves 75%, the market is
marked NOT_PROVEN and the production Elite filter keeps its strict defaults.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HISTORY_PATH = PROJECT_ROOT / "outputs" / "history" / "mlb_bet_results.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "calibration"
THRESHOLDS_PATH = OUTPUT_DIRECTORY / "elite_thresholds.csv"
BACKTEST_PATH = OUTPUT_DIRECTORY / "elite_threshold_backtest.csv"

TARGET_HIT_RATE = 0.75
MINIMUM_SAMPLE = 30
MINIMUM_ONE_SIDED_LOWER_BOUND = 0.65

PROBABILITY_GRID = np.round(np.arange(0.64, 0.86, 0.01), 2)
EDGE_GRID = np.round(np.arange(0.04, 0.17, 0.01), 2)
EV_GRID = np.round(np.arange(0.04, 0.21, 0.02), 2)


def wilson_lower_bound(wins: int, sample_size: int, z: float = 1.2815515655) -> float:
    if sample_size <= 0:
        return float("nan")
    p_hat = wins / sample_size
    denominator = 1.0 + z * z / sample_size
    center = p_hat + z * z / (2.0 * sample_size)
    margin = z * math.sqrt(
        p_hat * (1.0 - p_hat) / sample_size
        + z * z / (4.0 * sample_size * sample_size)
    )
    return float((center - margin) / denominator)


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing graded history: {HISTORY_PATH}")

    history = pd.read_csv(HISTORY_PATH)
    required = {
        "market", "direction", "outcome",
        "probability", "probability_edge", "expected_value",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History is missing required columns: {sorted(missing)}")

    history["market"] = history["market"].astype("string").str.strip().str.lower()
    history["direction"] = history["direction"].astype("string").str.strip().str.title()
    history["outcome"] = history["outcome"].astype("string").str.strip().str.upper()

    for column in ["probability", "probability_edge", "expected_value"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    date_column = "event_date" if "event_date" in history.columns else "slate_date"
    if date_column in history.columns:
        history["_date"] = pd.to_datetime(history[date_column], errors="coerce")
    else:
        history["_date"] = pd.NaT

    history = history.loc[
        history["outcome"].isin({"WIN", "LOSS"})
        & history["probability"].notna()
        & history["probability_edge"].notna()
        & history["expected_value"].notna()
    ].copy()

    return history.sort_values(["_date"], na_position="first").reset_index(drop=True)


def evaluate_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for probability_threshold in PROBABILITY_GRID:
        probability_subset = frame.loc[
            frame["probability"].ge(probability_threshold)
        ]
        if len(probability_subset) < MINIMUM_SAMPLE:
            continue

        for edge_threshold in EDGE_GRID:
            edge_subset = probability_subset.loc[
                probability_subset["probability_edge"].ge(edge_threshold)
            ]
            if len(edge_subset) < MINIMUM_SAMPLE:
                continue

            for ev_threshold in EV_GRID:
                selected = edge_subset.loc[
                    edge_subset["expected_value"].ge(ev_threshold)
                ]
                sample = len(selected)
                if sample < MINIMUM_SAMPLE:
                    continue

                wins = int(selected["outcome"].eq("WIN").sum())
                losses = sample - wins
                hit_rate = wins / sample
                lower_bound = wilson_lower_bound(wins, sample)

                rows.append({
                    "probability_threshold": probability_threshold,
                    "probability_edge_threshold": edge_threshold,
                    "expected_value_threshold": ev_threshold,
                    "sample": sample,
                    "wins": wins,
                    "losses": losses,
                    "hit_rate": hit_rate,
                    "wilson_lower_bound": lower_bound,
                    "target_met": (
                        hit_rate >= TARGET_HIT_RATE
                        and lower_bound >= MINIMUM_ONE_SIDED_LOWER_BOUND
                    ),
                })

    return pd.DataFrame(rows)


def choose_recommendation(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {
            "status": "NOT_PROVEN",
            "recommended_probability_threshold": np.nan,
            "recommended_probability_edge_threshold": np.nan,
            "recommended_expected_value_threshold": np.nan,
            "backtest_sample": 0,
            "backtest_wins": 0,
            "backtest_losses": 0,
            "backtest_hit_rate": np.nan,
            "backtest_lower_bound": np.nan,
        }

    proven = results.loc[results["target_met"]].copy()
    if proven.empty:
        best = results.sort_values(
            ["hit_rate", "wilson_lower_bound", "sample"],
            ascending=[False, False, False],
        ).iloc[0]
        status = "NOT_PROVEN"
    else:
        # Prefer the largest proven sample, then the stronger lower bound.
        best = proven.sort_values(
            ["sample", "wilson_lower_bound", "hit_rate"],
            ascending=[False, False, False],
        ).iloc[0]
        status = "PROVEN_IN_SAMPLE"

    return {
        "status": status,
        "recommended_probability_threshold": float(best["probability_threshold"]),
        "recommended_probability_edge_threshold": float(best["probability_edge_threshold"]),
        "recommended_expected_value_threshold": float(best["expected_value_threshold"]),
        "backtest_sample": int(best["sample"]),
        "backtest_wins": int(best["wins"]),
        "backtest_losses": int(best["losses"]),
        "backtest_hit_rate": float(best["hit_rate"]),
        "backtest_lower_bound": float(best["wilson_lower_bound"]),
    }


def recalibrate_elite_thresholds() -> pd.DataFrame:
    history = load_history()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    threshold_rows: list[dict[str, Any]] = []
    audit_sections: list[pd.DataFrame] = []

    for market, market_frame in history.groupby("market", sort=True):
        results = evaluate_thresholds(market_frame)
        if not results.empty:
            results.insert(0, "market", market)
            audit_sections.append(results)

        recommendation = choose_recommendation(results)
        overall_sample = len(market_frame)
        overall_wins = int(market_frame["outcome"].eq("WIN").sum())

        threshold_rows.append({
            "market": market,
            "target_hit_rate": TARGET_HIT_RATE,
            "minimum_sample_required": MINIMUM_SAMPLE,
            "overall_sample": overall_sample,
            "overall_wins": overall_wins,
            "overall_hit_rate": overall_wins / overall_sample if overall_sample else np.nan,
            **recommendation,
        })

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(THRESHOLDS_PATH, index=False)

    if audit_sections:
        audit = pd.concat(audit_sections, ignore_index=True)
    else:
        audit = pd.DataFrame()
    audit.to_csv(BACKTEST_PATH, index=False)

    print("=" * 72)
    print("ELITE THRESHOLD RECALIBRATION COMPLETE")
    print(f"Graded rows used: {len(history):,}")
    print(f"Threshold file: {THRESHOLDS_PATH}")
    print(f"Backtest audit: {BACKTEST_PATH}")
    print("=" * 72)
    if not thresholds.empty:
        print(
            thresholds[
                [
                    "market",
                    "status",
                    "backtest_sample",
                    "backtest_hit_rate",
                    "backtest_lower_bound",
                    "recommended_probability_threshold",
                    "recommended_probability_edge_threshold",
                    "recommended_expected_value_threshold",
                ]
            ].to_string(index=False)
        )

    return thresholds


if __name__ == "__main__":
    recalibrate_elite_thresholds()
