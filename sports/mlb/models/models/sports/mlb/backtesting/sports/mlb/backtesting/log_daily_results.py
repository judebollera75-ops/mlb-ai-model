"""
Automatically logs every daily MLB recommendation into historical results.

Pipeline

Daily Card
        ↓
Log Picks
        ↓
Wait for games to finish
        ↓
Fill Actual Result
        ↓
Append to History
        ↓
Backtester reads history
"""

from pathlib import Path
import pandas as pd
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DAILY_CARD = PROJECT_ROOT / "outputs" / "mlb_daily_card.csv"

HISTORY_FOLDER = PROJECT_ROOT / "outputs" / "history"
HISTORY_FOLDER.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = HISTORY_FOLDER / "mlb_bet_results.csv"


REQUIRED_COLUMNS = [
    "event_date",
    "player",
    "market",
    "direction",
    "line",
    "sportsbook_odds",
    "probability",
    "expected_value",
    "grade",
    "confidence_tier",
    "platform",
    "projection",
    "recommended_bankroll_fraction",
    "actual_result",
    "closing_line",
    "closing_odds",
]


def load_history():

    if HISTORY_FILE.exists():

        history = pd.read_csv(HISTORY_FILE)

    else:

        history = pd.DataFrame(columns=REQUIRED_COLUMNS)

    return history


def load_daily_card():

    if not DAILY_CARD.exists():

        raise FileNotFoundError(
            f"Missing {DAILY_CARD}"
        )

    return pd.read_csv(DAILY_CARD)


def append_new_picks():

    history = load_history()

    today = load_daily_card()

    if "event_date" not in today.columns:

        today["event_date"] = pd.Timestamp.today().date()

    for column in REQUIRED_COLUMNS:

        if column not in today.columns:

            today[column] = pd.NA

    unique_columns = [
        "event_date",
        "player",
        "market",
        "direction",
        "line",
        "platform",
    ]

    history = pd.concat(
        [
            history,
            today[REQUIRED_COLUMNS]
        ],
        ignore_index=True
    )

    history = history.drop_duplicates(
        subset=unique_columns,
        keep="first"
    )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

    print()

    print("="*60)
    print("Historical bet log updated")
    print("="*60)

    print("Total historical bets:", len(history))

    unresolved = history["actual_result"].isna().sum()

    print("Need grading:", unresolved)

    print("Saved:", HISTORY_FILE)

    return history


if __name__ == "__main__":

    append_new_picks()
