import os
from datetime import date
import pandas as pd


CARD_PATH = "outputs/mlb_daily_card.csv"
LOG_PATH = "data/model_results_log.csv"


def log_daily_card():
    if not os.path.exists(CARD_PATH):
        raise FileNotFoundError(f"Missing daily card: {CARD_PATH}")

    card = pd.read_csv(CARD_PATH)

    today = date.today().isoformat()

    card["date"] = today
    card["actual_result"] = pd.NA
    card["win_loss"] = pd.NA
    card["profit_units"] = pd.NA

    columns = [
        "date",
        "platform",
        "player",
        "market",
        "line",
        "projection",
        "edge",
        "pick",
        "grade",
        "actual_result",
        "win_loss",
        "profit_units",
    ]

    for column in columns:
        if column not in card.columns:
            card[column] = pd.NA

    card = card[columns]

    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)

        combined = pd.concat(
            [existing, card],
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset=[
                "date",
                "platform",
                "player",
                "market",
                "line",
            ],
            keep="last"
        )
    else:
        combined = card

    combined.to_csv(LOG_PATH, index=False)

    print(f"Logged {len(card)} picks to {LOG_PATH}")
    print()
    print(card.to_string(index=False))


if __name__ == "__main__":
    log_daily_card()
