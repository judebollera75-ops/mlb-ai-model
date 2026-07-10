import os
from datetime import date
import pandas as pd


CARD_PATH = "outputs/top_mlb_props.csv"
LOG_PATH = "data/model_results_log.csv"


def log_daily_card():
    if not os.path.exists(CARD_PATH):
        raise FileNotFoundError(
            f"Missing clean prop board: {CARD_PATH}"
        )

    card = pd.read_csv(CARD_PATH)

    if card.empty:
        print("No actionable props to log.")
        return

    card["date"] = date.today().isoformat()
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
        "tier",
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

    print(
        f"Logged {len(card)} actionable props "
        f"to {LOG_PATH}"
    )


if __name__ == "__main__":
    log_daily_card()
