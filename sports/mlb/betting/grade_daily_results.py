import os
import pandas as pd


LOG_PATH = "data/model_results_log.csv"
SLATE_DATE = "2026-07-09"

HITTER_RESULTS_PATH = f"data/results/{SLATE_DATE}_hitters.csv"
PITCHER_RESULTS_PATH = f"data/results/{SLATE_DATE}_pitchers.csv"


def normalize_name(value):
    return str(value).strip().lower()


def innings_to_outs(value):
    if pd.isna(value):
        return None

    text = str(value)

    if "." in text:
        whole, fraction = text.split(".", 1)
        fraction = int(fraction[0]) if fraction else 0
    else:
        whole = text
        fraction = 0

    return int(float(whole)) * 3 + fraction


def grade_pick(pick, line, actual):
    if pd.isna(actual):
        return None, None

    if actual == line:
        return "PUSH", 0.0

    pick = str(pick).upper()

    if pick in ["MORE/YES", "OVER", "YES"]:
        won = actual > line
    elif pick in ["LESS/NO", "UNDER", "NO"]:
        won = actual < line
    else:
        return None, None

    return ("WIN", 1.0) if won else ("LOSS", -1.0)


def grade_results():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Missing log: {LOG_PATH}")

    if not os.path.exists(HITTER_RESULTS_PATH):
        raise FileNotFoundError(
            f"Missing hitter results: {HITTER_RESULTS_PATH}"
        )

    if not os.path.exists(PITCHER_RESULTS_PATH):
        raise FileNotFoundError(
            f"Missing pitcher results: {PITCHER_RESULTS_PATH}"
        )

    log = pd.read_csv(LOG_PATH)
    hitters = pd.read_csv(HITTER_RESULTS_PATH)
    pitchers = pd.read_csv(PITCHER_RESULTS_PATH)

    hitters["_normalized_name"] = (
        hitters["player_name"].apply(normalize_name)
    )

    pitchers["_normalized_name"] = (
        pitchers["pitcher_name"].apply(normalize_name)
    )

    # Make result columns object/float-friendly
    log["actual_result"] = pd.to_numeric(
        log["actual_result"],
        errors="coerce"
    )
    log["profit_units"] = pd.to_numeric(
        log["profit_units"],
        errors="coerce"
    )
    log["win_loss"] = log["win_loss"].astype("object")

    rows_to_grade = (
        log["date"].astype(str).eq(SLATE_DATE)
        & log["actual_result"].isna()
    )

    for index in log[rows_to_grade].index:
        player = log.at[index, "player"]
        market = log.at[index, "market"]
        line = pd.to_numeric(
            log.at[index, "line"],
            errors="coerce"
        )

        normalized_player = normalize_name(player)
        actual = None

        if market == "hitter_hits":
            matches = hitters[
                hitters["_normalized_name"] == normalized_player
            ]

            if not matches.empty:
                actual = pd.to_numeric(
                    matches.iloc[0]["hits"],
                    errors="coerce"
                )

        elif market == "hitter_total_bases":
            matches = hitters[
                hitters["_normalized_name"] == normalized_player
            ]

            if not matches.empty:
                actual = pd.to_numeric(
                    matches.iloc[0]["total_bases"],
                    errors="coerce"
                )

        elif market == "pitcher_strikeouts":
            matches = pitchers[
                pitchers["_normalized_name"] == normalized_player
            ]

            if not matches.empty:
                actual = pd.to_numeric(
                    matches.iloc[0]["strikeouts"],
                    errors="coerce"
                )

        elif market == "pitcher_outs":
            matches = pitchers[
                pitchers["_normalized_name"] == normalized_player
            ]

            if not matches.empty:
                actual = innings_to_outs(
                    matches.iloc[0]["innings_pitched"]
                )

        if actual is None or pd.isna(actual):
            print(f"No result found: {player} — {market}")
            continue

        result, units = grade_pick(
            log.at[index, "pick"],
            line,
            actual,
        )

        log.at[index, "actual_result"] = actual
        log.at[index, "win_loss"] = result
        log.at[index, "profit_units"] = units

    log.to_csv(LOG_PATH, index=False)

    slate = log[
        log["date"].astype(str).eq(SLATE_DATE)
    ].copy()

    print("\nGRADED PICKS")
    print(slate.to_string(index=False))

    graded = log[
        log["win_loss"].isin(["WIN", "LOSS", "PUSH"])
    ].copy()

    wins = (graded["win_loss"] == "WIN").sum()
    losses = (graded["win_loss"] == "LOSS").sum()
    pushes = (graded["win_loss"] == "PUSH").sum()

    decisions = wins + losses
    hit_rate = wins / decisions * 100 if decisions else 0

    units = pd.to_numeric(
        graded["profit_units"],
        errors="coerce"
    ).sum()

    print("\nLIFETIME SUMMARY")
    print("Wins:", wins)
    print("Losses:", losses)
    print("Pushes:", pushes)
    print("Hit rate:", round(hit_rate, 1), "%")
    print("Flat-unit score:", round(units, 2))


if __name__ == "__main__":
    grade_results()
