import os
from pathlib import Path

import pandas as pd


PITCHERS_DIRECTORY = Path("data/pitchers")
GAME_LOGS_DIRECTORY = Path("data/game_logs")
OUTPUT_PATH = Path("outputs/pitcher_outs_projections.csv")

DEFAULT_PROJECTED_OUTS = 15.0
MIN_PROJECTED_OUTS = 6.0
MAX_PROJECTED_OUTS = 24.0


def baseball_innings_to_outs(value):
    """
    Convert baseball innings notation into outs.

    Examples:
        5.0 -> 15 outs
        5.1 -> 16 outs
        5.2 -> 17 outs
    """
    if pd.isna(value):
        return None

    try:
        innings_text = str(value).strip()
        innings = float(innings_text)
    except (TypeError, ValueError):
        return None

    whole_innings = int(innings)

    if "." in innings_text:
        decimal_part = innings_text.split(".", 1)[1]
    else:
        decimal_part = "0"

    if decimal_part.startswith("1"):
        extra_outs = 1
    elif decimal_part.startswith("2"):
        extra_outs = 2
    else:
        extra_outs = 0

    return (whole_innings * 3) + extra_outs


def find_latest_csv(directory):
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}"
        )

    return csv_files[-1]


def load_todays_pitchers():
    pitchers_path = find_latest_csv(PITCHERS_DIRECTORY)
    pitchers = pd.read_csv(pitchers_path)

    required_columns = {
        "pitcher_id",
        "pitcher_name",
    }

    missing_columns = required_columns - set(pitchers.columns)

    if missing_columns:
        raise KeyError(
            f"{pitchers_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    pitchers["pitcher_id"] = pd.to_numeric(
        pitchers["pitcher_id"],
        errors="coerce",
    )

    pitchers = pitchers.dropna(
        subset=["pitcher_id", "pitcher_name"]
    ).copy()

    pitchers["pitcher_id"] = pitchers["pitcher_id"].astype(int)

    pitchers = pitchers.drop_duplicates(
        subset=["pitcher_id"],
        keep="first",
    )

    print(f"Using pitcher file: {pitchers_path}")

    return pitchers


def load_pitcher_game_logs():
    logs_path = find_latest_csv(GAME_LOGS_DIRECTORY)
    logs = pd.read_csv(logs_path)

    required_columns = {
        "pitcher_id",
        "pitcher_name",
        "game_date",
        "innings",
    }

    missing_columns = required_columns - set(logs.columns)

    if missing_columns:
        raise KeyError(
            f"{logs_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    logs["pitcher_id"] = pd.to_numeric(
        logs["pitcher_id"],
        errors="coerce",
    )

    logs["game_date"] = pd.to_datetime(
        logs["game_date"],
        errors="coerce",
    )

    logs["game_outs"] = logs["innings"].apply(
        baseball_innings_to_outs
    )

    logs = logs.dropna(
        subset=[
            "pitcher_id",
            "game_outs",
        ]
    ).copy()

    logs["pitcher_id"] = logs["pitcher_id"].astype(int)

    logs = logs[
        (logs["game_outs"] >= 0)
        & (logs["game_outs"] <= 27)
    ].copy()

    logs = logs.sort_values(
        ["pitcher_id", "game_date"],
        ascending=[True, False],
    )

    print(f"Using pitcher game-log file: {logs_path}")
    print(f"Loaded {len(logs)} pitcher game-log rows.")

    return logs


def calculate_projection(pitcher_logs):
    if pitcher_logs.empty:
        return {
            "history_games": 0,
            "last3_avg_outs": None,
            "last5_avg_outs": None,
            "season_avg_outs": None,
            "projected_outs": DEFAULT_PROJECTED_OUTS,
            "projection_confidence": "LOW",
        }

    outs = pitcher_logs["game_outs"].dropna()

    if outs.empty:
        return {
            "history_games": 0,
            "last3_avg_outs": None,
            "last5_avg_outs": None,
            "season_avg_outs": None,
            "projected_outs": DEFAULT_PROJECTED_OUTS,
            "projection_confidence": "LOW",
        }

    last3_avg = outs.head(3).mean()
    last5_avg = outs.head(5).mean()
    season_avg = outs.mean()

    projected_outs = (
        last3_avg * 0.50
        + last5_avg * 0.30
        + season_avg * 0.20
    )

    projected_outs = max(
        MIN_PROJECTED_OUTS,
        min(MAX_PROJECTED_OUTS, projected_outs),
    )

    history_games = len(outs)

    if history_games >= 5:
        confidence = "HIGH"
    elif history_games >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "history_games": history_games,
        "last3_avg_outs": round(last3_avg, 2),
        "last5_avg_outs": round(last5_avg, 2),
        "season_avg_outs": round(season_avg, 2),
        "projected_outs": round(projected_outs, 2),
        "projection_confidence": confidence,
    }


def project_pitcher_outs():
    os.makedirs("outputs", exist_ok=True)

    pitchers = load_todays_pitchers()
    logs = load_pitcher_game_logs()

    rows = []

    for _, pitcher in pitchers.iterrows():
        pitcher_id = int(pitcher["pitcher_id"])

        pitcher_logs = logs[
            logs["pitcher_id"] == pitcher_id
        ].copy()

        projection = calculate_projection(pitcher_logs)

        row = pitcher.to_dict()
        row.update(projection)

        rows.append(row)

    projections = pd.DataFrame(rows)

    preferred_columns = [
        "date",
        "game_id",
        "pitcher_id",
        "pitcher_name",
        "team",
        "opponent",
        "side",
        "status",
        "history_games",
        "last3_avg_outs",
        "last5_avg_outs",
        "season_avg_outs",
        "projected_outs",
        "projection_confidence",
    ]

    output_columns = [
        column
        for column in preferred_columns
        if column in projections.columns
    ]

    extra_columns = [
        column
        for column in projections.columns
        if column not in output_columns
    ]

    projections = projections[
        output_columns + extra_columns
    ].copy()

    projections.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved {len(projections)} pitcher-out projections "
        f"to {OUTPUT_PATH}"
    )

    display_columns = [
        column
        for column in [
            "pitcher_name",
            "team",
            "history_games",
            "last3_avg_outs",
            "last5_avg_outs",
            "season_avg_outs",
            "projected_outs",
            "projection_confidence",
        ]
        if column in projections.columns
    ]

    if not projections.empty:
        print()
        print(
            projections[display_columns]
            .sort_values(
                "projected_outs",
                ascending=False,
            )
            .to_string(index=False)
        )

    return projections


if __name__ == "__main__":
    project_pitcher_outs()
