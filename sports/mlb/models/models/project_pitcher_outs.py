import os
import re
from pathlib import Path

import pandas as pd


PITCHERS_DIRECTORY = Path("data/pitchers")
PITCHER_LOGS_PATH = Path("data/pitcher_game_logs/2026.csv")
OUTPUT_PATH = Path("outputs/pitcher_outs_projections.csv")

DEFAULT_PROJECTED_OUTS = 15.0
MIN_PROJECTED_OUTS = 6.0
MAX_PROJECTED_OUTS = 27.0


def normalize_player_name(value):
    """Normalize names so pitcher files and game logs match reliably."""
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value)

    return value


def baseball_innings_to_outs(value):
    """
    Convert baseball innings notation into outs.

    Examples:
        5.0 -> 15 outs
        5.1 -> 16 outs
        5.2 -> 17 outs

    This should only be used on individual game-log innings values.
    """
    if pd.isna(value):
        return None

    try:
        innings = float(value)
    except (TypeError, ValueError):
        return None

    whole_innings = int(innings)
    decimal = round(innings - whole_innings, 1)

    if decimal == 0.1:
        extra_outs = 1
    elif decimal == 0.2:
        extra_outs = 2
    elif decimal == 0.0:
        extra_outs = 0
    else:
        # Handles ordinary decimal values safely.
        return innings * 3

    return (whole_innings * 3) + extra_outs


def find_latest_pitchers_file():
    """Find the newest daily pitcher file."""
    if not PITCHERS_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Pitcher directory does not exist: {PITCHERS_DIRECTORY}"
        )

    pitcher_files = sorted(PITCHERS_DIRECTORY.glob("*.csv"))

    if not pitcher_files:
        raise FileNotFoundError(
            f"No pitcher CSV files found in {PITCHERS_DIRECTORY}"
        )

    return pitcher_files[-1]


def find_column(dataframe, possible_names):
    """Return the first matching column name."""
    for column in possible_names:
        if column in dataframe.columns:
            return column

    return None


def load_todays_pitchers():
    """Load only the pitchers listed in the newest daily pitcher file."""
    pitchers_path = find_latest_pitchers_file()
    pitchers = pd.read_csv(pitchers_path)

    print(f"Using pitcher file: {pitchers_path}")

    pitcher_name_column = find_column(
        pitchers,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
            "name",
        ],
    )

    if pitcher_name_column is None:
        raise KeyError(
            "Could not find a pitcher-name column in "
            f"{pitchers_path}. Columns: {list(pitchers.columns)}"
        )

    if pitcher_name_column != "pitcher_name":
        pitchers = pitchers.rename(
            columns={pitcher_name_column: "pitcher_name"}
        )

    pitchers["pitcher_key"] = pitchers["pitcher_name"].apply(
        normalize_player_name
    )

    pitchers = pitchers.drop_duplicates(
        subset=["pitcher_key"],
        keep="first",
    ).copy()

    return pitchers


def load_pitcher_game_logs():
    """Load pitcher game logs and calculate outs for each appearance."""
    if not PITCHER_LOGS_PATH.exists():
        print(
            f"WARNING: {PITCHER_LOGS_PATH} was not found. "
            "Default projections will be used."
        )

        return pd.DataFrame(
            columns=[
                "pitcher_key",
                "game_date",
                "game_outs",
            ]
        )

    logs = pd.read_csv(PITCHER_LOGS_PATH)

    pitcher_name_column = find_column(
        logs,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
            "name",
        ],
    )

    innings_column = find_column(
        logs,
        [
            "innings_pitched",
            "innings",
            "ip",
            "IP",
        ],
    )

    date_column = find_column(
        logs,
        [
            "date",
            "game_date",
            "gameDate",
        ],
    )

    if pitcher_name_column is None:
        raise KeyError(
            "Could not find a pitcher-name column in "
            f"{PITCHER_LOGS_PATH}. Columns: {list(logs.columns)}"
        )

    if innings_column is None:
        raise KeyError(
            "Could not find an innings-pitched column in "
            f"{PITCHER_LOGS_PATH}. Columns: {list(logs.columns)}"
        )

    logs["pitcher_key"] = logs[pitcher_name_column].apply(
        normalize_player_name
    )

    logs["game_outs"] = logs[innings_column].apply(
        baseball_innings_to_outs
    )

    if date_column is not None:
        logs["game_date"] = pd.to_datetime(
            logs[date_column],
            errors="coerce",
        )
    else:
        logs["game_date"] = pd.NaT

    logs = logs.dropna(
        subset=[
            "pitcher_key",
            "game_outs",
        ]
    ).copy()

    logs = logs[
        (logs["game_outs"] >= 0)
        & (logs["game_outs"] <= 27)
    ].copy()

    logs = logs.sort_values(
        ["pitcher_key", "game_date"],
        ascending=[True, False],
    )

    return logs


def calculate_pitcher_projection(pitcher_logs):
    """
    Build a real pitcher-outs projection from recent appearances.

    Weighting:
        50% last 3 starts
        30% last 5 starts
        20% season average

    When fewer games are available, the available averages are reweighted.
    """
    if pitcher_logs.empty:
        return {
            "last3_avg_outs": None,
            "last5_avg_outs": None,
            "season_avg_outs": None,
            "projected_outs": DEFAULT_PROJECTED_OUTS,
            "history_games": 0,
            "projection_confidence": "LOW",
        }

    outs = pitcher_logs["game_outs"].dropna()

    if outs.empty:
        return {
            "last3_avg_outs": None,
            "last5_avg_outs": None,
            "season_avg_outs": None,
            "projected_outs": DEFAULT_PROJECTED_OUTS,
            "history_games": 0,
            "projection_confidence": "LOW",
        }

    last3_average = outs.head(3).mean()
    last5_average = outs.head(5).mean()
    season_average = outs.mean()

    weighted_values = []
    weighted_weights = []

    if pd.notna(last3_average):
        weighted_values.append(last3_average)
        weighted_weights.append(0.50)

    if pd.notna(last5_average):
        weighted_values.append(last5_average)
        weighted_weights.append(0.30)

    if pd.notna(season_average):
        weighted_values.append(season_average)
        weighted_weights.append(0.20)

    if weighted_values:
        projected_outs = sum(
            value * weight
            for value, weight in zip(
                weighted_values,
                weighted_weights,
            )
        ) / sum(weighted_weights)
    else:
        projected_outs = DEFAULT_PROJECTED_OUTS

    projected_outs = max(
        MIN_PROJECTED_OUTS,
        min(MAX_PROJECTED_OUTS, projected_outs),
    )

    history_games = len(outs)

    if history_games >= 10:
        confidence = "HIGH"
    elif history_games >= 5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "last3_avg_outs": round(last3_average, 2),
        "last5_avg_outs": round(last5_average, 2),
        "season_avg_outs": round(season_average, 2),
        "projected_outs": round(projected_outs, 2),
        "history_games": history_games,
        "projection_confidence": confidence,
    }


def project_pitcher_outs():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pitchers = load_todays_pitchers()
    game_logs = load_pitcher_game_logs()

    projection_rows = []

    for _, pitcher in pitchers.iterrows():
        pitcher_key = pitcher["pitcher_key"]

        pitcher_history = game_logs[
            game_logs["pitcher_key"] == pitcher_key
        ].copy()

        projection = calculate_pitcher_projection(
            pitcher_history
        )

        row = pitcher.to_dict()
        row.update(projection)

        projection_rows.append(row)

    projections = pd.DataFrame(projection_rows)

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
        and column != "pitcher_key"
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
            "opponent",
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

    low_confidence_count = (
        projections["projection_confidence"]
        .eq("LOW")
        .sum()
        if "projection_confidence" in projections.columns
        else 0
    )

    print(
        f"\nLow-confidence pitcher projections: "
        f"{low_confidence_count}"
    )

    return projections


if __name__ == "__main__":
    project_pitcher_outs()
