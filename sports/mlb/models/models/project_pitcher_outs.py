from pathlib import Path

import pandas as pd


PITCHERS_DIRECTORY = Path("data/pitchers")
GAME_LOGS_DIRECTORY = Path("data/game_logs")
OUTPUT_PATH = Path("outputs/pitcher_outs_projections.csv")

MIN_PROJECTED_OUTS = 6.0
MAX_PROJECTED_OUTS = 24.0


def innings_to_outs(value):
    """
    Convert MLB innings notation into outs.

    Examples:
        5.0 -> 15
        5.1 -> 16
        5.2 -> 17
        6.0 -> 18
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    try:
        if "." in text:
            whole_text, fraction_text = text.split(".", 1)
        else:
            whole_text, fraction_text = text, "0"

        whole_innings = int(whole_text)

        fraction_digit = (
            int(fraction_text[0])
            if fraction_text
            else 0
        )
    except (TypeError, ValueError):
        return pd.NA

    if fraction_digit not in (0, 1, 2):
        return pd.NA

    return whole_innings * 3 + fraction_digit


def find_latest_pitcher_file():
    if not PITCHERS_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Missing directory: {PITCHERS_DIRECTORY}"
        )

    files = sorted(PITCHERS_DIRECTORY.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No pitcher CSV files found in {PITCHERS_DIRECTORY}"
        )

    return files[-1]


def load_matching_files():
    pitchers_path = find_latest_pitcher_file()

    target_date = pitchers_path.stem
    logs_path = GAME_LOGS_DIRECTORY / f"{target_date}.csv"

    if not logs_path.exists():
        raise FileNotFoundError(
            "Could not find the matching pitcher game-log file.\n"
            f"Pitcher file: {pitchers_path}\n"
            f"Expected logs file: {logs_path}"
        )

    pitchers = pd.read_csv(pitchers_path)
    logs = pd.read_csv(logs_path)

    print(f"Using pitcher file: {pitchers_path}")
    print(f"Using game-log file: {logs_path}")
    print(f"Pitcher rows loaded: {len(pitchers)}")
    print(f"Game-log rows loaded: {len(logs)}")

    required_pitcher_columns = {
        "pitcher_id",
        "pitcher_name",
        "team",
    }

    required_log_columns = {
        "pitcher_id",
        "pitcher_name",
        "game_date",
        "innings",
    }

    missing_pitcher_columns = (
        required_pitcher_columns - set(pitchers.columns)
    )

    missing_log_columns = (
        required_log_columns - set(logs.columns)
    )

    if missing_pitcher_columns:
        raise KeyError(
            f"{pitchers_path} is missing columns: "
            f"{sorted(missing_pitcher_columns)}"
        )

    if missing_log_columns:
        raise KeyError(
            f"{logs_path} is missing columns: "
            f"{sorted(missing_log_columns)}"
        )

    return target_date, pitchers, logs


def clean_data(pitchers, logs):
    pitchers = pitchers.copy()
    logs = logs.copy()

    pitchers["pitcher_id"] = pd.to_numeric(
        pitchers["pitcher_id"],
        errors="coerce",
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
        innings_to_outs
    )

    pitchers = pitchers.dropna(
        subset=[
            "pitcher_id",
            "pitcher_name",
        ]
    ).copy()

    logs = logs.dropna(
        subset=[
            "pitcher_id",
            "game_outs",
        ]
    ).copy()

    pitchers["pitcher_id"] = (
        pitchers["pitcher_id"].astype(int)
    )

    logs["pitcher_id"] = (
        logs["pitcher_id"].astype(int)
    )

    logs["game_outs"] = pd.to_numeric(
        logs["game_outs"],
        errors="coerce",
    )

    logs = logs.dropna(
        subset=["game_outs"]
    ).copy()

    logs = logs[
        logs["game_outs"].between(0, 27)
    ].copy()

    pitchers = pitchers.drop_duplicates(
        subset=["pitcher_id"],
        keep="first",
    )

    logs = logs.sort_values(
        [
            "pitcher_id",
            "game_date",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return pitchers, logs


def calculate_pitcher_summary(logs):
    summaries = []

    for pitcher_id, pitcher_logs in logs.groupby(
        "pitcher_id"
    ):
        pitcher_logs = pitcher_logs.sort_values(
            "game_date",
            ascending=False,
        )

        outs = pitcher_logs["game_outs"].dropna()

        history_games = len(outs)

        if history_games == 0:
            continue

        last3_avg = outs.head(3).mean()
        last5_avg = outs.head(5).mean()
        season_avg = outs.mean()

        available_values = []
        available_weights = []

        if pd.notna(last3_avg):
            available_values.append(last3_avg)
            available_weights.append(0.50)

        if pd.notna(last5_avg):
            available_values.append(last5_avg)
            available_weights.append(0.30)

        if pd.notna(season_avg):
            available_values.append(season_avg)
            available_weights.append(0.20)

        projected_outs = sum(
            value * weight
            for value, weight in zip(
                available_values,
                available_weights,
            )
        ) / sum(available_weights)

        projected_outs = max(
            MIN_PROJECTED_OUTS,
            min(
                MAX_PROJECTED_OUTS,
                projected_outs,
            ),
        )

        if history_games >= 5:
            confidence = "HIGH"
        elif history_games >= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        summaries.append(
            {
                "pitcher_id": int(pitcher_id),
                "history_games": history_games,
                "last3_avg_outs": round(
                    last3_avg,
                    2,
                ),
                "last5_avg_outs": round(
                    last5_avg,
                    2,
                ),
                "season_avg_outs": round(
                    season_avg,
                    2,
                ),
                "projected_outs": round(
                    projected_outs,
                    2,
                ),
                "projection_confidence": confidence,
            }
        )

    return pd.DataFrame(summaries)


def project_pitcher_outs():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_date, pitchers, logs = load_matching_files()
    pitchers, logs = clean_data(pitchers, logs)

    summaries = calculate_pitcher_summary(logs)

    print(
        f"Pitchers with calculated history: "
        f"{len(summaries)}"
    )

    projections = pitchers.merge(
        summaries,
        on="pitcher_id",
        how="left",
        validate="one_to_one",
    )

    unmatched = projections[
        projections["history_games"].isna()
    ].copy()

    if not unmatched.empty:
        print("\nPitchers with no matching game logs:")
        print(
            unmatched[
                [
                    "pitcher_id",
                    "pitcher_name",
                    "team",
                ]
            ].to_string(index=False)
        )

        raise RuntimeError(
            f"{len(unmatched)} pitchers failed to match their "
            "game logs. Refusing to create fallback projections."
        )

    projections.insert(
        0,
        "date",
        target_date,
    )

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

    projections = projections[
        output_columns
    ].copy()

    projections.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved {len(projections)} pitcher-out projections "
        f"to {OUTPUT_PATH}"
    )

    print()
    print(
        projections[
            [
                "pitcher_name",
                "team",
                "history_games",
                "last3_avg_outs",
                "last5_avg_outs",
                "season_avg_outs",
                "projected_outs",
                "projection_confidence",
            ]
        ]
        .sort_values(
            "projected_outs",
            ascending=False,
        )
        .to_string(index=False)
    )

    return projections


if __name__ == "__main__":
    project_pitcher_outs()
