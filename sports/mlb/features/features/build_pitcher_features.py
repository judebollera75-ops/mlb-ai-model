from datetime import date
from pathlib import Path

import pandas as pd


GAME_LOGS_DIRECTORY = Path("data/game_logs")
OUTPUT_PATH = Path("data/features/pitcher_features.csv")


def innings_to_decimal(value):
    """
    Convert MLB innings notation to decimal innings.

    Examples:
        5.0 -> 5.0
        5.1 -> 5.333...
        5.2 -> 5.666...
    """
    if value is None or pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    if "." not in text:
        return float(text)

    whole_text, partial_text = text.split(".", 1)

    whole = int(whole_text)
    partial = int(partial_text[0]) if partial_text else 0

    if partial == 1:
        return whole + (1 / 3)

    if partial == 2:
        return whole + (2 / 3)

    return float(whole)


def build_pitcher_features(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_path = (
        GAME_LOGS_DIRECTORY
        / f"{target_date}.csv"
    )

    if not logs_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher game-log file: {logs_path}"
        )

    logs = pd.read_csv(logs_path)

    required_columns = {
        "pitcher_id",
        "pitcher_name",
        "team",
        "innings",
        "strikeouts",
        "walks",
        "hits",
        "earned_runs",
        "home_runs",
        "era",
    }

    missing_columns = required_columns - set(logs.columns)

    if missing_columns:
        raise KeyError(
            f"{logs_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if logs.empty:
        empty_features = pd.DataFrame(
            columns=[
                "pitcher_id",
                "pitcher_name",
                "team",
                "recent_games",
                "avg_ip",
                "avg_k",
                "avg_bb",
                "avg_hits",
                "avg_er",
                "avg_hr",
                "avg_era",
            ]
        )

        empty_features.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print(
            f"No pitcher game logs found for {target_date}."
        )

        return empty_features

    logs["pitcher_id"] = pd.to_numeric(
        logs["pitcher_id"],
        errors="coerce",
    )

    logs = logs.dropna(
        subset=[
            "pitcher_id",
            "pitcher_name",
        ]
    ).copy()

    logs["pitcher_id"] = (
        logs["pitcher_id"].astype(int)
    )

    logs["innings_decimal"] = (
        logs["innings"].apply(innings_to_decimal)
    )

    numeric_columns = [
        "innings_decimal",
        "strikeouts",
        "walks",
        "hits",
        "earned_runs",
        "home_runs",
        "era",
    ]

    for column in numeric_columns:
        logs[column] = pd.to_numeric(
            logs[column],
            errors="coerce",
        )

    if "game_date" in logs.columns:
        logs["game_date"] = pd.to_datetime(
            logs["game_date"],
            errors="coerce",
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

        logs = (
            logs.groupby(
                "pitcher_id",
                group_keys=False,
            )
            .head(5)
            .copy()
        )

    features = (
        logs.groupby(
            [
                "pitcher_id",
                "pitcher_name",
                "team",
            ],
            dropna=False,
        )
        .agg(
            recent_games=(
                "pitcher_id",
                "size",
            ),
            avg_ip=(
                "innings_decimal",
                "mean",
            ),
            avg_k=(
                "strikeouts",
                "mean",
            ),
            avg_bb=(
                "walks",
                "mean",
            ),
            avg_hits=(
                "hits",
                "mean",
            ),
            avg_er=(
                "earned_runs",
                "mean",
            ),
            avg_hr=(
                "home_runs",
                "mean",
            ),
            avg_era=(
                "era",
                "mean",
            ),
        )
        .reset_index()
    )

    numeric_feature_columns = [
        "avg_ip",
        "avg_k",
        "avg_bb",
        "avg_hits",
        "avg_er",
        "avg_hr",
        "avg_era",
    ]

    for column in numeric_feature_columns:
        features[column] = (
            pd.to_numeric(
                features[column],
                errors="coerce",
            )
            .round(3)
        )

    features = features.sort_values(
        [
            "pitcher_name",
        ],
        ascending=True,
    ).reset_index(drop=True)

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(features)} pitcher feature rows "
        f"to {OUTPUT_PATH}"
    )

    print(
        features[
            [
                "pitcher_name",
                "team",
                "recent_games",
                "avg_ip",
                "avg_k",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    return features


if __name__ == "__main__":
    build_pitcher_features()
