from datetime import date
from pathlib import Path

import pandas as pd


PITCHERS_DIRECTORY = Path("data/pitchers")
PITCHER_STATS_DIRECTORY = Path("data/pitcher_stats")
FEATURES_PATH = Path("data/features/pitcher_features.csv")
OUTPUT_PATH = Path("data/final/master_dataset.csv")


def build_master_dataset(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pitchers_path = (
        PITCHERS_DIRECTORY
        / f"{target_date}.csv"
    )

    stats_path = (
        PITCHER_STATS_DIRECTORY
        / f"{target_date}.csv"
    )

    if not pitchers_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher file: {pitchers_path}"
        )

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher stats file: {stats_path}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing pitcher features file: {FEATURES_PATH}"
        )

    pitchers = pd.read_csv(pitchers_path)
    stats = pd.read_csv(stats_path)
    features = pd.read_csv(FEATURES_PATH)

    required_pitcher_columns = {
        "pitcher_id",
        "pitcher_name",
        "team",
        "game_id",
        "side",
    }

    missing_pitcher_columns = (
        required_pitcher_columns
        - set(pitchers.columns)
    )

    if missing_pitcher_columns:
        raise KeyError(
            f"{pitchers_path} is missing columns: "
            f"{sorted(missing_pitcher_columns)}"
        )

    if "pitcher_id" not in stats.columns:
        raise KeyError(
            f"{stats_path} is missing pitcher_id"
        )

    if "pitcher_id" not in features.columns:
        raise KeyError(
            f"{FEATURES_PATH} is missing pitcher_id"
        )

    for dataframe in [
        pitchers,
        stats,
        features,
    ]:
        dataframe["pitcher_id"] = pd.to_numeric(
            dataframe["pitcher_id"],
            errors="coerce",
        )

        dataframe.dropna(
            subset=["pitcher_id"],
            inplace=True,
        )

        dataframe["pitcher_id"] = (
            dataframe["pitcher_id"].astype(int)
        )

    master = (
        pitchers
        .merge(
            stats,
            on="pitcher_id",
            how="left",
            suffixes=("", "_season"),
        )
        .merge(
            features,
            on="pitcher_id",
            how="left",
            suffixes=("", "_recent"),
        )
    )

    master = master.drop_duplicates(
        subset=[
            "game_id",
            "pitcher_id",
        ],
        keep="first",
    ).reset_index(drop=True)

    master.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(master)} current-slate pitcher rows "
        f"to {OUTPUT_PATH}"
    )

    display_columns = [
        "pitcher_name",
        "team",
        "games_started",
        "strikeouts",
        "season_k_per_start",
        "avg_k",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in master.columns
    ]

    if display_columns:
        print(
            master[display_columns]
            .head(30)
            .to_string(index=False)
        )

    return master


if __name__ == "__main__":
    build_master_dataset()
