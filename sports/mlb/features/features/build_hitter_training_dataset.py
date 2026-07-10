import os
import pandas as pd

INPUT_PATH = "data/hitter_game_logs/2026.csv"
OUTPUT_PATH = "data/training/hitter_training_dataset.csv"

STAT_COLUMNS = [
    "plate_appearances",
    "at_bats",
    "hits",
    "doubles",
    "triples",
    "home_runs",
    "total_bases",
    "runs",
    "rbi",
    "walks",
    "strikeouts",
    "stolen_bases",
    "hit_by_pitch",
]


def build_hitter_training_dataset():
    os.makedirs("data/training", exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "player_id"])
    df = df.sort_values(
        ["player_id", "date", "game_id"]
    ).reset_index(drop=True)

    for column in STAT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)

    # Days since the hitter's previous game
    df["days_rest"] = (
        df.groupby("player_id")["date"]
        .diff()
        .dt.days
        .clip(lower=0, upper=14)
    )

    # Shift by one game before rolling.
    # This ensures the current game's result is never used
    # to predict that same game.
    grouped = df.groupby("player_id", group_keys=False)

    for window in [3, 5, 10]:
        for stat in [
            "plate_appearances",
            "at_bats",
            "hits",
            "total_bases",
            "home_runs",
            "runs",
            "rbi",
            "walks",
            "strikeouts",
            "stolen_bases",
        ]:
            if stat not in df.columns:
                continue

            feature_name = f"last{window}_avg_{stat}"

            df[feature_name] = grouped[stat].transform(
                lambda series: (
                    series.shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            )

    # Previous-game indicators
    for stat in [
        "hits",
        "total_bases",
        "home_runs",
        "runs",
        "rbi",
        "walks",
        "strikeouts",
    ]:
        if stat in df.columns:
            df[f"previous_game_{stat}"] = grouped[stat].shift(1)

    # Targets: what actually happened in the current game
    df["target_hits"] = df["hits"]
    df["target_total_bases"] = df["total_bases"]
    df["target_home_runs"] = df["home_runs"]
    df["target_runs"] = df["runs"]
    df["target_rbi"] = df["rbi"]

    # Require some prior history before using a row for training
    df["prior_games"] = grouped.cumcount()
    df = df[df["prior_games"] >= 3].copy()

    df.to_csv(OUTPUT_PATH, index=False)

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("last")
        or column.startswith("previous_game")
        or column in ["days_rest", "prior_games"]
    ]

    print("Hitter training dataset created")
    print("Rows:", len(df))
    print("Players:", df["player_id"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Feature count:", len(feature_columns))
    print("Saved to:", OUTPUT_PATH)

    print(
        df[[
            "date",
            "player_name",
            "prior_games",
            "last5_avg_hits",
            "last5_avg_total_bases",
            "last10_avg_home_runs",
            "target_hits",
            "target_total_bases",
        ]].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    build_hitter_training_dataset()
