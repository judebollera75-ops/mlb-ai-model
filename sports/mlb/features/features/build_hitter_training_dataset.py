import os
import pandas as pd

INPUT_PATH = "data/hitter_game_logs/2026.csv"
OUTPUT_PATH = "data/training/hitter_training_dataset.csv"

# Update these if your platform uses different hitter fantasy scoring
SINGLE_PTS = 3
DOUBLE_PTS = 5
TRIPLE_PTS = 8
HR_PTS = 10
RUN_PTS = 2
RBI_PTS = 2
WALK_PTS = 2
HBP_PTS = 2
SB_PTS = 5

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

    # Make sure required raw stat columns exist even if missing from source
    for column in STAT_COLUMNS:
        if column not in df.columns:
            df[column] = 0

    # Derived stats
    df["singles"] = (
        df["hits"]
        - df["doubles"]
        - df["triples"]
        - df["home_runs"]
    ).clip(lower=0)

    df["hits_runs_rbis"] = (
        df["hits"]
        + df["runs"]
        + df["rbi"]
    )

    df["fantasy_score"] = (
        df["singles"] * SINGLE_PTS
        + df["doubles"] * DOUBLE_PTS
        + df["triples"] * TRIPLE_PTS
        + df["home_runs"] * HR_PTS
        + df["runs"] * RUN_PTS
        + df["rbi"] * RBI_PTS
        + df["walks"] * WALK_PTS
        + df["hit_by_pitch"] * HBP_PTS
        + df["stolen_bases"] * SB_PTS
    )

    # Days since the hitter's previous game
    df["days_rest"] = (
        df.groupby("player_id")["date"]
        .diff()
        .dt.days
        .clip(lower=0, upper=14)
    )

    grouped = df.groupby("player_id", group_keys=False)

    rolling_stats = [
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
        "hits_runs_rbis",
        "fantasy_score",
    ]

    # Shift by one game before rolling so only pregame info is used
    for window in [3, 5, 10]:
        for stat in rolling_stats:
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
    previous_game_stats = [
        "hits",
        "total_bases",
        "home_runs",
        "runs",
        "rbi",
        "walks",
        "strikeouts",
        "hits_runs_rbis",
        "fantasy_score",
    ]

    for stat in previous_game_stats:
        if stat in df.columns:
            df[f"previous_game_{stat}"] = grouped[stat].shift(1)

    # Optional rate features per plate appearance
    for window in [3, 5, 10]:
        pa_col = f"last{window}_avg_plate_appearances"

        if pa_col not in df.columns:
            continue

        pa_values = df[pa_col].replace(0, pd.NA)

        for stat in [
            "hits",
            "total_bases",
            "home_runs",
            "runs",
            "rbi",
            "walks",
            "strikeouts",
            "stolen_bases",
            "hits_runs_rbis",
            "fantasy_score",
        ]:
            avg_col = f"last{window}_avg_{stat}"
            rate_col = f"last{window}_{stat}_per_pa"

            if avg_col in df.columns:
                df[rate_col] = df[avg_col] / pa_values

    # Targets: what actually happened in the current game
    df["target_hits"] = df["hits"]
    df["target_total_bases"] = df["total_bases"]
    df["target_home_runs"] = df["home_runs"]
    df["target_runs"] = df["runs"]
    df["target_rbi"] = df["rbi"]
    df["target_hits_runs_rbis"] = df["hits_runs_rbis"]
    df["target_fantasy_score"] = df["fantasy_score"]

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

    preview_columns = [
        "date",
        "player_name",
        "prior_games",
        "last5_avg_hits",
        "last5_avg_total_bases",
        "last5_avg_hits_runs_rbis",
        "last5_avg_fantasy_score",
        "target_hits",
        "target_total_bases",
        "target_hits_runs_rbis",
        "target_fantasy_score",
    ]

    preview_columns = [
        col for col in preview_columns
        if col in df.columns
    ]

    print(df[preview_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    build_hitter_training_dataset()
