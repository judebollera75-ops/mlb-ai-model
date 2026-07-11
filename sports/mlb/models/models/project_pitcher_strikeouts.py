from datetime import date
from pathlib import Path

import pandas as pd


FINAL_DATA_DIRECTORY = Path("data/final")
PITCHERS_DIRECTORY = Path("data/pitchers")
OUTPUT_PATH = Path("outputs/pitcher_strikeout_projections.csv")


def find_current_pitcher_file(target_date):
    pitcher_path = PITCHERS_DIRECTORY / f"{target_date}.csv"

    if not pitcher_path.exists():
        raise FileNotFoundError(
            f"Missing current pitcher file: {pitcher_path}"
        )

    return pitcher_path


def project_strikeouts(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_path = FINAL_DATA_DIRECTORY / "master_dataset.csv"
    pitcher_path = find_current_pitcher_file(target_date)

    if not master_path.exists():
        raise FileNotFoundError(
            f"Missing master dataset: {master_path}"
        )

    master = pd.read_csv(master_path)
    current_pitchers = pd.read_csv(pitcher_path)

    required_master_columns = {
        "pitcher_name",
        "team",
        "games_started",
        "strikeouts",
        "avg_k",
    }

    missing_master_columns = (
        required_master_columns - set(master.columns)
    )

    if missing_master_columns:
        raise KeyError(
            f"{master_path} is missing columns: "
            f"{sorted(missing_master_columns)}"
        )

    required_pitcher_columns = {
        "pitcher_name",
        "team",
        "game_id",
    }

    missing_pitcher_columns = (
        required_pitcher_columns - set(current_pitchers.columns)
    )

    if missing_pitcher_columns:
        raise KeyError(
            f"{pitcher_path} is missing columns: "
            f"{sorted(missing_pitcher_columns)}"
        )

    for dataframe in [master, current_pitchers]:
        dataframe["pitcher_name"] = (
            dataframe["pitcher_name"]
            .astype(str)
            .str.strip()
        )

        dataframe["pitcher_key"] = (
            dataframe["pitcher_name"]
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    numeric_columns = [
        "games_started",
        "strikeouts",
        "avg_k",
    ]

    for column in numeric_columns:
        master[column] = pd.to_numeric(
            master[column],
            errors="coerce",
        )

    master["season_k_per_start"] = (
        master["strikeouts"]
        / master["games_started"].replace(0, pd.NA)
    )

    master["projected_ks"] = (
        master["avg_k"] * 0.70
        + master["season_k_per_start"] * 0.30
    )

    current_pitchers = current_pitchers[
        [
            "game_id",
            "pitcher_name",
            "team",
            "side",
            "pitcher_key",
        ]
    ].copy()

    projections = current_pitchers.merge(
        master[
            [
                "pitcher_key",
                "games_started",
                "strikeouts",
                "season_k_per_start",
                "avg_k",
                "projected_ks",
            ]
        ],
        on="pitcher_key",
        how="left",
    )

    fallback_columns = [
        "avg_k",
        "season_k_per_start",
        "projected_ks",
    ]

    for column in fallback_columns:
        projections[column] = pd.to_numeric(
            projections[column],
            errors="coerce",
        )

    league_average_k = pd.to_numeric(
        master["projected_ks"],
        errors="coerce",
    ).median()

    if pd.isna(league_average_k):
        league_average_k = 5.0

    projections["projected_ks"] = (
        projections["projected_ks"]
        .fillna(league_average_k)
    )

    projections["projection_confidence"] = (
        projections["games_started"]
        .apply(
            lambda value: (
                "HIGH"
                if pd.notna(value) and value >= 5
                else "MEDIUM"
                if pd.notna(value) and value >= 3
                else "LOW"
            )
        )
    )

    projections["date"] = target_date

    output_columns = [
        "date",
        "game_id",
        "pitcher_name",
        "team",
        "side",
        "games_started",
        "strikeouts",
        "season_k_per_start",
        "avg_k",
        "projected_ks",
        "projection_confidence",
    ]

    output = projections[output_columns].copy()

    output["projected_ks"] = (
        output["projected_ks"].round(3)
    )

    output = output.sort_values(
        "projected_ks",
        ascending=False,
    ).reset_index(drop=True)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(output)} current-slate strikeout "
        f"projections to {OUTPUT_PATH}"
    )

    print(
        output.head(30).to_string(
            index=False
        )
    )

    missing_history = output[
        output["games_started"].isna()
    ]

    if not missing_history.empty:
        print(
            "\nPitchers using league-average fallback:"
        )
        print(
            missing_history[
                [
                    "pitcher_name",
                    "team",
                    "projected_ks",
                ]
            ].to_string(index=False)
        )

    return output


if __name__ == "__main__":
    project_strikeouts()
