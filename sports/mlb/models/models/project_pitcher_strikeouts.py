from datetime import date
from pathlib import Path

import pandas as pd


MASTER_PATH = Path("data/final/master_dataset.csv")
OUTPUT_PATH = Path("outputs/pitcher_strikeout_projections.csv")


def project_strikeouts(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing master dataset: {MASTER_PATH}"
        )

    master = pd.read_csv(MASTER_PATH)

    required_columns = {
        "pitcher_name",
        "team",
        "game_id",
        "side",
        "games_started",
        "strikeouts",
    }

    missing_columns = required_columns - set(master.columns)

    if missing_columns:
        raise KeyError(
            f"{MASTER_PATH} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    numeric_columns = [
        "games_started",
        "strikeouts",
        "season_k_per_start",
        "avg_k",
    ]

    for column in numeric_columns:
        if column not in master.columns:
            master[column] = pd.NA

        master[column] = pd.to_numeric(
            master[column],
            errors="coerce",
        )

    # Recalculate season K/start when necessary.
    calculated_season_rate = (
        master["strikeouts"]
        / master["games_started"].replace(0, pd.NA)
    )

    master["season_k_per_start"] = (
        master["season_k_per_start"]
        .fillna(calculated_season_rate)
    )

    # Use recent and season data when both are available.
    both_available = (
        master["avg_k"].notna()
        & master["season_k_per_start"].notna()
    )

    master["projected_ks"] = pd.NA

    master.loc[both_available, "projected_ks"] = (
        master.loc[both_available, "avg_k"] * 0.70
        + master.loc[
            both_available,
            "season_k_per_start",
        ] * 0.30
    )

    # If recent average is unavailable, use season K/start.
    season_only = (
        master["projected_ks"].isna()
        & master["season_k_per_start"].notna()
    )

    master.loc[season_only, "projected_ks"] = (
        master.loc[
            season_only,
            "season_k_per_start",
        ]
    )

    # If season data is unavailable but recent data exists, use recent.
    recent_only = (
        master["projected_ks"].isna()
        & master["avg_k"].notna()
    )

    master.loc[recent_only, "projected_ks"] = (
        master.loc[
            recent_only,
            "avg_k",
        ]
    )

    master["projected_ks"] = pd.to_numeric(
        master["projected_ks"],
        errors="coerce",
    )

    valid_projection_median = master[
        "projected_ks"
    ].median()

    if pd.isna(valid_projection_median):
        valid_projection_median = 5.0

    # Final fallback only when no usable pitcher data exists.
    master["projected_ks"] = (
        master["projected_ks"]
        .fillna(valid_projection_median)
    )

    def projection_source(row):
        if (
            pd.notna(row["avg_k"])
            and pd.notna(row["season_k_per_start"])
        ):
            return "RECENT_AND_SEASON"

        if pd.notna(row["season_k_per_start"]):
            return "SEASON_ONLY"

        if pd.notna(row["avg_k"]):
            return "RECENT_ONLY"

        return "LEAGUE_FALLBACK"

    master["projection_source"] = master.apply(
        projection_source,
        axis=1,
    )

    def confidence(row):
        games_started = row["games_started"]
        source = row["projection_source"]

        if source == "LEAGUE_FALLBACK":
            return "LOW"

        if source == "SEASON_ONLY":
            if pd.notna(games_started) and games_started >= 10:
                return "MEDIUM"
            return "LOW"

        if pd.notna(games_started) and games_started >= 10:
            return "HIGH"

        if pd.notna(games_started) and games_started >= 5:
            return "MEDIUM"

        return "LOW"

    master["projection_confidence"] = master.apply(
        confidence,
        axis=1,
    )

    master["date"] = target_date

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
        "projection_source",
        "projection_confidence",
    ]

    output = master[output_columns].copy()

    output["projected_ks"] = (
        pd.to_numeric(
            output["projected_ks"],
            errors="coerce",
        )
        .round(3)
    )

    output = output.drop_duplicates(
        subset=[
            "game_id",
            "pitcher_name",
        ],
        keep="first",
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
        f"Saved {len(output)} strikeout projections "
        f"to {OUTPUT_PATH}"
    )

    print()
    print(
        output[
            [
                "pitcher_name",
                "team",
                "season_k_per_start",
                "avg_k",
                "projected_ks",
                "projection_source",
                "projection_confidence",
            ]
        ].to_string(index=False)
    )

    return output


if __name__ == "__main__":
    project_strikeouts()
