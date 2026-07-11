import os
import pandas as pd

PROJECTIONS_PATH = "outputs/mlb_universal_projections.csv"
LINES_PATH = "data/platform_lines.csv"
OUTPUT_PATH = "outputs/mlb_daily_card.csv"

MARKET_THRESHOLDS = {
    "hitter_hits": {"A+": 0.45, "A": 0.30, "B": 0.15},
    "hitter_total_bases": {"A+": 0.80, "A": 0.50, "B": 0.25},
    "pitcher_strikeouts": {"A+": 1.25, "A": 0.85, "B": 0.50},
    "pitcher_outs": {"A+": 2.00, "A": 1.25, "B": 0.75},
    "pitcher_fantasy_score": {"A+": 5.00, "A": 3.00, "B": 1.50},
}


def grade_edge(market, edge):
    absolute_edge = abs(edge)
    thresholds = MARKET_THRESHOLDS.get(market)

    if thresholds is None:
        return "UNRATED"

    if absolute_edge >= thresholds["A+"]:
        return "A+"
    if absolute_edge >= thresholds["A"]:
        return "A"
    if absolute_edge >= thresholds["B"]:
        return "B"

    return "PASS"


def normalize_player_name(series):
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def build_daily_card():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load Data
    projections = pd.read_csv(PROJECTIONS_PATH)
    lines = pd.read_csv(LINES_PATH)

    # 2. Basic Cleaning
    for dataframe in [projections, lines]:
        dataframe["player"] = dataframe["player"].astype(str).str.strip()
        dataframe["market"] = (
            dataframe["market"].astype(str).str.strip().str.lower()
        )

    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")

    # 3. Create Normalization Keys
    lines["player_key"] = normalize_player_name(lines["player"])
    projections["player_key"] = normalize_player_name(projections["player"])

    # 4. Merge DataFrames
    merged = lines.merge(
        projections,
        left_on=["player_key", "market"],
        right_on=["player_key", "market"],
        how="left",
        suffixes=("_line", "_projection"),
    )

    merged["projection"] = pd.to_numeric(
        merged["projection"], errors="coerce"
    )

    # 5. Calculations and Logic
    merged["edge"] = merged["projection"] - merged["line"]

    merged["pick"] = merged["edge"].apply(
        lambda value: (
            "MORE/YES"
            if pd.notna(value) and value > 0
            else "LESS/NO"
            if pd.notna(value)
            else "NO PROJECTION"
        )
    )

    merged["grade"] = merged.apply(
        lambda row: (
            grade_edge(row["market"], row["edge"])
            if pd.notna(row["edge"])
            else "NO PROJECTION"
        ),
        axis=1,
    )

    # --- YOUR UPDATED SECTION BEGINS HERE ---
    merged["absolute_edge"] = merged["edge"].abs()

    # Restore player/team/opponent columns after the merge
    if "player_line" in merged.columns:
        merged["player"] = merged["player_line"]
    elif "player_projection" in merged.columns:
        merged["player"] = merged["player_projection"]

    if "team_line" in merged.columns:
        merged["team"] = merged["team_line"]
    elif "team_projection" in merged.columns:
        merged["team"] = merged["team_projection"]

    if "opponent_line" in merged.columns:
        merged["opponent"] = merged["opponent_line"]
    elif "opponent_projection" in merged.columns:
        merged["opponent"] = merged["opponent_projection"]

    # Remove rows that do not have a model projection
    merged = merged[merged["projection"].notna()].copy()

    # Rank grades correctly
    grade_rank = {
        "A+": 1,
        "A": 2,
        "B": 3,
        "PASS": 4,
        "UNRATED": 5,
        "NO PROJECTION": 6,
    }

    merged["grade_rank"] = merged["grade"].map(grade_rank).fillna(99)

    # Sort best props first
    merged = merged.sort_values(
        ["grade_rank", "absolute_edge"],
        ascending=[True, False],
    )

    # Keep only the highest-ranked prop for each player
    merged = merged.drop_duplicates(
        subset=["player_key"],
        keep="first",
    )

    output_columns = [
        "grade",
        "platform",
        "player",
        "market",
        "line",
        "projection",
        "edge",
        "pick",
        "team",
        "opponent",
    ]

    output_columns = [
        column for column in output_columns if column in merged.columns
    ]

    output = merged[output_columns].copy()
    # --- YOUR UPDATED SECTION ENDS HERE ---

    # 6. Save and Output Results
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(output)} rows to {OUTPUT_PATH}\n")
    print(output.to_string(index=False))

    missing = output[output["projection"].isna()]

    if not missing.empty:
        print("\nLines with no matching projection:")
        print(
            missing[["platform", "player", "market", "line"]].to_string(
                index=False
            )
        )

    return output


if __name__ == "__main__":
    build_daily_card()
