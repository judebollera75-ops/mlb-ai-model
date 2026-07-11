import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECTIONS_PATH = Path("outputs/mlb_universal_projections.csv")
LINES_PATH = Path("data/platform_lines.csv")
OUTPUT_PATH = Path("outputs/mlb_daily_card.csv")
AUDIT_PATH = Path("outputs/mlb_daily_card_audit.csv")

# Only grades that should appear as recommendations in the app.
ACTIONABLE_GRADES = ["A+", "A", "B"]

# Keep one recommendation per player.
ONE_PROP_PER_PLAYER = True

# Prevent the board from becoming excessively large.
MAX_CARD_ROWS = 50

MARKET_THRESHOLDS = {
    "hitter_hits": {
        "A+": 0.45,
        "A": 0.30,
        "B": 0.15,
    },
    "hitter_total_bases": {
        "A+": 0.80,
        "A": 0.50,
        "B": 0.25,
    },
    "pitcher_strikeouts": {
        "A+": 1.25,
        "A": 0.85,
        "B": 0.50,
    },
    "pitcher_outs": {
        "A+": 2.00,
        "A": 1.25,
        "B": 0.75,
    },
    "pitcher_fantasy_score": {
        "A+": 5.00,
        "A": 3.00,
        "B": 1.50,
    },
}

# Remove obviously abnormal or misleading alternate lines.
VALID_LINE_RANGES = {
    "hitter_hits": (0.5, 2.5),
    "hitter_total_bases": (0.5, 3.5),
    "pitcher_strikeouts": (1.5, 12.5),
    "pitcher_outs": (11.5, 21.5),
    "pitcher_fantasy_score": (10.0, 55.0),
}

GRADE_RANK = {
    "A+": 1,
    "A": 2,
    "B": 3,
    "PASS": 4,
    "UNRATED": 5,
    "NO PROJECTION": 6,
}


def normalize_text(value):
    """Normalize names and team names for reliable matching."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.replace("&", "and")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_series(series):
    return series.apply(normalize_text)


def grade_edge(market, edge):
    if pd.isna(edge):
        return "NO PROJECTION"

    thresholds = MARKET_THRESHOLDS.get(market)

    if thresholds is None:
        return "UNRATED"

    absolute_edge = abs(float(edge))

    if absolute_edge >= thresholds["A+"]:
        return "A+"

    if absolute_edge >= thresholds["A"]:
        return "A"

    if absolute_edge >= thresholds["B"]:
        return "B"

    return "PASS"


def normalize_direction(value):
    """Convert sportsbook side labels into the app's pick labels."""
    normalized = normalize_text(value)

    if normalized in {
        "over",
        "more",
        "yes",
        "higher",
    }:
        return "MORE/YES"

    if normalized in {
        "under",
        "less",
        "no",
        "lower",
    }:
        return "LESS/NO"

    return ""


def line_is_valid(row):
    market = row.get("market")
    line = row.get("line")

    if market not in VALID_LINE_RANGES:
        return False

    if pd.isna(line):
        return False

    minimum_line, maximum_line = VALID_LINE_RANGES[market]

    return minimum_line <= float(line) <= maximum_line


def matchup_matches(row):
    """
    Verify that the projection matchup matches the sportsbook event.

    This prevents a current projection from being joined to a line from
    another date or another matchup simply because the player name matches.
    """
    projection_team = row.get("team_key", "")
    projection_opponent = row.get("opponent_key", "")
    home_team = row.get("home_team_key", "")
    away_team = row.get("away_team_key", "")

    # If matchup data is unavailable, do not automatically reject the row.
    if not projection_team or not projection_opponent:
        return True

    if not home_team or not away_team:
        return True

    event_teams = {
        home_team,
        away_team,
    }

    projection_teams = {
        projection_team,
        projection_opponent,
    }

    return event_teams == projection_teams


def restore_column(merged, base_name):
    """Restore a readable column after pandas adds merge suffixes."""
    line_column = f"{base_name}_line"
    projection_column = f"{base_name}_projection"

    if line_column in merged.columns:
        merged[base_name] = merged[line_column]
        return

    if projection_column in merged.columns:
        merged[base_name] = merged[projection_column]


def load_data():
    if not PROJECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing projections file: {PROJECTIONS_PATH}"
        )

    if not LINES_PATH.exists():
        raise FileNotFoundError(
            f"Missing sportsbook lines file: {LINES_PATH}"
        )

    projections = pd.read_csv(PROJECTIONS_PATH)
    lines = pd.read_csv(LINES_PATH)

    required_projection_columns = {
        "player",
        "market",
        "projection",
    }

    required_line_columns = {
        "player",
        "market",
        "line",
        "platform",
    }

    missing_projection_columns = (
        required_projection_columns - set(projections.columns)
    )

    missing_line_columns = (
        required_line_columns - set(lines.columns)
    )

    if missing_projection_columns:
        raise KeyError(
            "Projection file is missing columns: "
            f"{sorted(missing_projection_columns)}"
        )

    if missing_line_columns:
        raise KeyError(
            "Sportsbook file is missing columns: "
            f"{sorted(missing_line_columns)}"
        )

    return projections, lines


def build_daily_card():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    projections, lines = load_data()

    print(
        f"Loaded {len(projections)} model projections and "
        f"{len(lines)} sportsbook rows."
    )

    if projections.empty or lines.empty:
        empty_columns = [
            "grade",
            "platform",
            "player",
            "market",
            "direction",
            "line",
            "sportsbook_odds",
            "projection",
            "edge",
            "absolute_edge",
            "pick",
            "team",
            "opponent",
            "commence_time",
        ]

        empty_output = pd.DataFrame(columns=empty_columns)
        empty_output.to_csv(OUTPUT_PATH, index=False)

        print(f"Saved 0 rows to {OUTPUT_PATH}")
        return empty_output

    # Clean market names and numeric values.
    for dataframe in [projections, lines]:
        dataframe["player"] = (
            dataframe["player"]
            .astype(str)
            .str.strip()
        )

        dataframe["market"] = (
            dataframe["market"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    lines["line"] = pd.to_numeric(
        lines["line"],
        errors="coerce",
    )

    projections["projection"] = pd.to_numeric(
        projections["projection"],
        errors="coerce",
    )

    # Create normalized player keys.
    lines["player_key"] = normalize_series(lines["player"])
    projections["player_key"] = normalize_series(
        projections["player"]
    )

    # Normalize matchup fields when available.
    if "team" in projections.columns:
        projections["team_key"] = normalize_series(
            projections["team"]
        )
    else:
        projections["team_key"] = ""

    if "opponent" in projections.columns:
        projections["opponent_key"] = normalize_series(
            projections["opponent"]
        )
    else:
        projections["opponent_key"] = ""

    if "home_team" in lines.columns:
        lines["home_team_key"] = normalize_series(
            lines["home_team"]
        )
    else:
        lines["home_team_key"] = ""

    if "away_team" in lines.columns:
        lines["away_team_key"] = normalize_series(
            lines["away_team"]
        )
    else:
        lines["away_team_key"] = ""

    # Remove unusable input rows before merging.
    lines = lines.dropna(
        subset=[
            "player",
            "market",
            "line",
            "platform",
        ]
    ).copy()

    projections = projections.dropna(
        subset=[
            "player",
            "market",
            "projection",
        ]
    ).copy()

    lines = lines[
        lines.apply(
            line_is_valid,
            axis=1,
        )
    ].copy()

    print(
        f"{len(lines)} sportsbook rows remained after "
        "valid-line filtering."
    )

    # Merge sportsbook lines with model projections.
    merged = lines.merge(
        projections,
        on=[
            "player_key",
            "market",
        ],
        how="left",
        suffixes=(
            "_line",
            "_projection",
        ),
    )

    restore_column(merged, "player")
    restore_column(merged, "team")
    restore_column(merged, "opponent")

    # Remove unmatched model rows.
    merged["projection"] = pd.to_numeric(
        merged["projection"],
        errors="coerce",
    )

    merged = merged[
        merged["projection"].notna()
    ].copy()

    print(
        f"{len(merged)} rows matched a model projection."
    )

    # Reject lines from the wrong game or wrong slate.
    merged = merged[
        merged.apply(
            matchup_matches,
            axis=1,
        )
    ].copy()

    print(
        f"{len(merged)} rows remained after matchup validation."
    )

    # Calculate model edge and recommendation.
    merged["edge"] = (
        merged["projection"]
        - merged["line"]
    )

    merged["absolute_edge"] = merged["edge"].abs()

    merged["pick"] = merged["edge"].apply(
        lambda edge: (
            "MORE/YES"
            if edge > 0
            else "LESS/NO"
        )
    )

    # Only use the sportsbook outcome whose direction agrees with the model.
    if "direction" in merged.columns:
        merged["normalized_direction"] = merged[
            "direction"
        ].apply(normalize_direction)

        known_direction = (
            merged["normalized_direction"] != ""
        )

        matching_direction = (
            merged["normalized_direction"]
            == merged["pick"]
        )

        merged = merged[
            (~known_direction)
            | matching_direction
        ].copy()

    merged["grade"] = merged.apply(
        lambda row: grade_edge(
            row["market"],
            row["edge"],
        ),
        axis=1,
    )

    merged["grade_rank"] = (
        merged["grade"]
        .map(GRADE_RANK)
        .fillna(99)
        .astype(int)
    )

    # Keep a complete diagnostic file before final card filtering.
    merged.to_csv(
        AUDIT_PATH,
        index=False,
    )

    # Only actionable recommendations belong on the live app.
    merged = merged[
        merged["grade"].isin(ACTIONABLE_GRADES)
    ].copy()

    # Prefer the best grade, then largest edge.
    sort_columns = [
        "grade_rank",
        "absolute_edge",
    ]

    ascending_values = [
        True,
        False,
    ]

    if "sportsbook_odds" in merged.columns:
        merged["sportsbook_odds"] = pd.to_numeric(
            merged["sportsbook_odds"],
            errors="coerce",
        )

    merged = merged.sort_values(
        sort_columns,
        ascending=ascending_values,
    )

    # Remove repeated identical lines.
    duplicate_columns = [
        column
        for column in [
            "platform",
            "player_key",
            "market",
            "line",
            "pick",
        ]
        if column in merged.columns
    ]

    if duplicate_columns:
        merged = merged.drop_duplicates(
            subset=duplicate_columns,
            keep="first",
        )

    # Keep only the highest-ranked recommendation for each player.
    if ONE_PROP_PER_PLAYER:
        merged = merged.drop_duplicates(
            subset=["player_key"],
            keep="first",
        )

    merged = merged.head(MAX_CARD_ROWS).copy()

    output_columns = [
        "grade",
        "platform",
        "player",
        "market",
        "direction",
        "line",
        "sportsbook_odds",
        "projection",
        "edge",
        "absolute_edge",
        "pick",
        "team",
        "opponent",
        "commence_time",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in merged.columns
    ]

    output = merged[
        output_columns
    ].copy()

    numeric_columns = [
        "line",
        "sportsbook_odds",
        "projection",
        "edge",
        "absolute_edge",
    ]

    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce",
            )

    for column in [
        "line",
        "projection",
        "edge",
        "absolute_edge",
    ]:
        if column in output.columns:
            output[column] = output[column].round(3)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved {len(output)} actionable props "
        f"to {OUTPUT_PATH}"
    )

    print(
        f"Saved full merge diagnostics to {AUDIT_PATH}"
    )

    if output.empty:
        print(
            "No actionable props survived the matching "
            "and quality filters."
        )
    else:
        print()
        print(output.to_string(index=False))

        print("\nCard breakdown by grade:")
        print(
            output["grade"]
            .value_counts()
            .to_string()
        )

        print("\nCard breakdown by market:")
        print(
            output["market"]
            .value_counts()
            .to_string()
        )

    return output


if __name__ == "__main__":
    build_daily_card()
