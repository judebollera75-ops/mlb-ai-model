import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECTIONS_PATH = Path("outputs/mlb_universal_projections.csv")
LINES_PATH = Path("data/platform_lines.csv")
PROBABILITY_PATH = Path("outputs/probability_table.csv")

OUTPUT_PATH = Path("outputs/mlb_daily_card.csv")
AUDIT_PATH = Path("outputs/mlb_daily_card_audit.csv")

ACTIONABLE_GRADES = ["A+", "A", "B"]

MARKET_LIMITS = {
    "hitter_hits": 15,
    "hitter_total_bases": 15,
    "pitcher_strikeouts": 10,
    "pitcher_outs": 10,
    "pitcher_fantasy_score": 5,
}

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

VALID_LINE_RANGES = {
    "hitter_hits": (0.5, 1.5),
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
    projection_team = row.get("team_key", "")
    projection_opponent = row.get("opponent_key", "")
    home_team = row.get("home_team_key", "")
    away_team = row.get("away_team_key", "")

    if not projection_team or not projection_opponent:
        return True

    if not home_team or not away_team:
        return True

    return {
        projection_team,
        projection_opponent,
    } == {
        home_team,
        away_team,
    }


def restore_column(merged, base_name):
    line_column = f"{base_name}_line"
    projection_column = f"{base_name}_projection"

    if line_column in merged.columns:
        merged[base_name] = merged[line_column]
    elif projection_column in merged.columns:
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


def load_probability_table():
    if not PROBABILITY_PATH.exists():
        print(
            f"Probability file not found: {PROBABILITY_PATH}. "
            "Probabilities will remain blank."
        )

        return pd.DataFrame(
            columns=[
                "probability_player_key",
                "probability_line",
                "over_prob",
                "under_prob",
                "fair_over_odds",
                "fair_under_odds",
            ]
        )

    probabilities = pd.read_csv(PROBABILITY_PATH)

    required_columns = {
        "pitcher",
        "line",
        "over_prob",
        "under_prob",
        "fair_over_odds",
        "fair_under_odds",
    }

    missing_columns = required_columns - set(probabilities.columns)

    if missing_columns:
        print(
            "Probability table is missing columns: "
            f"{sorted(missing_columns)}"
        )

        return pd.DataFrame(
            columns=[
                "probability_player_key",
                "probability_line",
                "over_prob",
                "under_prob",
                "fair_over_odds",
                "fair_under_odds",
            ]
        )

    probabilities = probabilities.copy()

    probabilities["probability_player_key"] = normalize_series(
        probabilities["pitcher"]
    )

    probabilities["probability_line"] = pd.to_numeric(
        probabilities["line"],
        errors="coerce",
    ).round(3)

    for column in [
        "over_prob",
        "under_prob",
        "fair_over_odds",
        "fair_under_odds",
    ]:
        probabilities[column] = pd.to_numeric(
            probabilities[column],
            errors="coerce",
        )

    probabilities = probabilities.dropna(
        subset=[
            "probability_player_key",
            "probability_line",
        ]
    ).copy()

    probabilities = probabilities.drop_duplicates(
        subset=[
            "probability_player_key",
            "probability_line",
        ],
        keep="first",
    )

    print(
        f"Loaded {len(probabilities)} strikeout probability rows "
        f"from {PROBABILITY_PATH}."
    )

    return probabilities[
        [
            "probability_player_key",
            "probability_line",
            "over_prob",
            "under_prob",
            "fair_over_odds",
            "fair_under_odds",
        ]
    ].copy()


def add_strikeout_probabilities(merged):
    merged = merged.copy()

    merged["win_probability"] = pd.NA
    merged["fair_odds"] = pd.NA
    merged["calibration_status"] = "PENDING"

    probabilities = load_probability_table()

    if probabilities.empty:
        return merged

    strikeout_mask = (
        merged["market"] == "pitcher_strikeouts"
    )

    if not strikeout_mask.any():
        return merged

    strikeout_rows = merged.loc[
        strikeout_mask
    ].copy()

    strikeout_rows["probability_player_key"] = (
        strikeout_rows["player_key"]
    )

    strikeout_rows["probability_line"] = pd.to_numeric(
        strikeout_rows["line"],
        errors="coerce",
    ).round(3)

    strikeout_rows["_original_index"] = strikeout_rows.index

    strikeout_rows = strikeout_rows.merge(
        probabilities,
        on=[
            "probability_player_key",
            "probability_line",
        ],
        how="left",
    )

    strikeout_rows["win_probability"] = strikeout_rows.apply(
        lambda row: (
            row["over_prob"]
            if row["pick"] == "MORE/YES"
            else row["under_prob"]
            if row["pick"] == "LESS/NO"
            else pd.NA
        ),
        axis=1,
    )

    strikeout_rows["fair_odds"] = strikeout_rows.apply(
        lambda row: (
            row["fair_over_odds"]
            if row["pick"] == "MORE/YES"
            else row["fair_under_odds"]
            if row["pick"] == "LESS/NO"
            else pd.NA
        ),
        axis=1,
    )

    strikeout_rows["calibration_status"] = (
        strikeout_rows["win_probability"]
        .notna()
        .map(
            {
                True: "CALIBRATED",
                False: "PENDING",
            }
        )
    )

    strikeout_rows = strikeout_rows.set_index(
        "_original_index"
    )

    merged.loc[
        strikeout_rows.index,
        "win_probability",
    ] = strikeout_rows["win_probability"]

    merged.loc[
        strikeout_rows.index,
        "fair_odds",
    ] = strikeout_rows["fair_odds"]

    merged.loc[
        strikeout_rows.index,
        "calibration_status",
    ] = strikeout_rows["calibration_status"]

    calibrated_count = (
        merged["win_probability"].notna().sum()
    )

    print(
        f"Matched probabilities for {calibrated_count} "
        "strikeout rows."
    )

    return merged


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
            "win_probability",
            "fair_odds",
            "calibration_status",
            "team",
            "opponent",
            "commence_time",
        ]

        empty_output = pd.DataFrame(columns=empty_columns)
        empty_output.to_csv(OUTPUT_PATH, index=False)

        print(f"Saved 0 rows to {OUTPUT_PATH}")
        return empty_output

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

    lines["player_key"] = normalize_series(lines["player"])
    projections["player_key"] = normalize_series(
        projections["player"]
    )

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

    merged = merged[
        merged.apply(
            matchup_matches,
            axis=1,
        )
    ].copy()

    print(
        f"{len(merged)} rows remained after matchup validation."
    )

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

    # Add calibrated probabilities for pitcher strikeouts.
    merged = add_strikeout_probabilities(merged)

    merged.to_csv(
        AUDIT_PATH,
        index=False,
    )

    merged = merged[
        merged["grade"].isin(ACTIONABLE_GRADES)
    ].copy()

    if "sportsbook_odds" in merged.columns:
        merged["sportsbook_odds"] = pd.to_numeric(
            merged["sportsbook_odds"],
            errors="coerce",
        )

    merged = merged.sort_values(
        [
            "grade_rank",
            "absolute_edge",
        ],
        ascending=[
            True,
            False,
        ],
    )

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

    merged = merged.drop_duplicates(
        subset=[
            "player_key",
            "market",
        ],
        keep="first",
    )

    market_sections = []

    for market, limit in MARKET_LIMITS.items():
        market_rows = merged[
            merged["market"] == market
        ].copy()

        market_rows = market_rows.sort_values(
            [
                "grade_rank",
                "absolute_edge",
            ],
            ascending=[
                True,
                False,
            ],
        )

        market_rows = market_rows.head(limit)

        print(
            f"Selected {len(market_rows)} rows "
            f"for {market}."
        )

        market_sections.append(market_rows)

    if market_sections:
        selected = pd.concat(
            market_sections,
            ignore_index=True,
        )
    else:
        selected = pd.DataFrame()

    selected = selected.sort_values(
        [
            "grade_rank",
            "market",
            "absolute_edge",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

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
        "win_probability",
        "fair_odds",
        "calibration_status",
        "team",
        "opponent",
        "commence_time",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in selected.columns
    ]

    output = selected[
        output_columns
    ].copy()

    for column in [
        "line",
        "projection",
        "edge",
        "absolute_edge",
        "win_probability",
        "fair_odds",
    ]:
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

    if "win_probability" in output.columns:
        output["win_probability"] = (
            output["win_probability"].round(3)
        )

    if "fair_odds" in output.columns:
        output["fair_odds"] = (
            output["fair_odds"].round(0)
        )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved {len(output)} actionable props "
        f"to {OUTPUT_PATH}"
    )

    print(
        f"Saved full diagnostics to {AUDIT_PATH}"
    )

    if output.empty:
        print(
            "No actionable props survived the quality filters."
        )
    else:
        print("\nCard breakdown by market:")
        print(
            output["market"]
            .value_counts()
            .to_string()
        )

        calibrated = output[
            output["win_probability"].notna()
        ]

        print(
            f"\nCalibrated picks in final card: "
            f"{len(calibrated)}"
        )

        print()
        print(output.to_string(index=False))

    return output


if __name__ == "__main__":
    build_daily_card()
