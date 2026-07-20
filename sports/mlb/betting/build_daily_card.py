"""Build the production MLB daily prop card.

Inputs:
    outputs/probability_table.csv
    outputs/mlb_universal_projections.csv
    data/platform_lines.csv

Outputs:
    outputs/mlb_daily_card.csv
    outputs/mlb_daily_card_audit.csv

The final card contains only exact platform/player/market/line combinations
that pass probability, expected-value, freshness, calibration, and matchup
quality checks.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elite_filter import apply_elite_filter


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROJECTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "mlb_universal_projections.csv"
)

LINES_PATH = PROJECT_ROOT / "data" / "platform_lines.csv"

PROBABILITY_PATH = (
    PROJECT_ROOT / "outputs" / "probability_table.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "mlb_daily_card.csv"
)

AUDIT_PATH = (
    PROJECT_ROOT / "outputs" / "mlb_daily_card_audit.csv"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "history"
    / "mlb_bet_results.csv"
)

MINIMUM_WIN_PROBABILITY = 0.56
MINIMUM_PROBABILITY_EDGE = 0.025
MINIMUM_EXPECTED_VALUE = 0.015
MINIMUM_CALIBRATION_SAMPLE = 100
MAXIMUM_LINE_AGE_MINUTES = 30
MAXIMUM_KELLY_FRACTION = 0.025
KELLY_MULTIPLIER = 0.25

MARKET_LIMITS = {
    "hitter_hits": 12,
    "hitter_total_bases": 12,
    "hitter_runs": 8,
    "hitter_rbis": 8,
    "hitter_hits_runs_rbis": 12,
    "hitter_fantasy_score": 10,
    "pitcher_strikeouts": 10,
    "pitcher_outs": 8,
}

VALID_LINE_RANGES = {
    "hitter_hits": (0.5, 2.5),
    "hitter_total_bases": (0.5, 5.5),
    "hitter_runs": (0.5, 1.5),
    "hitter_rbis": (0.5, 2.5),
    "hitter_hits_runs_rbis": (0.5, 5.5),
    "hitter_fantasy_score": (0.5, 30.5),
    "pitcher_strikeouts": (1.5, 12.5),
    "pitcher_outs": (8.5, 24.5),
}

# Data-quality guardrail: reject lines that are implausibly far from the
# model projection. This removes malformed/extreme alternate lines without
# changing the underlying projection model.
MAXIMUM_ABSOLUTE_PROJECTION_GAP = {
    "hitter_hits": 1.5,
    "hitter_total_bases": 3.0,
    "hitter_runs": 1.25,
    "hitter_rbis": 1.75,
    "hitter_hits_runs_rbis": 3.5,
    "hitter_fantasy_score": 12.0,
    "pitcher_strikeouts": 4.0,
    "pitcher_outs": 7.0,
}

MARKET_MINIMUM_PROBABILITIES = {
    "hitter_hits": 0.58,
    "hitter_total_bases": 0.58,
    "hitter_runs": 0.59,
    "hitter_rbis": 0.59,
    "hitter_hits_runs_rbis": 0.58,
    "hitter_fantasy_score": 0.58,
    "pitcher_strikeouts": 0.57,
    "pitcher_outs": 0.58,
}

OUTPUT_COLUMNS = [
    "grade",
    "confidence_tier",
    "platform",
    "player",
    "market",
    "direction",
    "line",
    "sportsbook_odds",
    "projection",
    "raw_projection_edge",
    "probability",
    "sportsbook_implied_probability",
    "no_vig_implied_probability",
    "probability_edge",
    "expected_value",
    "fair_odds",
    "kelly_fraction",
    "recommended_bankroll_fraction",
    "distribution_method",
    "calibration_sample_size",
    "validation_mae",
    "elite_score",
    "elite_eligible",
    "elite_rejection_reasons",
    "elite_history_sample",
    "elite_history_win_rate",
    "elite_history_lower_bound",
    "team",
    "opponent",
    "home_team",
    "away_team",
    "commence_time",
    "fetched_at",
]


def normalize_text(value: Any) -> str:
    """Normalize text for safe cross-source matching."""
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_direction(value: Any) -> str:
    """Normalize platform direction names."""
    normalized = normalize_text(value)

    if normalized in {
        "over",
        "more",
        "yes",
        "higher",
    }:
        return "Over"

    if normalized in {
        "under",
        "less",
        "no",
        "lower",
    }:
        return "Under"

    return ""


def american_odds_to_probability(odds: Any) -> float:
    """Convert American odds to raw implied probability."""
    try:
        numeric_odds = float(odds)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(numeric_odds) or numeric_odds == 0:
        return float("nan")

    if numeric_odds > 0:
        return 100.0 / (numeric_odds + 100.0)

    return abs(numeric_odds) / (
        abs(numeric_odds) + 100.0
    )


def american_odds_decimal_return(odds: Any) -> float:
    """Return decimal profit per unit risked."""
    try:
        numeric_odds = float(odds)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(numeric_odds) or numeric_odds == 0:
        return float("nan")

    if numeric_odds > 0:
        return numeric_odds / 100.0

    return 100.0 / abs(numeric_odds)


def expected_value_per_unit(
    probability: Any,
    odds: Any,
) -> float:
    """Calculate expected profit per unit staked."""
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        return float("nan")

    profit_multiple = american_odds_decimal_return(odds)

    if (
        not np.isfinite(probability)
        or not np.isfinite(profit_multiple)
    ):
        return float("nan")

    loss_probability = 1.0 - probability

    return (
        probability * profit_multiple
        - loss_probability
    )


def full_kelly_fraction(
    probability: Any,
    odds: Any,
) -> float:
    """Calculate the full Kelly bankroll fraction."""
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        return float("nan")

    profit_multiple = american_odds_decimal_return(odds)

    if (
        not np.isfinite(probability)
        or not np.isfinite(profit_multiple)
        or profit_multiple <= 0
    ):
        return float("nan")

    loss_probability = 1.0 - probability

    kelly = (
        profit_multiple * probability
        - loss_probability
    ) / profit_multiple

    return max(0.0, float(kelly))


def recommended_bankroll_fraction(
    probability: Any,
    odds: Any,
) -> float:
    """Return capped quarter-Kelly sizing."""
    full_kelly = full_kelly_fraction(
        probability,
        odds,
    )

    if not np.isfinite(full_kelly):
        return 0.0

    fractional_kelly = (
        full_kelly * KELLY_MULTIPLIER
    )

    return min(
        MAXIMUM_KELLY_FRACTION,
        max(0.0, fractional_kelly),
    )


def line_is_valid(
    market: Any,
    line: Any,
) -> bool:
    """Validate market-specific line ranges."""
    market = str(market).strip()

    if market not in VALID_LINE_RANGES:
        return False

    try:
        numeric_line = float(line)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(numeric_line):
        return False

    minimum_line, maximum_line = (
        VALID_LINE_RANGES[market]
    )

    return (
        minimum_line
        <= numeric_line
        <= maximum_line
    )


def line_is_fresh(
    fetched_at: Any,
) -> bool:
    """Reject stale platform prices."""
    parsed = pd.to_datetime(
        fetched_at,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return False

    age = (
        datetime.now(timezone.utc)
        - parsed.to_pydatetime()
    )

    return age.total_seconds() <= (
        MAXIMUM_LINE_AGE_MINUTES * 60
    )


def game_has_not_started(
    commence_time: Any,
) -> bool:
    """Reject markets for games that have begun."""
    parsed = pd.to_datetime(
        commence_time,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return False

    return (
        parsed.to_pydatetime()
        > datetime.now(timezone.utc)
    )


def calculate_no_vig_probabilities(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Remove sportsbook margin when both sides are present."""
    result = frame.copy()

    result["sportsbook_implied_probability"] = (
        result["sportsbook_odds"]
        .apply(american_odds_to_probability)
    )

    result["no_vig_implied_probability"] = np.nan

    grouping_columns = [
        "event_id",
        "platform_key",
        "player_key",
        "market",
        "line",
    ]

    for _, group in result.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    ):
        directions = set(
            group["direction"].dropna()
        )

        if not {"Over", "Under"}.issubset(directions):
            continue

        implied_sum = float(
            group[
                "sportsbook_implied_probability"
            ].sum()
        )

        if (
            not np.isfinite(implied_sum)
            or implied_sum <= 0
        ):
            continue

        result.loc[
            group.index,
            "no_vig_implied_probability",
        ] = (
            group[
                "sportsbook_implied_probability"
            ]
            / implied_sum
        )

    result["market_implied_probability"] = (
        result["no_vig_implied_probability"]
        .fillna(
            result[
                "sportsbook_implied_probability"
            ]
        )
    )

    return result


def confidence_tier(
    probability: Any,
    probability_edge: Any,
    expected_value: Any,
) -> str:
    """Convert model quality into an interpretable tier."""
    try:
        probability = float(probability)
        probability_edge = float(probability_edge)
        expected_value = float(expected_value)
    except (TypeError, ValueError):
        return "PASS"

    if (
        probability >= 0.70
        and probability_edge >= 0.08
        and expected_value >= 0.10
    ):
        return "Elite"

    if (
        probability >= 0.65
        and probability_edge >= 0.06
        and expected_value >= 0.07
    ):
        return "Strong"

    if (
        probability >= 0.60
        and probability_edge >= 0.04
        and expected_value >= 0.04
    ):
        return "Good"

    if (
        probability >= MINIMUM_WIN_PROBABILITY
        and probability_edge >= MINIMUM_PROBABILITY_EDGE
        and expected_value >= MINIMUM_EXPECTED_VALUE
    ):
        return "Playable"

    return "PASS"


def grade_from_tier(tier: str) -> str:
    """Map confidence tiers to card grades."""
    return {
        "Elite": "A+",
        "Strong": "A",
        "Good": "B+",
        "Playable": "B",
        "PASS": "PASS",
    }.get(tier, "PASS")


def load_probability_table() -> pd.DataFrame:
    """Load exact live-line probabilities."""
    if not PROBABILITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing probability table: {PROBABILITY_PATH}"
        )

    try:
        probabilities = pd.read_csv(
            PROBABILITY_PATH
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError(
            f"Could not read probability table: {PROBABILITY_PATH}"
        ) from exc

    required_columns = {
        "platform",
        "player",
        "market",
        "direction",
        "line",
        "projection",
        "probability",
        "probability_status",
    }

    missing_columns = (
        required_columns - set(probabilities.columns)
    )

    if missing_columns:
        raise ValueError(
            "Probability table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return probabilities


def load_projection_context() -> pd.DataFrame:
    """Load team and opponent context from universal projections."""
    if not PROJECTIONS_PATH.exists():
        return pd.DataFrame(
            columns=[
                "player_key",
                "market",
                "team",
                "opponent",
            ]
        )

    try:
        projections = pd.read_csv(
            PROJECTIONS_PATH
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame(
            columns=[
                "player_key",
                "market",
                "team",
                "opponent",
            ]
        )

    required_columns = {
        "player",
        "market",
    }

    if not required_columns.issubset(
        projections.columns
    ):
        return pd.DataFrame(
            columns=[
                "player_key",
                "market",
                "team",
                "opponent",
            ]
        )

    projections["player_key"] = projections[
        "player"
    ].apply(normalize_text)

    projections["market"] = (
        projections["market"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for column in ["team", "opponent"]:
        if column not in projections.columns:
            projections[column] = pd.NA

    projections = projections[
        [
            "player_key",
            "market",
            "team",
            "opponent",
        ]
    ].drop_duplicates(
        subset=["player_key", "market"],
        keep="last",
    )

    return projections


def add_rejection_reason(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Record the first major reason each row is rejected."""
    result = frame.copy()
    result["rejection_reason"] = ""

    checks = [
        (
            ~result["probability_status"].eq(
                "calculated"
            ),
            "probability_not_calculated",
        ),
        (
            result["probability"].isna(),
            "missing_probability",
        ),
        (
            ~result.apply(
                lambda row: line_is_valid(
                    row.get("market"),
                    row.get("line"),
                ),
                axis=1,
            ),
            "invalid_line",
        ),
        (
            result.apply(
                lambda row: (
                    abs(
                        float(row.get("projection"))
                        - float(row.get("line"))
                    )
                    > MAXIMUM_ABSOLUTE_PROJECTION_GAP.get(
                        str(row.get("market")).strip(),
                        float("inf"),
                    )
                )
                if pd.notna(row.get("projection"))
                and pd.notna(row.get("line"))
                else True,
                axis=1,
            ),
            "line_too_far_from_projection",
        ),
        (
            ~result["line_is_fresh"],
            "stale_line",
        ),
        (
            ~result["game_not_started"],
            "game_started",
        ),
        (
            result["calibration_sample_size"]
            .fillna(0)
            .lt(MINIMUM_CALIBRATION_SAMPLE)
            & result["distribution_method"]
            .eq("empirical_holdout_residuals"),
            "insufficient_calibration_sample",
        ),
        (
            result["probability"]
            .lt(result["minimum_market_probability"]),
            "probability_below_threshold",
        ),
        (
            result["probability_edge"]
            .lt(MINIMUM_PROBABILITY_EDGE),
            "probability_edge_below_threshold",
        ),
        (
            result["expected_value"]
            .lt(MINIMUM_EXPECTED_VALUE),
            "expected_value_below_threshold",
        ),
        (
            result["sportsbook_odds"].isna(),
            "missing_price",
        ),
    ]

    for mask, reason in checks:
        apply_mask = (
            result["rejection_reason"].eq("")
            & mask.fillna(True)
        )

        result.loc[
            apply_mask,
            "rejection_reason",
        ] = reason

    result.loc[
        result["rejection_reason"].eq(""),
        "rejection_reason",
    ] = "accepted"

    return result



def choose_consensus_market_lines(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only the consensus/main line for each event, player, and market.

    The main line is the line offered by the greatest number of distinct
    platforms. Ties are broken by choosing the line whose average market
    implied probability is closest to 50%, then by choosing the line closest
    to the median offered line. Alternate lines remain in the raw probability
    table but are excluded from the production daily card.
    """
    if frame.empty:
        return frame

    result = frame.copy()

    # Do not group by event_id: provider-specific event IDs can differ across
    # sportsbooks, which would make every book look like its own market and
    # allow alternate lines to win. The daily pipeline is already slate-bound.
    group_columns = [
        "player_key",
        "market",
    ]

    result["line"] = pd.to_numeric(
        result["line"],
        errors="coerce",
    )

    result["platform_key"] = (
        result.get("platform_key", result.get("platform"))
        .astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
    )

    result["consensus_price_distance"] = (
        result["market_implied_probability"] - 0.5
    ).abs()

    line_summary = (
        result.dropna(subset=["line"])
        .groupby(
            group_columns + ["line"],
            dropna=False,
            sort=False,
        )
        .agg(
            platform_count=("platform_key", "nunique"),
            average_price_distance=(
                "consensus_price_distance",
                "mean",
            ),
        )
        .reset_index()
    )

    if line_summary.empty:
        return result.iloc[0:0].copy()

    medians = (
        line_summary.groupby(
            group_columns,
            dropna=False,
        )["line"]
        .median()
        .rename("group_median_line")
        .reset_index()
    )

    line_summary = line_summary.merge(
        medians,
        on=group_columns,
        how="left",
    )

    line_summary["median_distance"] = (
        line_summary["line"]
        - line_summary["group_median_line"]
    ).abs()

    line_summary = line_summary.sort_values(
        by=group_columns
        + [
            "platform_count",
            "average_price_distance",
            "median_distance",
            "line",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
            True,
        ],
        na_position="last",
    )

    selected_lines = line_summary.drop_duplicates(
        subset=group_columns,
        keep="first",
    )[group_columns + ["line"]]

    selected_lines = selected_lines.rename(
        columns={"line": "consensus_line"}
    )

    result = result.merge(
        selected_lines,
        on=group_columns,
        how="inner",
    )

    result = result.loc[
        np.isclose(
            result["line"],
            result["consensus_line"],
            equal_nan=False,
        )
    ].copy()

    result = result.drop(
        columns=[
            "consensus_price_distance",
            "consensus_line",
        ],
        errors="ignore",
    )

    return result

def choose_best_platform_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the best priced exact side for each player and market."""
    if frame.empty:
        return frame

    ranked = frame.sort_values(
        [
            "player_key",
            "market",
            "direction",
            "expected_value",
            "probability_edge",
            "probability",
            "sportsbook_odds",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ],
    )

    return ranked.drop_duplicates(
        subset=[
            "player_key",
            "market",
        ],
        keep="first",
    )


def event_date_from_commence_time(value: Any) -> str | None:
    """Convert a UTC game timestamp into its Central Time calendar date."""
    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return None

    return parsed.tz_convert(
        "America/Chicago"
    ).date().isoformat()


def load_existing_history() -> pd.DataFrame:
    """Load the saved recommendation history without failing on a new file."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_PATH)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return pd.DataFrame()


def build_history_rows(selected: pd.DataFrame) -> pd.DataFrame:
    """Create unresolved history records from the current actionable card."""
    if selected.empty:
        return pd.DataFrame()

    history = selected.copy()

    history["event_date"] = history[
        "commence_time"
    ].apply(event_date_from_commence_time)

    history["logged_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    history["actual_result"] = pd.NA
    history["outcome"] = pd.NA
    history["grading_status"] = "UNRESOLVED"
    history["grading_note"] = pd.NA
    history["matched_game_pk"] = pd.NA
    history["graded_at"] = pd.NA

    # A one-unit reference stake lets later reports calculate standardized ROI.
    history["stake"] = 1.0
    history["profit"] = pd.NA

    history_columns = [
        "event_date",
        "event_id",
        "platform",
        "platform_key",
        "player",
        "market",
        "direction",
        "line",
        "sportsbook_odds",
        "projection",
        "raw_projection_edge",
        "probability",
        "sportsbook_implied_probability",
        "no_vig_implied_probability",
        "probability_edge",
        "expected_value",
        "fair_odds",
        "kelly_fraction",
        "recommended_bankroll_fraction",
        "distribution_method",
        "calibration_sample_size",
        "validation_mae",
        "grade",
        "confidence_tier",
        "team",
        "opponent",
        "home_team",
        "away_team",
        "commence_time",
        "fetched_at",
        "logged_at",
        "stake",
        "profit",
        "actual_result",
        "outcome",
        "grading_status",
        "grading_note",
        "matched_game_pk",
        "graded_at",
    ]

    for column in history_columns:
        if column not in history.columns:
            history[column] = pd.NA

    history = history.loc[
        history["event_date"].notna()
    ].copy()

    return history[
        history_columns
    ].reset_index(drop=True)


def append_card_to_history(selected: pd.DataFrame) -> int:
    """Append only previously unseen actionable props to permanent history."""
    new_rows = build_history_rows(selected)

    if new_rows.empty:
        print(
            "History logger: no actionable recommendations "
            "were available to save."
        )
        return 0

    existing = load_existing_history()

    all_columns = list(
        dict.fromkeys(
            list(existing.columns)
            + list(new_rows.columns)
        )
    )

    for column in all_columns:
        if column not in existing.columns:
            existing[column] = pd.NA

        if column not in new_rows.columns:
            new_rows[column] = pd.NA

    existing = existing[
        all_columns
    ].copy()

    new_rows = new_rows[
        all_columns
    ].copy()

    dedupe_columns = [
        "event_date",
        "event_id",
        "platform",
        "player",
        "market",
        "direction",
        "line",
    ]

    # Normalize dedupe fields so repeated morning/afternoon workflow runs do
    # not save the same exact recommendation twice.
    def build_keys(frame: pd.DataFrame) -> pd.Series:
        key_frame = frame.copy()

        for column in dedupe_columns:
            if column not in key_frame.columns:
                key_frame[column] = pd.NA

        for column in [
            "event_date",
            "event_id",
            "platform",
            "player",
            "market",
            "direction",
        ]:
            key_frame[column] = (
                key_frame[column]
                .astype("string")
                .fillna("")
                .str.strip()
                .str.casefold()
            )

        key_frame["line"] = pd.to_numeric(
            key_frame["line"],
            errors="coerce",
        ).round(4)

        return key_frame[
            dedupe_columns
        ].astype("string").agg(
            "|".join,
            axis=1,
        )

    existing_keys = set(
        build_keys(existing).tolist()
    ) if not existing.empty else set()

    new_keys = build_keys(new_rows)

    unseen_mask = ~new_keys.isin(
        existing_keys
    )

    unseen = new_rows.loc[
        unseen_mask
    ].copy()

    if unseen.empty:
        print(
            "History logger: every current recommendation "
            "was already saved."
        )
        return 0

    combined = pd.concat(
        [
            existing,
            unseen,
        ],
        ignore_index=True,
    )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = HISTORY_PATH.with_suffix(
        ".tmp.csv"
    )

    combined.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        HISTORY_PATH
    )

    print(
        f"History logger: saved {len(unseen):,} new recommendations "
        f"to {HISTORY_PATH}"
    )

    return len(unseen)


def build_daily_card() -> pd.DataFrame:
    """Create the final production betting card."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    probabilities = load_probability_table()
    projection_context = load_projection_context()

    print("=" * 72)
    print("Building production MLB daily card")
    print(
        f"Probability rows loaded: "
        f"{len(probabilities):,}"
    )
    print("=" * 72)

    if probabilities.empty:
        empty_output = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

        empty_output.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        empty_output.to_csv(
            AUDIT_PATH,
            index=False,
        )

        return empty_output

    probabilities["player_key"] = probabilities[
        "player"
    ].apply(normalize_text)

    probabilities["market"] = (
        probabilities["market"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    probabilities["direction"] = probabilities[
        "direction"
    ].apply(normalize_direction)

    for column in [
        "line",
        "sportsbook_odds",
        "projection",
        "probability",
        "push_probability",
        "opposite_probability",
        "fair_odds",
        "calibration_sample_size",
        "validation_mae",
    ]:
        if column in probabilities.columns:
            probabilities[column] = pd.to_numeric(
                probabilities[column],
                errors="coerce",
            )

    probabilities = probabilities.loc[
        probabilities["player_key"].ne("")
        & probabilities["direction"].isin(
            {"Over", "Under"}
        )
    ].copy()

    probabilities = probabilities.merge(
        projection_context,
        on=[
            "player_key",
            "market",
        ],
        how="left",
    )

    probabilities = calculate_no_vig_probabilities(
        probabilities
    )

    probabilities["raw_projection_edge"] = (
        probabilities["projection"]
        - probabilities["line"]
    )

    probabilities["probability_edge"] = (
        probabilities["probability"]
        - probabilities[
            "market_implied_probability"
        ]
    )

    probabilities["expected_value"] = (
        probabilities.apply(
            lambda row: expected_value_per_unit(
                row.get("probability"),
                row.get("sportsbook_odds"),
            ),
            axis=1,
        )
    )

    probabilities["kelly_fraction"] = (
        probabilities.apply(
            lambda row: full_kelly_fraction(
                row.get("probability"),
                row.get("sportsbook_odds"),
            ),
            axis=1,
        )
    )

    probabilities[
        "recommended_bankroll_fraction"
    ] = probabilities.apply(
        lambda row: recommended_bankroll_fraction(
            row.get("probability"),
            row.get("sportsbook_odds"),
        ),
        axis=1,
    )

    probabilities["line_is_fresh"] = probabilities[
        "fetched_at"
    ].apply(line_is_fresh)

    probabilities["game_not_started"] = probabilities[
        "commence_time"
    ].apply(game_has_not_started)

    probabilities[
        "minimum_market_probability"
    ] = probabilities["market"].map(
        MARKET_MINIMUM_PROBABILITIES
    ).fillna(MINIMUM_WIN_PROBABILITY)

    probabilities = choose_consensus_market_lines(
        probabilities
    )

    # v7: assign Elite only after strict historical and sanity validation.
    probabilities = apply_elite_filter(
        probabilities,
        history_path=HISTORY_PATH,
    )

    probabilities = add_rejection_reason(
        probabilities
    )

    probabilities.to_csv(
        AUDIT_PATH,
        index=False,
    )

    accepted = probabilities.loc[
        probabilities["rejection_reason"]
        .eq("accepted")
    ].copy()

    accepted = choose_best_platform_rows(
        accepted
    )

    accepted = accepted.sort_values(
        [
            "elite_score",
            "expected_value",
            "probability_edge",
            "probability",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    market_sections: list[pd.DataFrame] = []

    for market, limit in MARKET_LIMITS.items():
        market_rows = accepted.loc[
            accepted["market"].eq(market)
        ].head(limit)

        print(
            f"{market}: selected "
            f"{len(market_rows)} of "
            f"{int(accepted['market'].eq(market).sum())} "
            "accepted rows"
        )

        market_sections.append(market_rows)

    if market_sections:
        selected = pd.concat(
            market_sections,
            ignore_index=True,
        )
    else:
        selected = pd.DataFrame(
            columns=accepted.columns
        )

    selected = selected.sort_values(
        [
            "elite_score",
            "expected_value",
            "probability_edge",
            "probability",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    for column in OUTPUT_COLUMNS:
        if column not in selected.columns:
            selected[column] = pd.NA

    output = selected[
        OUTPUT_COLUMNS
    ].copy()

    for column in [
        "line",
        "projection",
        "raw_projection_edge",
        "probability",
        "sportsbook_implied_probability",
        "no_vig_implied_probability",
        "probability_edge",
        "expected_value",
        "kelly_fraction",
        "recommended_bankroll_fraction",
        "validation_mae",
    ]:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).round(4)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    newly_logged = append_card_to_history(
        selected
    )

    print("\n" + "=" * 72)
    print("MLB DAILY CARD COMPLETE")
    print(f"Accepted props: {len(output):,}")
    print(f"Final card: {OUTPUT_PATH}")
    print(f"Full audit: {AUDIT_PATH}")
    print(f"New history rows logged: {newly_logged:,}")
    print("=" * 72)

    if output.empty:
        print(
            "No props passed every probability, price, "
            "freshness, and calibration filter."
        )
    else:
        print("\nCard by market:")
        print(
            output["market"]
            .value_counts()
            .to_string()
        )

        preview_columns = [
            "grade",
            "platform",
            "player",
            "market",
            "direction",
            "line",
            "sportsbook_odds",
            "projection",
            "probability",
            "probability_edge",
            "expected_value",
            "recommended_bankroll_fraction",
        ]

        print("\nTop plays:")
        print(
            output[preview_columns]
            .head(40)
            .to_string(index=False)
        )

    print("\nAudit rejection reasons:")
    print(
        probabilities["rejection_reason"]
        .value_counts()
        .to_string()
    )

    return output


if __name__ == "__main__":
    build_daily_card()
