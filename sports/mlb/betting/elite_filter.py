"""Apply a conservative, history-aware Elite filter to MLB prop candidates.

Place this file at:
    sports/mlb/betting/elite_filter.py

The filter does not promise a future hit rate. It makes the Elite label scarce,
requires strong live-line evidence, and only trusts historical segments with
enough graded observations.

Pitcher strikeouts use an additional strict gate designed to improve hit rate
by rejecting marginal plays before they reach the displayed card.

Inputs:
    A pandas DataFrame of candidate daily-card rows
    outputs/history/mlb_bet_results.csv
    outputs/calibration/elite_thresholds.csv (optional)

Output:
    The input DataFrame with updated confidence_tier/grade and Elite audit fields
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "outputs" / "history" / "mlb_bet_results.csv"
)

DEFAULT_THRESHOLDS_PATH = (
    PROJECT_ROOT / "outputs" / "calibration" / "elite_thresholds.csv"
)


TARGET_ELITE_HIT_RATE = 0.75
MINIMUM_ELITE_HISTORY_SAMPLE = 30
MINIMUM_ELITE_CALIBRATION_SAMPLE = 200


# ---------------------------------------------------------------------------
# PITCHER STRIKEOUT FILTERS
# ---------------------------------------------------------------------------

# These are minimums for strikeout Elite consideration. Candidates that fail
# these checks can still receive Strong, Good, or Playable tiers when their
# calibrated probability, edge, and expected value justify it.
STRIKEOUT_MINIMUM_OVER_PROBABILITY = 0.70
STRIKEOUT_MINIMUM_UNDER_PROBABILITY = 0.72
STRIKEOUT_MINIMUM_PROBABILITY_EDGE = 0.08
STRIKEOUT_MINIMUM_ABS_PROJECTION_EDGE = 0.65

# A 3.5 strikeout line has performed less reliably in the current sample, so it
# requires stronger evidence.
STRIKEOUT_35_MINIMUM_PROBABILITY = 0.74
STRIKEOUT_35_MINIMUM_PROBABILITY_EDGE = 0.12
STRIKEOUT_35_MINIMUM_ABS_PROJECTION_EDGE = 0.90

# Elite strikeouts must be even more selective.
STRIKEOUT_ELITE_OVER_PROBABILITY = 0.76
STRIKEOUT_ELITE_UNDER_PROBABILITY = 0.78
STRIKEOUT_ELITE_PROBABILITY_EDGE = 0.14
STRIKEOUT_ELITE_MINIMUM_ABS_PROJECTION_EDGE = 0.90


# Market-specific history standards.
MARKET_HISTORY_RULES: dict[str, dict[str, float]] = {
    "pitcher_strikeouts": {
        "minimum_sample": 20,
        "target_hit_rate": 0.66,
        "minimum_lower_bound": 0.55,
    },
}


ELITE_DISTRIBUTION_METHODS: dict[str, set[str]] = {
    "pitcher_strikeouts": {
        "poisson",
        "empirical_holdout_residuals",
    },
    "pitcher_outs": {
        "normal_residual_std",
        "empirical_holdout_residuals",
    },
}


# These are strict starting requirements. A nightly recalibration file can make
# the requirements stricter, but never weaker than these defaults.
DEFAULT_MARKET_RULES: dict[str, dict[str, float]] = {
    "hitter_hits": {
        "probability": 0.76,
        "probability_edge": 0.10,
        "expected_value": 0.12,
        "max_abs_projection_edge": 1.10,
        "max_validation_mae": 0.95,
    },
    "hitter_total_bases": {
        "probability": 0.77,
        "probability_edge": 0.11,
        "expected_value": 0.13,
        "max_abs_projection_edge": 2.25,
        "max_validation_mae": 1.65,
    },
    "hitter_runs": {
        "probability": 0.78,
        "probability_edge": 0.12,
        "expected_value": 0.14,
        "max_abs_projection_edge": 0.95,
        "max_validation_mae": 0.65,
    },
    "hitter_rbis": {
        "probability": 0.79,
        "probability_edge": 0.13,
        "expected_value": 0.15,
        "max_abs_projection_edge": 1.10,
        "max_validation_mae": 0.80,
    },
    "hitter_hits_runs_rbis": {
        "probability": 0.78,
        "probability_edge": 0.12,
        "expected_value": 0.14,
        "max_abs_projection_edge": 2.60,
        "max_validation_mae": 2.10,
    },
    "hitter_fantasy_score": {
        "probability": 0.78,
        "probability_edge": 0.12,
        "expected_value": 0.14,
        "max_abs_projection_edge": 8.50,
        "max_validation_mae": 6.25,
    },

    # These thresholds determine Elite eligibility after a strikeout candidate
    # has already passed the strict hard gate.
    "pitcher_strikeouts": {
        "probability": STRIKEOUT_ELITE_OVER_PROBABILITY,
        "probability_edge": STRIKEOUT_ELITE_PROBABILITY_EDGE,
        "expected_value": 0.08,
        "max_abs_projection_edge": 4.50,
        "max_validation_mae": 1.80,
    },

    "pitcher_outs": {
        "probability": 0.77,
        "probability_edge": 0.11,
        "expected_value": 0.13,
        "max_abs_projection_edge": 4.50,
        "max_validation_mae": 2.75,
    },
}


def _numeric(value: Any) -> float:
    """Convert a value to a finite float or return NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")

    return number if np.isfinite(number) else float("nan")


def _wilson_lower_bound(
    wins: int,
    sample_size: int,
    z: float = 1.2815515655,
) -> float:
    """Return an 80% one-sided Wilson lower confidence bound."""
    if sample_size <= 0:
        return float("nan")

    p_hat = wins / sample_size
    denominator = 1.0 + (z * z / sample_size)
    center = p_hat + (z * z / (2.0 * sample_size))

    margin = z * math.sqrt(
        (p_hat * (1.0 - p_hat) / sample_size)
        + (z * z / (4.0 * sample_size * sample_size))
    )

    return float((center - margin) / denominator)


def load_history(
    path: Path = DEFAULT_HISTORY_PATH,
) -> pd.DataFrame:
    """Load settled MLB prop history."""
    if not path.exists():
        return pd.DataFrame()

    try:
        history = pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return pd.DataFrame()

    required = {"market", "direction", "outcome"}

    if not required.issubset(history.columns):
        return pd.DataFrame()

    history["market"] = (
        history["market"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    history["direction"] = (
        history["direction"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    history["outcome"] = (
        history["outcome"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    history = history.loc[
        history["outcome"].isin({"WIN", "LOSS"})
    ].copy()

    for column in [
        "probability",
        "probability_edge",
        "expected_value",
    ]:
        if column in history.columns:
            history[column] = pd.to_numeric(
                history[column],
                errors="coerce",
            )

    return history


def load_thresholds(
    path: Path = DEFAULT_THRESHOLDS_PATH,
) -> pd.DataFrame:
    """Load optional recalibrated Elite thresholds."""
    if not path.exists():
        return pd.DataFrame()

    try:
        thresholds = pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return pd.DataFrame()

    if "market" not in thresholds.columns:
        return pd.DataFrame()

    thresholds["market"] = (
        thresholds["market"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return thresholds


def _market_rule(
    market: str,
    thresholds: pd.DataFrame,
) -> dict[str, float]:
    """Return the strictest applicable Elite rule for a market."""
    rule = dict(
        DEFAULT_MARKET_RULES.get(
            market,
            {
                "probability": 0.80,
                "probability_edge": 0.14,
                "expected_value": 0.16,
                "max_abs_projection_edge": 2.0,
                "max_validation_mae": 1.5,
            },
        )
    )

    if thresholds.empty:
        return rule

    row = thresholds.loc[
        thresholds["market"].eq(market)
    ]

    if row.empty:
        return rule

    record = row.iloc[-1]

    # Recalibration may tighten thresholds, but never weaken defaults.
    for output_key, csv_key in [
        (
            "probability",
            "recommended_probability_threshold",
        ),
        (
            "probability_edge",
            "recommended_probability_edge_threshold",
        ),
        (
            "expected_value",
            "recommended_expected_value_threshold",
        ),
    ]:
        value = _numeric(record.get(csv_key))

        if np.isfinite(value):
            rule[output_key] = max(
                rule[output_key],
                value,
            )

    return rule


def _segment_history(
    history: pd.DataFrame,
    market: str,
    direction: str,
    probability: float,
) -> tuple[int, int, float, float]:
    """Return settled history for a similar market/direction segment."""
    if history.empty:
        return 0, 0, float("nan"), float("nan")

    segment = history.loc[
        history["market"].eq(market)
        & history["direction"].eq(direction)
    ].copy()

    if (
        "probability" in segment.columns
        and np.isfinite(probability)
    ):
        # Compare candidates with similarly strong or stronger historical plays.
        minimum_probability = max(
            0.55,
            probability - 0.03,
        )

        segment = segment.loc[
            segment["probability"].ge(minimum_probability)
        ]

    sample_size = len(segment)
    wins = int(segment["outcome"].eq("WIN").sum())

    win_rate = (
        wins / sample_size
        if sample_size
        else float("nan")
    )

    lower_bound = _wilson_lower_bound(
        wins,
        sample_size,
    )

    return (
        sample_size,
        wins,
        win_rate,
        lower_bound,
    )


def _base_tier(
    probability: float,
    edge: float,
    expected_value: float,
) -> str:
    """Assign a non-Elite tier to an eligible candidate."""
    if (
        probability >= 0.67
        and edge >= 0.07
        and expected_value >= 0.08
    ):
        return "Strong"

    if (
        probability >= 0.61
        and edge >= 0.045
        and expected_value >= 0.045
    ):
        return "Good"

    if (
        probability >= 0.56
        and edge >= 0.025
        and expected_value >= 0.015
    ):
        return "Playable"

    return "PASS"


def _strikeout_hard_gate(
    direction: str,
    probability: float,
    probability_edge: float,
    projection: float,
    line: float,
) -> tuple[bool, list[str], float]:
    """Apply strict minimum requirements to pitcher strikeout candidates.

    A candidate that fails this gate cannot receive Elite status, but it may
    still receive a normal tier from calibrated probability, edge, and EV.
    """
    reasons: list[str] = []

    if not np.isfinite(projection) or not np.isfinite(line):
        return (
            False,
            ["strikeout_missing_projection_or_line"],
            float("nan"),
        )

    absolute_projection_edge = abs(projection - line)

    normalized_direction = direction.strip().title()

    if normalized_direction == "Under":
        minimum_probability = (
            STRIKEOUT_MINIMUM_UNDER_PROBABILITY
        )
    else:
        minimum_probability = (
            STRIKEOUT_MINIMUM_OVER_PROBABILITY
        )

    if (
        not np.isfinite(probability)
        or probability < minimum_probability
    ):
        reasons.append(
            "strikeout_probability_below_hard_gate"
        )

    if (
        not np.isfinite(probability_edge)
        or probability_edge
        < STRIKEOUT_MINIMUM_PROBABILITY_EDGE
    ):
        reasons.append(
            "strikeout_probability_edge_below_hard_gate"
        )

    if (
        absolute_projection_edge
        < STRIKEOUT_MINIMUM_ABS_PROJECTION_EDGE
    ):
        reasons.append(
            "strikeout_projection_edge_below_hard_gate"
        )

    # Lines of exactly 3.5 must meet stronger requirements.
    if math.isclose(line, 3.5, abs_tol=1e-9):
        if (
            not np.isfinite(probability)
            or probability
            < STRIKEOUT_35_MINIMUM_PROBABILITY
        ):
            reasons.append(
                "strikeout_3_5_probability_too_low"
            )

        if (
            not np.isfinite(probability_edge)
            or probability_edge
            < STRIKEOUT_35_MINIMUM_PROBABILITY_EDGE
        ):
            reasons.append(
                "strikeout_3_5_probability_edge_too_low"
            )

        if (
            absolute_projection_edge
            < STRIKEOUT_35_MINIMUM_ABS_PROJECTION_EDGE
        ):
            reasons.append(
                "strikeout_3_5_projection_edge_too_low"
            )

    passed = len(reasons) == 0

    return (
        passed,
        reasons,
        absolute_projection_edge,
    )


def _strikeout_elite_requirements(
    direction: str,
    probability: float,
    probability_edge: float,
    absolute_projection_edge: float,
) -> list[str]:
    """Return additional rejection reasons for Elite strikeout status."""
    reasons: list[str] = []

    normalized_direction = direction.strip().title()

    if normalized_direction == "Under":
        minimum_probability = (
            STRIKEOUT_ELITE_UNDER_PROBABILITY
        )
    else:
        minimum_probability = (
            STRIKEOUT_ELITE_OVER_PROBABILITY
        )

    if (
        not np.isfinite(probability)
        or probability < minimum_probability
    ):
        reasons.append(
            "strikeout_probability_below_elite_gate"
        )

    if (
        not np.isfinite(probability_edge)
        or probability_edge
        < STRIKEOUT_ELITE_PROBABILITY_EDGE
    ):
        reasons.append(
            "strikeout_probability_edge_below_elite_gate"
        )

    if (
        not np.isfinite(absolute_projection_edge)
        or absolute_projection_edge
        < STRIKEOUT_ELITE_MINIMUM_ABS_PROJECTION_EDGE
    ):
        reasons.append(
            "strikeout_projection_edge_below_elite_gate"
        )

    return reasons


def apply_elite_filter(
    frame: pd.DataFrame,
    history_path: Path = DEFAULT_HISTORY_PATH,
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
) -> pd.DataFrame:
    """Return candidates with conservative Elite decisions and audit columns."""
    if frame.empty:
        return frame.copy()

    result = frame.copy()

    history = load_history(history_path)
    thresholds = load_thresholds(thresholds_path)

    audit_rows: list[dict[str, Any]] = []

    for index, row in result.iterrows():
        market = str(
            row.get("market", "")
        ).strip().lower()

        direction = str(
            row.get("direction", "")
        ).strip().title()

        probability = _numeric(
            row.get("probability")
        )

        edge = _numeric(
            row.get("probability_edge")
        )

        expected_value = _numeric(
            row.get("expected_value")
        )

        projection = _numeric(
            row.get("projection")
        )

        line = _numeric(
            row.get("line")
        )

        calibration_sample = _numeric(
            row.get("calibration_sample_size")
        )

        validation_mae = _numeric(
            row.get("validation_mae")
        )

        distribution_method = str(
            row.get("distribution_method", "")
        ).strip()

        rule = _market_rule(
            market,
            thresholds,
        )

        (
            sample,
            wins,
            historical_rate,
            lower_bound,
        ) = _segment_history(
            history,
            market,
            direction,
            probability,
        )

        reasons: list[str] = []

        strikeout_gate_passed = True
        strikeout_gate_reasons: list[str] = []

        # ------------------------------------------------------------------
        # Strict strikeout gate
        # ------------------------------------------------------------------
        if market == "pitcher_strikeouts":
            (
                strikeout_gate_passed,
                strikeout_gate_reasons,
                absolute_projection_edge,
            ) = _strikeout_hard_gate(
                direction=direction,
                probability=probability,
                probability_edge=edge,
                projection=projection,
                line=line,
            )

            reasons.extend(
                strikeout_gate_reasons
            )
        else:
            if (
                not np.isfinite(projection)
                or not np.isfinite(line)
            ):
                absolute_projection_edge = float("nan")
            else:
                absolute_projection_edge = abs(
                    projection - line
                )

        # ------------------------------------------------------------------
        # Standard Elite thresholds
        # ------------------------------------------------------------------
        if (
            not np.isfinite(probability)
            or probability < rule["probability"]
        ):
            reasons.append(
                "probability_below_elite_threshold"
            )

        if (
            not np.isfinite(edge)
            or edge < rule["probability_edge"]
        ):
            reasons.append(
                "edge_below_elite_threshold"
            )

        if (
            not np.isfinite(expected_value)
            or expected_value < rule["expected_value"]
        ):
            reasons.append(
                "ev_below_elite_threshold"
            )

        if (
            not np.isfinite(projection)
            or not np.isfinite(line)
        ):
            reasons.append(
                "missing_projection_or_line"
            )
        elif (
            absolute_projection_edge
            > rule["max_abs_projection_edge"]
        ):
            reasons.append(
                "projection_sanity_rejection"
            )

        # Additional Elite-only requirements for pitcher strikeouts.
        if (
            market == "pitcher_strikeouts"
            and strikeout_gate_passed
        ):
            reasons.extend(
                _strikeout_elite_requirements(
                    direction=direction,
                    probability=probability,
                    probability_edge=edge,
                    absolute_projection_edge=(
                        absolute_projection_edge
                    ),
                )
            )

        allowed_methods = (
            ELITE_DISTRIBUTION_METHODS.get(
                market,
                {"empirical_holdout_residuals"},
            )
        )

        if distribution_method not in allowed_methods:
            reasons.append(
                "distribution_not_elite_eligible"
            )

        elif (
            distribution_method
            == "empirical_holdout_residuals"
        ):
            if (
                not np.isfinite(calibration_sample)
                or calibration_sample
                < MINIMUM_ELITE_CALIBRATION_SAMPLE
            ):
                reasons.append(
                    "calibration_sample_too_small"
                )

        if (
            np.isfinite(validation_mae)
            and validation_mae
            > rule["max_validation_mae"]
        ):
            reasons.append(
                "validation_error_too_high"
            )

        history_rule = MARKET_HISTORY_RULES.get(
            market,
            {
                "minimum_sample": (
                    MINIMUM_ELITE_HISTORY_SAMPLE
                ),
                "target_hit_rate": (
                    TARGET_ELITE_HIT_RATE
                ),
                "minimum_lower_bound": 0.65,
            },
        )

        history_proven = (
            sample
            >= int(history_rule["minimum_sample"])
            and np.isfinite(historical_rate)
            and historical_rate
            >= history_rule["target_hit_rate"]
            and np.isfinite(lower_bound)
            and lower_bound
            >= history_rule["minimum_lower_bound"]
        )

        if not history_proven:
            reasons.append(
                "historical_segment_not_proven"
            )

        elite = (
            strikeout_gate_passed
            and len(reasons) == 0
        )

        serious_rejection_reasons = {
            "missing_projection_or_line",
            "projection_sanity_rejection",
            "calibration_sample_too_small",
            "distribution_not_elite_eligible",
            "validation_error_too_high",
        }

        has_serious_rejection = any(
            reason in serious_rejection_reasons
            for reason in reasons
        )

        # ------------------------------------------------------------------
        # Final tier decision
        # ------------------------------------------------------------------
        if elite:
            tier = "Elite"

        elif has_serious_rejection:
            tier = "Playable"

        else:
            # A failed strikeout hard gate now blocks only Elite status.
            # It no longer forces an otherwise positive-EV candidate to PASS.
            tier = _base_tier(
                probability,
                edge,
                expected_value,
            )

        grade = {
            "Elite": "A+",
            "Strong": "A",
            "Good": "B+",
            "Playable": "B",
            "PASS": "PASS",
        }[tier]

        # Transparent ranking score. This is not a probability claim.
        score_components = [
            (
                np.clip(
                    (probability - 0.50) / 0.35,
                    0.0,
                    1.0,
                )
                if np.isfinite(probability)
                else 0.0
            ),
            (
                np.clip(
                    edge / 0.18,
                    0.0,
                    1.0,
                )
                if np.isfinite(edge)
                else 0.0
            ),
            (
                np.clip(
                    expected_value / 0.22,
                    0.0,
                    1.0,
                )
                if np.isfinite(expected_value)
                else 0.0
            ),
            (
                np.clip(
                    (historical_rate - 0.50) / 0.30,
                    0.0,
                    1.0,
                )
                if np.isfinite(historical_rate)
                else 0.0
            ),
            np.clip(
                sample / 100.0,
                0.0,
                1.0,
            ),
        ]

        elite_score = round(
            100.0
            * (
                0.30 * score_components[0]
                + 0.20 * score_components[1]
                + 0.15 * score_components[2]
                + 0.25 * score_components[3]
                + 0.10 * score_components[4]
            ),
            2,
        )

        audit_rows.append(
            {
                "_index": index,
                "confidence_tier": tier,
                "grade": grade,
                "elite_score": elite_score,
                "elite_eligible": elite,
                "elite_rejection_reasons": (
                    ""
                    if elite
                    else "|".join(
                        dict.fromkeys(reasons)
                    )
                ),
                "elite_history_sample": sample,
                "elite_history_wins": wins,
                "elite_history_win_rate": historical_rate,
                "elite_history_lower_bound": lower_bound,
                "elite_probability_threshold": (
                    rule["probability"]
                ),
                "elite_probability_edge_threshold": (
                    rule["probability_edge"]
                ),
                "elite_expected_value_threshold": (
                    rule["expected_value"]
                ),
                "elite_absolute_projection_edge": (
                    absolute_projection_edge
                ),
                "strikeout_hard_gate_passed": (
                    strikeout_gate_passed
                    if market == "pitcher_strikeouts"
                    else np.nan
                ),
                "strikeout_hard_gate_reasons": (
                    "|".join(strikeout_gate_reasons)
                    if market == "pitcher_strikeouts"
                    else ""
                ),
            }
        )

    audit = (
        pd.DataFrame(audit_rows)
        .set_index("_index")
    )

    for column in audit.columns:
        result.loc[
            audit.index,
            column,
        ] = audit[column]

    return result
