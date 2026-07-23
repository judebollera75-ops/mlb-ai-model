"""Calculate probabilities for current MLB player-prop lines.

Inputs:
    data/platform_lines.csv
    outputs/hitters/today_hitter_projections.csv
    outputs/calibrated_strikeout_projections.csv
    models/hitters/<market>_model.pkl

Output:
    outputs/probability_table.csv

Hitter and pitcher-strikeout probabilities are estimated from chronological
holdout residuals saved during model training. Poisson remains available only
as a safety fallback when the strikeout residual sample is too small.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson


PROJECT_ROOT = Path(__file__).resolve().parents[4]

PLATFORM_LINES_PATH = PROJECT_ROOT / "data" / "platform_lines.csv"

HITTER_PROJECTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "hitters"
    / "today_hitter_projections.csv"
)

PITCHER_STRIKEOUT_PROJECTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "calibrated_strikeout_projections.csv"
)

PITCHER_OUTS_PROJECTION_PATHS = [
    PROJECT_ROOT / "outputs" / "pitcher_outs_projections.csv",
    PROJECT_ROOT / "outputs" / "pitcher_outs.csv",
]

HITTER_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "hitters"
)

PITCHER_STRIKEOUT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "leakage_free_strikeout_model.pkl"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "probability_table.csv"

MINIMUM_EMPIRICAL_RESIDUALS = 100
EMPIRICAL_PRIOR_STRENGTH = 40.0
PROBABILITY_FLOOR = 0.08
PROBABILITY_CEILING = 0.92
PUSH_TOLERANCE = 1e-9
MODEL_VERSION = "2.2.0"
INTERVAL_Z_80 = 1.2815515655446004


PICKEM_PLATFORMS = {
    "prizepicks",
    "sleeper",
    "underdog",
}

# Shrink raw distribution probabilities toward 50% until historical
# calibration proves the model deserves full confidence.
METHOD_SHRINKAGE = {
    "empirical_holdout_residuals": 0.84,
    "empirical_pitcher_holdout_residuals": 0.88,
    "poisson": 0.70,
    "poisson_fallback": 0.68,
    "normal_residual_std": 0.74,
}

HITTER_MARKET_CONFIG = {
    "hitter_hits": {
        "projection_column": "projected_hits",
        "bundle_market": "hits",
    },
    "hitter_total_bases": {
        "projection_column": "projected_total_bases",
        "bundle_market": "total_bases",
    },
    "hitter_runs": {
        "projection_column": "projected_runs",
        "bundle_market": "runs",
    },
    "hitter_rbis": {
        "projection_column": "projected_rbi",
        "bundle_market": "rbi",
    },
    "hitter_hits_runs_rbis": {
        "projection_column": "projected_hits_runs_rbis",
        "bundle_market": "hits_runs_rbis",
    },
    "hitter_fantasy_score": {
        "projection_column": "projected_fantasy_score",
        "bundle_market": "fantasy_score",
    },
}

OUTPUT_COLUMNS = [
    "event_id",
    "event_date",
    "commence_time",
    "platform",
    "platform_key",
    "player",
    "market",
    "direction",
    "line",
    "sportsbook_odds",
    "projection",
    "raw_probability",
    "probability",
    "calibration_shrinkage",
    "push_probability",
    "opposite_probability",
    "fair_odds",
    "distribution_method",
    "calibration_sample_size",
    "validation_mae",
    "probability_status",
    "probability_note",
    "projection_std",
    "projection_floor",
    "projection_ceiling",
    "consensus_line",
    "consensus_difference",
    "sportsbook_count",
    "line_type",
    "market_quality_score",
    "volatility_score",
    "projection_quality_score",
    "confidence_score",
    "risk_score",
    "model_version",
    "fetched_at",
]


def normalize_player_name(value: Any) -> str:
    """Normalize player names for cross-source matching."""
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clamp_probability(probability: float) -> float:
    """Prevent impossible fair odds and unstable downstream calculations."""
    if not np.isfinite(probability):
        return float("nan")

    return float(
        np.clip(
            probability,
            PROBABILITY_FLOOR,
            PROBABILITY_CEILING,
        )
    )



def is_pickem_platform(value: Any) -> bool:
    """Return True for DFS/pick'em operators without standard bet odds."""
    return str(value).strip().casefold() in PICKEM_PLATFORMS


def calibrate_probability(
    raw_probability: float,
    distribution_method: str,
    calibration_sample_size: Any,
) -> tuple[float, float]:
    """Shrink unproven probabilities toward 50% to reduce overconfidence."""
    if not np.isfinite(raw_probability):
        return float("nan"), float("nan")

    base_shrinkage = METHOD_SHRINKAGE.get(
        distribution_method,
        0.65,
    )

    sample_size = pd.to_numeric(
        calibration_sample_size,
        errors="coerce",
    )

    # Empirical hitter models earn slightly more trust with large holdouts.
    if (
        distribution_method in {"empirical_holdout_residuals", "empirical_pitcher_holdout_residuals"}
        and pd.notna(sample_size)
    ):
        sample_factor = float(
            np.clip(
                float(sample_size) / 750.0,
                0.70,
                1.0,
            )
        )
        shrinkage = base_shrinkage * sample_factor
    else:
        shrinkage = base_shrinkage

    calibrated = 0.5 + (
        float(raw_probability) - 0.5
    ) * shrinkage

    return clamp_probability(calibrated), float(shrinkage)


def fair_odds(probability: float) -> int | None:
    """Convert probability to fair American odds."""
    if not np.isfinite(probability):
        return None

    probability = clamp_probability(probability)

    if probability >= 0.5:
        return int(
            round(
                -100.0
                * probability
                / (1.0 - probability)
            )
        )

    return int(
        round(
            100.0
            * (1.0 - probability)
            / probability
        )
    )


def american_odds_to_probability(odds: Any) -> float:
    """Convert American odds into raw implied probability."""
    try:
        numeric_odds = float(odds)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(numeric_odds) or numeric_odds == 0:
        return float("nan")

    if numeric_odds > 0:
        return 100.0 / (numeric_odds + 100.0)

    return abs(numeric_odds) / (abs(numeric_odds) + 100.0)


def normalize_direction(value: Any) -> str:
    """Normalize prop sides."""
    cleaned = str(value).strip().casefold()

    if cleaned in {"over", "more"}:
        return "Over"

    if cleaned in {"under", "less"}:
        return "Under"

    if cleaned == "yes":
        return "Yes"

    if cleaned == "no":
        return "No"

    return str(value).strip().title()


def poisson_probabilities(
    projection: float,
    line: float,
) -> tuple[float, float, float]:
    """Return over, under, and push probabilities for a count market."""
    if (
        not np.isfinite(projection)
        or projection < 0
        or not np.isfinite(line)
    ):
        return float("nan"), float("nan"), float("nan")

    integer_line = math.isclose(
        line,
        round(line),
        abs_tol=PUSH_TOLERANCE,
    )

    if integer_line:
        threshold = int(round(line))

        under_probability = float(
            poisson.cdf(threshold - 1, projection)
        )

        push_probability = float(
            poisson.pmf(threshold, projection)
        )

        over_probability = float(
            1.0
            - under_probability
            - push_probability
        )
    else:
        floor_line = math.floor(line)

        under_probability = float(
            poisson.cdf(floor_line, projection)
        )

        over_probability = float(
            1.0 - under_probability
        )

        push_probability = 0.0

    return (
        max(0.0, over_probability),
        max(0.0, under_probability),
        max(0.0, push_probability),
    )


def normal_probabilities(
    projection: float,
    line: float,
    standard_deviation: float,
) -> tuple[float, float, float]:
    """Return continuous-distribution probabilities."""
    if (
        not np.isfinite(projection)
        or not np.isfinite(line)
        or not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        return float("nan"), float("nan"), float("nan")

    under_probability = float(
        norm.cdf(
            line,
            loc=projection,
            scale=standard_deviation,
        )
    )

    over_probability = float(
        1.0 - under_probability
    )

    return over_probability, under_probability, 0.0


def smooth_empirical_probability(
    successes: int,
    sample_size: int,
) -> float:
    """Apply a weak neutral prior to empirical probabilities."""
    if sample_size <= 0:
        return float("nan")

    prior_successes = 0.5 * EMPIRICAL_PRIOR_STRENGTH

    return float(
        (
            successes
            + prior_successes
        )
        / (
            sample_size
            + EMPIRICAL_PRIOR_STRENGTH
        )
    )


def empirical_probabilities(
    projection: float,
    line: float,
    residuals: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate outcomes from out-of-sample model residuals."""
    if not np.isfinite(projection) or not np.isfinite(line):
        return float("nan"), float("nan"), float("nan")

    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    residuals = residuals[
        np.isfinite(residuals)
    ]

    if len(residuals) < MINIMUM_EMPIRICAL_RESIDUALS:
        return float("nan"), float("nan"), float("nan")

    simulated_outcomes = projection + residuals

    over_count = int(
        np.sum(simulated_outcomes > line + PUSH_TOLERANCE)
    )

    under_count = int(
        np.sum(simulated_outcomes < line - PUSH_TOLERANCE)
    )

    push_count = int(
        len(simulated_outcomes)
        - over_count
        - under_count
    )

    over_probability = smooth_empirical_probability(
        over_count,
        len(simulated_outcomes),
    )

    under_probability = smooth_empirical_probability(
        under_count,
        len(simulated_outcomes),
    )

    push_probability = float(
        push_count / len(simulated_outcomes)
    )

    total_probability = (
        over_probability
        + under_probability
        + push_probability
    )

    if total_probability > 0:
        over_probability /= total_probability
        under_probability /= total_probability
        push_probability /= total_probability

    return (
        over_probability,
        under_probability,
        push_probability,
    )


def load_hitter_bundle(
    bundle_market: str,
) -> dict[str, Any]:
    """Load one trained hitter-model bundle."""
    path = (
        HITTER_MODEL_DIRECTORY
        / f"{bundle_market}_model.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Hitter model bundle was not found: {path}"
        )

    bundle = joblib.load(path)

    if "holdout_residuals" not in bundle:
        raise ValueError(
            f"{path} does not contain holdout residuals. "
            "Retrain hitter models before calculating probabilities."
        )

    return bundle


def load_pitcher_strikeout_bundle() -> dict[str, Any]:
    """Load the trained leakage-safe pitcher strikeout bundle."""
    if not PITCHER_STRIKEOUT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Pitcher strikeout model bundle was not found: "
            f"{PITCHER_STRIKEOUT_MODEL_PATH}"
        )

    bundle = joblib.load(PITCHER_STRIKEOUT_MODEL_PATH)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Pitcher strikeout model file does not contain a model bundle."
        )

    if "holdout_residuals" not in bundle:
        raise ValueError(
            "Pitcher strikeout model bundle does not contain "
            "holdout_residuals. Retrain the leakage-safe strikeout model."
        )

    return bundle


def load_platform_lines() -> pd.DataFrame:
    """Load exact current platform lines."""
    if not PLATFORM_LINES_PATH.exists():
        raise FileNotFoundError(
            f"Platform lines were not found: {PLATFORM_LINES_PATH}"
        )

    try:
        lines = pd.read_csv(PLATFORM_LINES_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read platform lines: {PLATFORM_LINES_PATH}"
        ) from exc

    required_columns = {
        "player",
        "market",
        "direction",
        "line",
        "platform",
    }

    missing_columns = required_columns - set(lines.columns)

    if missing_columns:
        raise ValueError(
            "Platform lines are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    lines["line"] = pd.to_numeric(
        lines["line"],
        errors="coerce",
    )

    lines["sportsbook_odds"] = pd.to_numeric(
        lines.get("sportsbook_odds"),
        errors="coerce",
    )

    lines["market"] = (
        lines["market"]
        .astype(str)
        .str.strip()
    )

    lines["direction"] = lines[
        "direction"
    ].apply(normalize_direction)

    lines["normalized_player"] = lines[
        "player"
    ].apply(normalize_player_name)

    lines = lines.dropna(
        subset=[
            "player",
            "market",
            "direction",
            "line",
            "platform",
        ]
    ).copy()

    lines = lines.loc[
        lines["normalized_player"].ne("")
    ].copy()

    return lines


def build_hitter_projection_lookup() -> pd.DataFrame:
    """Convert current hitter projections to long market format."""
    if not HITTER_PROJECTIONS_PATH.exists():
        print(
            "No current hitter projections were found. "
            "Hitter probabilities will be skipped."
        )

        return pd.DataFrame(
            columns=[
                "player",
                "normalized_player",
                "market",
                "projection",
                "residual_standard_deviation",
            ]
        )

    projections = pd.read_csv(HITTER_PROJECTIONS_PATH)

    if "player_name" not in projections.columns:
        raise ValueError(
            "Hitter projections do not contain player_name."
        )

    rows: list[dict[str, Any]] = []

    for market, config in HITTER_MARKET_CONFIG.items():
        projection_column = config["projection_column"]

        if projection_column not in projections.columns:
            print(
                f"Skipping {market}: missing "
                f"{projection_column}."
            )
            continue

        residual_std_column = (
            f"{projection_column}_residual_std"
        )

        for _, row in projections.iterrows():
            player = row.get("player_name")
            projection = pd.to_numeric(
                row.get(projection_column),
                errors="coerce",
            )

            if pd.isna(player) or pd.isna(projection):
                continue

            residual_standard_deviation = pd.to_numeric(
                row.get(residual_std_column),
                errors="coerce",
            )

            rows.append(
                {
                    "player": str(player).strip(),
                    "normalized_player": (
                        normalize_player_name(player)
                    ),
                    "market": market,
                    "projection": float(projection),
                    "residual_standard_deviation": (
                        float(residual_standard_deviation)
                        if pd.notna(residual_standard_deviation)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def find_first_existing_path(
    paths: list[Path],
) -> Path | None:
    """Return the first available path."""
    for path in paths:
        if path.exists():
            return path

    return None


def build_pitcher_strikeout_lookup() -> pd.DataFrame:
    """Load calibrated pitcher-strikeout projections."""
    if not PITCHER_STRIKEOUT_PROJECTIONS_PATH.exists():
        print(
            "No calibrated pitcher strikeout projections were found."
        )

        return pd.DataFrame(
            columns=[
                "player",
                "normalized_player",
                "market",
                "projection",
            ]
        )

    projections = pd.read_csv(
        PITCHER_STRIKEOUT_PROJECTIONS_PATH
    )

    player_column = next(
        (
            column
            for column in [
                "pitcher_name",
                "player_name",
                "pitcher",
            ]
            if column in projections.columns
        ),
        None,
    )

    projection_column = next(
        (
            column
            for column in [
                "calibrated_projected_ks",
                "projected_strikeouts",
                "projected_ks",
            ]
            if column in projections.columns
        ),
        None,
    )

    if player_column is None or projection_column is None:
        print(
            "Pitcher strikeout projection schema was not recognized."
        )

        return pd.DataFrame(
            columns=[
                "player",
                "normalized_player",
                "market",
                "projection",
            ]
        )

    lookup = projections[
        [player_column, projection_column]
    ].copy()

    lookup = lookup.rename(
        columns={
            player_column: "player",
            projection_column: "projection",
        }
    )

    lookup["projection"] = pd.to_numeric(
        lookup["projection"],
        errors="coerce",
    )

    lookup = lookup.dropna(
        subset=["player", "projection"]
    )

    lookup["normalized_player"] = lookup[
        "player"
    ].apply(normalize_player_name)

    lookup["market"] = "pitcher_strikeouts"

    return lookup[
        [
            "player",
            "normalized_player",
            "market",
            "projection",
        ]
    ]


def build_pitcher_outs_lookup() -> pd.DataFrame:
    """Load pitcher-outs projections when a supported file exists."""
    path = find_first_existing_path(
        PITCHER_OUTS_PROJECTION_PATHS
    )

    if path is None:
        return pd.DataFrame(
            columns=[
                "player",
                "normalized_player",
                "market",
                "projection",
                "residual_standard_deviation",
            ]
        )

    projections = pd.read_csv(path)

    player_column = next(
        (
            column
            for column in [
                "pitcher_name",
                "player_name",
                "pitcher",
            ]
            if column in projections.columns
        ),
        None,
    )

    projection_column = next(
        (
            column
            for column in [
                "projected_outs",
                "calibrated_projected_outs",
                "projection",
            ]
            if column in projections.columns
        ),
        None,
    )

    standard_deviation_column = next(
        (
            column
            for column in [
                "projected_outs_residual_std",
                "residual_std",
                "projection_std",
            ]
            if column in projections.columns
        ),
        None,
    )

    if (
        player_column is None
        or projection_column is None
    ):
        return pd.DataFrame(
            columns=[
                "player",
                "normalized_player",
                "market",
                "projection",
                "residual_standard_deviation",
            ]
        )

    lookup = pd.DataFrame(
        {
            "player": projections[player_column],
            "projection": pd.to_numeric(
                projections[projection_column],
                errors="coerce",
            ),
        }
    )

    if standard_deviation_column:
        lookup["residual_standard_deviation"] = (
            pd.to_numeric(
                projections[standard_deviation_column],
                errors="coerce",
            )
        )
    else:
        lookup["residual_standard_deviation"] = np.nan

    lookup = lookup.dropna(
        subset=["player", "projection"]
    )

    lookup["normalized_player"] = lookup[
        "player"
    ].apply(normalize_player_name)

    lookup["market"] = "pitcher_outs"

    return lookup


def calculate_hitter_line_probability(
    row: pd.Series,
    bundle_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate one exact hitter-line probability."""
    market = row["market"]
    config = HITTER_MARKET_CONFIG[market]
    bundle_market = config["bundle_market"]

    if bundle_market not in bundle_cache:
        bundle_cache[bundle_market] = load_hitter_bundle(
            bundle_market
        )

    bundle = bundle_cache[bundle_market]

    residuals = pd.to_numeric(
        pd.Series(
            bundle.get("holdout_residuals", []),
            dtype="object",
        ),
        errors="coerce",
    ).dropna()

    residuals = residuals.loc[
        np.isfinite(residuals)
    ].to_numpy(dtype=float)

    sample_size = len(residuals)

    if sample_size < MINIMUM_EMPIRICAL_RESIDUALS:
        return {
            "over_probability": np.nan,
            "under_probability": np.nan,
            "push_probability": np.nan,
            "distribution_method": "unavailable",
            "calibration_sample_size": sample_size,
            "validation_mae": bundle.get(
                "validation_mae",
                np.nan,
            ),
            "probability_status": "rejected",
            "probability_note": (
                "Insufficient chronological holdout residuals."
            ),
            "projection_std": float(np.std(residuals, ddof=1)) if sample_size > 1 else np.nan,
        }

    (
        over_probability,
        under_probability,
        push_probability,
    ) = empirical_probabilities(
        projection=float(row["projection"]),
        line=float(row["line"]),
        residuals=residuals,
    )

    return {
        "over_probability": over_probability,
        "under_probability": under_probability,
        "push_probability": push_probability,
        "distribution_method": "empirical_holdout_residuals",
        "calibration_sample_size": sample_size,
        "validation_mae": bundle.get(
            "validation_mae",
            np.nan,
        ),
        "probability_status": "calculated",
        "probability_note": "",
        "projection_std": float(np.std(residuals, ddof=1)),
    }


def calculate_pitcher_line_probability(
    row: pd.Series,
) -> dict[str, Any]:
    """Calculate a pitcher market probability."""
    market = row["market"]

    if market == "pitcher_strikeouts":
        bundle = load_pitcher_strikeout_bundle()

        residuals = pd.to_numeric(
            pd.Series(
                bundle.get("holdout_residuals", []),
                dtype="object",
            ),
            errors="coerce",
        ).dropna()

        residuals = residuals.loc[
            np.isfinite(residuals)
        ].to_numpy(dtype=float)

        sample_size = len(residuals)
        validation_mae = pd.to_numeric(
            bundle.get("validation_mae"),
            errors="coerce",
        )

        if sample_size < MINIMUM_EMPIRICAL_RESIDUALS:
            (
                over_probability,
                under_probability,
                push_probability,
            ) = poisson_probabilities(
                float(row["projection"]),
                float(row["line"]),
            )

            return {
                "over_probability": over_probability,
                "under_probability": under_probability,
                "push_probability": push_probability,
                "distribution_method": "poisson_fallback",
                "calibration_sample_size": sample_size,
                "validation_mae": (
                    float(validation_mae)
                    if pd.notna(validation_mae)
                    else np.nan
                ),
                "probability_status": "calculated",
                "probability_note": (
                    "Poisson safety fallback because the pitcher "
                    "strikeout holdout residual sample is too small."
                ),
                "projection_std": float(
                    math.sqrt(
                        max(float(row["projection"]), 0.0)
                    )
                ),
            }

        (
            over_probability,
            under_probability,
            push_probability,
        ) = empirical_probabilities(
            projection=float(row["projection"]),
            line=float(row["line"]),
            residuals=residuals,
        )

        projection_std = (
            float(np.std(residuals, ddof=1))
            if sample_size > 1
            else pd.to_numeric(
                bundle.get("residual_std"),
                errors="coerce",
            )
        )

        return {
            "over_probability": over_probability,
            "under_probability": under_probability,
            "push_probability": push_probability,
            "distribution_method": (
                "empirical_pitcher_holdout_residuals"
            ),
            "calibration_sample_size": sample_size,
            "validation_mae": (
                float(validation_mae)
                if pd.notna(validation_mae)
                else np.nan
            ),
            "probability_status": "calculated",
            "probability_note": "",
            "projection_std": (
                float(projection_std)
                if pd.notna(projection_std)
                else np.nan
            ),
        }

    if market == "pitcher_outs":
        standard_deviation = pd.to_numeric(
            row.get("residual_standard_deviation"),
            errors="coerce",
        )

        if pd.isna(standard_deviation):
            return {
                "over_probability": np.nan,
                "under_probability": np.nan,
                "push_probability": np.nan,
                "distribution_method": "unavailable",
                "calibration_sample_size": np.nan,
                "validation_mae": np.nan,
                "probability_status": "rejected",
                "probability_note": (
                    "Pitcher-outs uncertainty data is unavailable."
                ),
                "projection_std": np.nan,
            }

        (
            over_probability,
            under_probability,
            push_probability,
        ) = normal_probabilities(
            float(row["projection"]),
            float(row["line"]),
            float(standard_deviation),
        )

        return {
            "over_probability": over_probability,
            "under_probability": under_probability,
            "push_probability": push_probability,
            "distribution_method": "normal_residual_std",
            "calibration_sample_size": np.nan,
            "validation_mae": np.nan,
            "probability_status": "calculated",
            "probability_note": "",
            "projection_std": float(standard_deviation),
        }

    return {
        "over_probability": np.nan,
        "under_probability": np.nan,
        "push_probability": np.nan,
        "distribution_method": "unsupported",
        "calibration_sample_size": np.nan,
        "validation_mae": np.nan,
        "probability_status": "rejected",
        "probability_note": "Unsupported pitcher market.",
        "projection_std": np.nan,
    }


def choose_side_probability(
    direction: str,
    over_probability: float,
    under_probability: float,
) -> tuple[float, float]:
    """Return selected-side and opposite-side probabilities."""
    if direction in {"Over", "Yes"}:
        return over_probability, under_probability

    if direction in {"Under", "No"}:
        return under_probability, over_probability

    return float("nan"), float("nan")


def build_probability_table() -> pd.DataFrame:
    """Match live lines to projections and calculate exact probabilities."""
    lines = load_platform_lines()

    hitter_lookup = build_hitter_projection_lookup()
    strikeout_lookup = build_pitcher_strikeout_lookup()
    outs_lookup = build_pitcher_outs_lookup()

    projection_lookup = pd.concat(
        [
            hitter_lookup,
            strikeout_lookup,
            outs_lookup,
        ],
        ignore_index=True,
        sort=False,
    )

    if projection_lookup.empty:
        raise RuntimeError(
            "No current hitter or pitcher projections were available."
        )

    projection_lookup = projection_lookup.drop_duplicates(
        subset=[
            "normalized_player",
            "market",
        ],
        keep="last",
    )

    matched = lines.merge(
        projection_lookup,
        on=[
            "normalized_player",
            "market",
        ],
        how="left",
        suffixes=("", "_projection"),
    )

    bundle_cache: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []

    for _, row in matched.iterrows():
        projection = pd.to_numeric(
            row.get("projection"),
            errors="coerce",
        )

        base_output = {
            column: row.get(column)
            for column in [
                "event_id",
                "event_date",
                "commence_time",
                "platform",
                "platform_key",
                "player",
                "market",
                "direction",
                "line",
                "sportsbook_odds",
                "fetched_at",
            ]
        }

        base_output["projection"] = (
            float(projection)
            if pd.notna(projection)
            else np.nan
        )

        if pd.isna(projection):
            base_output.update(
                {
                    "raw_probability": np.nan,
                    "probability": np.nan,
                    "calibration_shrinkage": np.nan,
                    "push_probability": np.nan,
                    "opposite_probability": np.nan,
                    "fair_odds": None,
                    "distribution_method": "unmatched",
                    "calibration_sample_size": np.nan,
                    "validation_mae": np.nan,
                    "probability_status": "rejected",
                    "probability_note": (
                        "No projection matched this player and market."
                    ),
                    "projection_std": np.nan,
                    "projection_floor": np.nan,
                    "projection_ceiling": np.nan,
                    "consensus_line": np.nan,
                    "consensus_difference": np.nan,
                    "sportsbook_count": 0,
                    "line_type": "unmatched",
                    "market_quality_score": 0.0,
                    "volatility_score": 0.0,
                    "projection_quality_score": 0.0,
                    "confidence_score": 0.0,
                    "risk_score": 100.0,
                    "model_version": MODEL_VERSION,
                }
            )

            output_rows.append(base_output)
            continue

        if row["market"] in HITTER_MARKET_CONFIG:
            probability_result = (
                calculate_hitter_line_probability(
                    row,
                    bundle_cache,
                )
            )
        else:
            probability_result = (
                calculate_pitcher_line_probability(row)
            )

        side_probability, opposite_probability = (
            choose_side_probability(
                direction=row["direction"],
                over_probability=probability_result[
                    "over_probability"
                ],
                under_probability=probability_result[
                    "under_probability"
                ],
            )
        )

        raw_side_probability = side_probability

        if np.isfinite(raw_side_probability):
            (
                side_probability,
                calibration_shrinkage,
            ) = calibrate_probability(
                raw_probability=raw_side_probability,
                distribution_method=probability_result[
                    "distribution_method"
                ],
                calibration_sample_size=probability_result[
                    "calibration_sample_size"
                ],
            )
        else:
            calibration_shrinkage = np.nan

        if np.isfinite(opposite_probability):
            opposite_probability = clamp_probability(
                1.0
                - side_probability
                - float(
                    probability_result[
                        "push_probability"
                    ]
                    if np.isfinite(
                        probability_result[
                            "push_probability"
                        ]
                    )
                    else 0.0
                )
            )

        base_output.update(
            {
                "raw_probability": raw_side_probability,
                "probability": side_probability,
                "calibration_shrinkage": calibration_shrinkage,
                "push_probability": probability_result[
                    "push_probability"
                ],
                "opposite_probability": opposite_probability,
                "fair_odds": fair_odds(side_probability),
                "distribution_method": probability_result[
                    "distribution_method"
                ],
                "calibration_sample_size": probability_result[
                    "calibration_sample_size"
                ],
                "validation_mae": probability_result[
                    "validation_mae"
                ],
                "probability_status": probability_result[
                    "probability_status"
                ],
                "probability_note": probability_result[
                    "probability_note"
                ],
                "projection_std": probability_result.get("projection_std", np.nan),
                "model_version": MODEL_VERSION,
            }
        )

        output_rows.append(base_output)

    output = pd.DataFrame(
        output_rows,
        columns=OUTPUT_COLUMNS,
    )

    # ------------------------------------------------------------------
    # V2.1 uncertainty, line intelligence, confidence, and risk metadata
    # ------------------------------------------------------------------
    output["projection_std"] = pd.to_numeric(
        output.get("projection_std"), errors="coerce"
    )
    output["projection_floor"] = (
        output["projection"] - INTERVAL_Z_80 * output["projection_std"]
    ).clip(lower=0)
    output["projection_ceiling"] = (
        output["projection"] + INTERVAL_Z_80 * output["projection_std"]
    ).clip(lower=0)

    output["_normalized_player_v2"] = output["player"].apply(normalize_player_name)
    consensus_group = ["_normalized_player_v2", "market", "direction"]
    output["consensus_line"] = output.groupby(consensus_group, dropna=False)["line"].transform("median")
    output["sportsbook_count"] = output.groupby(consensus_group, dropna=False)["platform"].transform("nunique")
    output["consensus_difference"] = output["line"] - output["consensus_line"]

    # A difference of at least one full unit is usually an alternate/stale line.
    output["line_type"] = np.select(
        [
            output["consensus_difference"].abs() >= 1.0,
            output["consensus_difference"].abs() >= 0.5,
        ],
        ["alternate", "off_consensus"],
        default="standard",
    )

    output["market_quality_score"] = (
        55.0
        + output["sportsbook_count"].clip(lower=0, upper=6) * 6.0
        - output["consensus_difference"].abs().clip(upper=2.0) * 18.0
    ).clip(lower=0, upper=100)

    relative_volatility = output["projection_std"] / output["projection"].replace(0, np.nan)
    output["volatility_score"] = (
        100.0 - relative_volatility.fillna(1.0).clip(lower=0, upper=1.5) * 55.0
    ).clip(lower=0, upper=100)

    projection_gap = (output["projection"] - output["line"]).abs()
    market_gap_scale = output["market"].map({
        "hitter_hits": 1.25,
        "hitter_total_bases": 2.5,
        "hitter_runs": 1.25,
        "hitter_rbis": 1.25,
        "hitter_hits_runs_rbis": 3.5,
        "hitter_fantasy_score": 12.0,
        "pitcher_strikeouts": 4.0,
        "pitcher_outs": 7.0,
    }).fillna(4.0)
    output["projection_quality_score"] = (
        100.0 - (projection_gap / market_gap_scale).clip(lower=0, upper=1.5) * 35.0
    ).clip(lower=0, upper=100)

    method_trust = output["distribution_method"].map({
        "empirical_holdout_residuals": 92.0,
        "empirical_pitcher_holdout_residuals": 95.0,
        "normal_residual_std": 78.0,
        "poisson": 68.0,
        "poisson_fallback": 62.0,
    }).fillna(25.0)
    sample_score = (
        pd.to_numeric(output["calibration_sample_size"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=750)
        / 750.0
        * 100.0
    )
    # Pitcher fallbacks have no residual sample size, so method trust carries them.
    sample_score = np.where(
        output["distribution_method"].isin(["poisson", "poisson_fallback", "normal_residual_std"]),
        method_trust,
        sample_score,
    )
    output["confidence_score"] = (
        method_trust * 0.30
        + pd.Series(sample_score, index=output.index) * 0.20
        + output["market_quality_score"] * 0.20
        + output["volatility_score"] * 0.15
        + output["projection_quality_score"] * 0.15
    ).clip(lower=0, upper=100)
    output.loc[output["probability_status"].ne("calculated"), "confidence_score"] = 0.0
    output["risk_score"] = (100.0 - output["confidence_score"]).clip(lower=0, upper=100)
    output["model_version"] = MODEL_VERSION
    output = output.drop(columns=["_normalized_player_v2"])

    output["is_pickem_platform"] = output[
        "platform"
    ].apply(is_pickem_platform)

    output["sportsbook_implied_probability"] = output[
        "sportsbook_odds"
    ].apply(american_odds_to_probability)

    # Pick'em operators do not provide normal American odds. Treat their
    # selection threshold as 50% instead of pretending -100 is a real price.
    output.loc[
        output["is_pickem_platform"],
        "sportsbook_implied_probability",
    ] = 0.50

    output["raw_probability_edge"] = (
        output["probability"]
        - output["sportsbook_implied_probability"]
    )

    # ------------------------------------------------------------------
    # Composite ranking score
    # ------------------------------------------------------------------

    output["ranking_score"] = (
        output["raw_probability_edge"].clip(lower=0).fillna(0) * 0.38
        + output["probability"].fillna(0) * 0.18
        + (output["confidence_score"].fillna(0) / 100.0) * 0.20
        + (output["market_quality_score"].fillna(0) / 100.0) * 0.12
        + (output["projection_quality_score"].fillna(0) / 100.0) * 0.07
        + output["calibration_shrinkage"].fillna(0) * 0.05
    )

    # Penalize extreme alternate-line favorites
    extreme_probability = output["probability"] > 0.85

    pickem_platform = (
        output["platform"]
        .astype(str)
        .str.lower()
        .isin([
            "prizepicks",
            "sleeper",
            "underdog",
        ])
    )

    output.loc[
        extreme_probability & pickem_platform,
        "ranking_score",
    ] *= 0.60

    output = output.sort_values(
        [
            "probability_status",
            "ranking_score",
            "probability",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    calculated_count = int(
        output["probability_status"]
        .eq("calculated")
        .sum()
    )

    rejected_count = int(
        output["probability_status"]
        .eq("rejected")
        .sum()
    )

    print("=" * 72)
    print("MLB probability table created")
    print(f"Current platform lines: {len(lines):,}")
    print(f"Matched/calculated rows: {calculated_count:,}")
    print(f"Rejected rows: {rejected_count:,}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 72)

    preview_columns = [
        "player",
        "platform",
        "market",
        "direction",
        "line",
        "projection",
        "raw_probability",
        "probability",
        "calibration_shrinkage",
        "fair_odds",
        "distribution_method",
        "probability_status",
    ]

    if not output.empty:
        print(
            output[preview_columns]
            .head(40)
            .to_string(index=False)
        )

    return output


if __name__ == "__main__":
    build_probability_table()
