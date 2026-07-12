"""Generate current-slate MLB pitcher fantasy projections.

Environment variables:
    MLB_TARGET_DATE=YYYY-MM-DD

Inputs:
    outputs/calibrated_strikeout_projections.csv
    outputs/pitcher_outs_projections.csv
    data/game_logs/<target-date>.csv

Output:
    outputs/pitcher_fantasy_projections.csv

The script uses current strikeout and outs projections plus each pitcher's
historical run-prevention and baserunner statistics when those fields are
available. Fallback assumptions are clearly identified rather than presented
as calibrated estimates.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

STRIKEOUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "calibrated_strikeout_projections.csv"
)

OUTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "pitcher_outs_projections.csv"
)

GAME_LOGS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "game_logs"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "pitcher_fantasy_projections.csv"
)

# DraftKings pitcher scoring.
DK_INNING_PITCHED_POINTS = 2.25
DK_STRIKEOUT_POINTS = 2.0
DK_WIN_POINTS = 4.0
DK_EARNED_RUN_POINTS = -2.0
DK_HIT_ALLOWED_POINTS = -0.6
DK_WALK_POINTS = -0.6

# FanDuel pitcher scoring.
FD_INNING_PITCHED_POINTS = 3.0
FD_STRIKEOUT_POINTS = 3.0
FD_WIN_POINTS = 6.0
FD_EARNED_RUN_POINTS = -3.0
FD_QUALITY_START_POINTS = 4.0

# Conservative fallbacks when historical fields are unavailable.
DEFAULT_PROJECTED_EARNED_RUNS = 2.5
DEFAULT_PROJECTED_HITS_ALLOWED = 5.0
DEFAULT_PROJECTED_WALKS = 2.0
DEFAULT_PROJECTED_WIN_PROBABILITY = 0.35

OUTPUT_COLUMNS = [
    "date",
    "game_id",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "side",
    "projected_ks",
    "projected_outs",
    "projected_ip",
    "projected_er",
    "projected_hits_allowed",
    "projected_walks",
    "projected_win_probability",
    "projected_quality_start_probability",
    "draftkings_pitcher_points",
    "fanduel_pitcher_points",
    "projected_fantasy_score",
    "projected_fantasy_score_lower_80",
    "projected_fantasy_score_upper_80",
    "projected_fantasy_score_residual_std",
    "history_games",
    "projection_method",
    "calibration_status",
    "uncertainty_method",
    "used_fallback_er",
    "used_fallback_hits_allowed",
    "used_fallback_walks",
    "used_fallback_win_probability",
]


def get_target_date() -> date:
    """Return the requested MLB slate date."""
    raw_value = os.getenv(
        "MLB_TARGET_DATE",
        date.today().isoformat(),
    )

    try:
        return datetime.strptime(
            raw_value,
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc


def normalize_name(value: Any) -> str:
    """Normalize names for safe cross-file matching."""
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
    text = re.sub(
        r"\b(jr|sr|ii|iii|iv)\b",
        " ",
        text,
    )
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def choose_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first available candidate column."""
    return next(
        (
            column
            for column in candidates
            if column in frame.columns
        ),
        None,
    )


def read_required_csv(
    path: Path,
    label: str,
) -> pd.DataFrame:
    """Read a required CSV with clear errors."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} was not found: {path}"
        )

    try:
        frame = pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError(
            f"Could not read {label}: {path}"
        ) from exc

    if frame.empty:
        raise ValueError(
            f"{label} is empty: {path}"
        )

    return frame


def load_strikeout_projections() -> pd.DataFrame:
    """Load current calibrated strikeout projections."""
    frame = read_required_csv(
        STRIKEOUT_PATH,
        "calibrated strikeout projections",
    )

    player_column = choose_column(
        frame,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
        ],
    )

    projection_column = choose_column(
        frame,
        [
            "calibrated_projected_ks",
            "projected_ks",
            "projected_strikeouts",
        ],
    )

    if player_column is None:
        raise ValueError(
            "Strikeout projection file has no pitcher-name column."
        )

    if projection_column is None:
        raise ValueError(
            "Strikeout projection file has no recognized "
            "strikeout projection column."
        )

    frame = frame.rename(
        columns={
            player_column: "pitcher_name",
            projection_column: "projected_ks",
        }
    )

    frame["pitcher_key"] = frame[
        "pitcher_name"
    ].apply(normalize_name)

    frame["projected_ks"] = pd.to_numeric(
        frame["projected_ks"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "pitcher_name",
            "projected_ks",
        ]
    ).copy()

    frame = frame.loc[
        frame["pitcher_key"].ne("")
        & frame["projected_ks"].ge(0)
    ].copy()

    frame = frame.drop_duplicates(
        subset=["pitcher_key"],
        keep="last",
    ).reset_index(drop=True)

    return frame


def load_outs_projections() -> pd.DataFrame:
    """Load current pitcher-outs projections."""
    frame = read_required_csv(
        OUTS_PATH,
        "pitcher outs projections",
    )

    player_column = choose_column(
        frame,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
        ],
    )

    projection_column = choose_column(
        frame,
        [
            "projected_outs",
            "calibrated_projected_outs",
        ],
    )

    if player_column is None:
        raise ValueError(
            "Pitcher outs file has no pitcher-name column."
        )

    if projection_column is None:
        raise ValueError(
            "Pitcher outs file has no projected-outs column."
        )

    frame = frame.rename(
        columns={
            player_column: "pitcher_name",
            projection_column: "projected_outs",
        }
    )

    frame["pitcher_key"] = frame[
        "pitcher_name"
    ].apply(normalize_name)

    frame["projected_outs"] = pd.to_numeric(
        frame["projected_outs"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "pitcher_name",
            "projected_outs",
        ]
    ).copy()

    frame = frame.loc[
        frame["pitcher_key"].ne("")
        & frame["projected_outs"].between(
            0,
            27,
            inclusive="both",
        )
    ].copy()

    frame = frame.drop_duplicates(
        subset=["pitcher_key"],
        keep="last",
    ).reset_index(drop=True)

    return frame


def merge_current_projections(
    strikeouts: pd.DataFrame,
    outs: pd.DataFrame,
) -> pd.DataFrame:
    """Merge strikeout and outs projections by normalized pitcher name.

    The source files may contain different or missing game_id values.
    Pitcher names are therefore the primary matching key.
    """
    strikeout_keys = set(
        strikeouts["pitcher_key"]
        .dropna()
        .astype(str)
    )

    outs_keys = set(
        outs["pitcher_key"]
        .dropna()
        .astype(str)
    )

    common_keys = strikeout_keys & outs_keys

    print(
        f"Strikeout pitchers: {len(strikeout_keys)}"
    )

    print(
        f"Outs pitchers: {len(outs_keys)}"
    )

    print(
        f"Common normalized pitchers: {len(common_keys)}"
    )

    if not common_keys:
        strikeout_only = sorted(
            strikeout_keys - outs_keys
        )[:20]

        outs_only = sorted(
            outs_keys - strikeout_keys
        )[:20]

        print(
            "First strikeout-only pitcher keys:",
            strikeout_only,
        )

        print(
            "First outs-only pitcher keys:",
            outs_only,
        )

        raise RuntimeError(
            "No pitchers matched between strikeout and outs projections."
        )

    merged = strikeouts.merge(
        outs,
        on="pitcher_key",
        how="inner",
        suffixes=(
            "_strikeouts",
            "_outs",
        ),
        validate="one_to_one",
    )

    print(
        f"Matched {len(merged)} pitchers "
        f"(Strikeouts={len(strikeouts)}, Outs={len(outs)})"
    )

    if merged.empty:
        raise RuntimeError(
            "No pitchers matched between strikeout and outs projections."
        )

    for base_column in [
        "pitcher_name",
        "pitcher_id",
        "team",
        "opponent",
        "side",
        "date",
    ]:
        strikeout_column = (
            f"{base_column}_strikeouts"
        )

        outs_column = (
            f"{base_column}_outs"
        )

        if strikeout_column in merged.columns:
            merged[base_column] = merged[
                strikeout_column
            ]
        elif outs_column in merged.columns:
            merged[base_column] = merged[
                outs_column
            ]
        elif base_column not in merged.columns:
            merged[base_column] = pd.NA

    if "game_id_strikeouts" in merged.columns:
        merged["game_id"] = merged[
            "game_id_strikeouts"
        ].combine_first(
            merged.get(
                "game_id_outs",
                pd.Series(
                    pd.NA,
                    index=merged.index,
                ),
            )
        )
    elif "game_id_outs" in merged.columns:
        merged["game_id"] = merged[
            "game_id_outs"
        ]
    elif "game_id" not in merged.columns:
        merged["game_id"] = pd.NA

    return merged


def get_game_logs_path(
    target_date: date,
) -> Path:
    """Return the matching pitcher game-log file."""
    return (
        GAME_LOGS_DIRECTORY
        / f"{target_date.isoformat()}.csv"
    )


def load_pitcher_history(
    target_date: date,
) -> pd.DataFrame:
    """Load historical pitcher statistics when available."""
    logs_path = get_game_logs_path(
        target_date
    )

    if not logs_path.exists():
        print(
            "Pitcher game logs were not found. "
            "Fantasy components will use conservative fallbacks."
        )

        return pd.DataFrame()

    try:
        logs = pd.read_csv(logs_path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        print(
            "Pitcher game logs could not be read. "
            "Fantasy components will use conservative fallbacks."
        )

        return pd.DataFrame()

    player_column = choose_column(
        logs,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
        ],
    )

    date_column = choose_column(
        logs,
        [
            "game_date",
            "date",
        ],
    )

    if player_column is None or date_column is None:
        print(
            "Pitcher game-log schema was not recognized. "
            "Fantasy components will use conservative fallbacks."
        )

        return pd.DataFrame()

    logs = logs.rename(
        columns={
            player_column: "pitcher_name",
            date_column: "game_date",
        }
    )

    logs["pitcher_key"] = logs[
        "pitcher_name"
    ].apply(normalize_name)

    logs["game_date"] = pd.to_datetime(
        logs["game_date"],
        errors="coerce",
    )

    logs = logs.dropna(
        subset=[
            "pitcher_key",
            "game_date",
        ]
    ).copy()

    logs = logs.loc[
        logs["game_date"]
        < pd.Timestamp(target_date)
    ].copy()

    return logs.sort_values(
        [
            "pitcher_key",
            "game_date",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def weighted_recent_average(
    values: pd.Series,
    default: float,
) -> tuple[float, bool]:
    """Blend recent and season averages."""
    clean_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    clean_values = clean_values.loc[
        np.isfinite(clean_values)
    ]

    if clean_values.empty:
        return default, True

    last3 = float(
        clean_values.head(3).mean()
    )

    last5 = float(
        clean_values.head(5).mean()
    )

    season = float(
        clean_values.mean()
    )

    if len(clean_values) >= 5:
        projection = (
            0.45 * last3
            + 0.35 * last5
            + 0.20 * season
        )
    else:
        projection = (
            0.60 * last3
            + 0.40 * season
        )

    return max(
        0.0,
        projection,
    ), False


def historical_component(
    pitcher_logs: pd.DataFrame,
    candidates: list[str],
    default: float,
) -> tuple[float, bool]:
    """Project one pitching component from available history."""
    column = choose_column(
        pitcher_logs,
        candidates,
    )

    if column is None:
        return default, True

    return weighted_recent_average(
        pitcher_logs[column],
        default,
    )


def estimate_win_probability(
    pitcher_logs: pd.DataFrame,
) -> tuple[float, bool]:
    """Estimate win probability from historical decisions when possible."""
    win_column = choose_column(
        pitcher_logs,
        [
            "wins",
            "win",
            "is_win",
            "pitcher_win",
        ],
    )

    if win_column is None:
        return (
            DEFAULT_PROJECTED_WIN_PROBABILITY,
            True,
        )

    values = pd.to_numeric(
        pitcher_logs[win_column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return (
            DEFAULT_PROJECTED_WIN_PROBABILITY,
            True,
        )

    values = values.clip(
        lower=0,
        upper=1,
    )

    last5_probability = float(
        values.head(5).mean()
    )

    season_probability = float(
        values.mean()
    )

    projection = (
        0.65 * last5_probability
        + 0.35 * season_probability
    )

    return (
        float(
            np.clip(
                projection,
                0.05,
                0.75,
            )
        ),
        False,
    )


def innings_to_decimal(value: Any) -> float:
    """Convert baseball innings notation to decimal innings."""
    if value is None or pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if not text:
        return float("nan")

    if "." in text:
        whole_text, partial_text = text.split(".", 1)
    else:
        whole_text, partial_text = text, "0"

    try:
        whole_innings = int(whole_text)
        partial_outs = int(
            partial_text[:1] or "0"
        )
    except (TypeError, ValueError):
        return float("nan")

    if partial_outs not in {0, 1, 2}:
        return float("nan")

    return float(
        whole_innings
        + partial_outs / 3.0
    )


def estimate_quality_start_probability(
    pitcher_logs: pd.DataFrame,
) -> float:
    """Estimate quality-start probability from historical starts."""
    innings_column = choose_column(
        pitcher_logs,
        [
            "innings",
            "innings_pitched",
            "ip",
        ],
    )

    earned_runs_column = choose_column(
        pitcher_logs,
        [
            "earned_runs",
            "er",
            "runs_allowed",
        ],
    )

    if (
        innings_column is None
        or earned_runs_column is None
    ):
        return float("nan")

    innings = pitcher_logs[
        innings_column
    ].apply(innings_to_decimal)

    earned_runs = pd.to_numeric(
        pitcher_logs[earned_runs_column],
        errors="coerce",
    )

    valid = (
        innings.notna()
        & earned_runs.notna()
    )

    if not valid.any():
        return float("nan")

    quality_start = (
        innings.loc[valid].ge(6.0)
        & earned_runs.loc[valid].le(3.0)
    ).astype(float)

    recent = quality_start.head(10)

    return float(
        np.clip(
            recent.mean(),
            0.0,
            1.0,
        )
    )


def add_historical_components(
    current: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Add projected ER, hits, walks, wins, and quality starts."""
    rows: list[dict[str, Any]] = []

    for _, pitcher in current.iterrows():
        pitcher_key = pitcher["pitcher_key"]

        if history.empty:
            pitcher_logs = pd.DataFrame()
        else:
            pitcher_logs = history.loc[
                history["pitcher_key"].eq(
                    pitcher_key
                )
            ].copy()

        projected_er, fallback_er = historical_component(
            pitcher_logs,
            [
                "earned_runs",
                "er",
                "runs_allowed",
            ],
            DEFAULT_PROJECTED_EARNED_RUNS,
        )

        projected_hits, fallback_hits = historical_component(
            pitcher_logs,
            [
                "hits_allowed",
                "hits",
                "h",
            ],
            DEFAULT_PROJECTED_HITS_ALLOWED,
        )

        projected_walks, fallback_walks = historical_component(
            pitcher_logs,
            [
                "walks",
                "base_on_balls",
                "bb",
            ],
            DEFAULT_PROJECTED_WALKS,
        )

        (
            projected_win_probability,
            fallback_win,
        ) = estimate_win_probability(
            pitcher_logs
        )

        quality_start_probability = (
            estimate_quality_start_probability(
                pitcher_logs
            )
        )

        rows.append(
            {
                "pitcher_key": pitcher_key,
                "history_games": len(
                    pitcher_logs
                ),
                "projected_er": projected_er,
                "projected_hits_allowed": projected_hits,
                "projected_walks": projected_walks,
                "projected_win_probability": (
                    projected_win_probability
                ),
                "projected_quality_start_probability": (
                    quality_start_probability
                ),
                "used_fallback_er": fallback_er,
                "used_fallback_hits_allowed": (
                    fallback_hits
                ),
                "used_fallback_walks": fallback_walks,
                "used_fallback_win_probability": (
                    fallback_win
                ),
            }
        )

    component_frame = pd.DataFrame(
        rows
    )

    return current.merge(
        component_frame,
        on="pitcher_key",
        how="left",
        validate="one_to_one",
    )


def calculate_draftkings_points(
    frame: pd.DataFrame,
) -> pd.Series:
    """Calculate expected DraftKings pitcher points."""
    return (
        frame["projected_ip"]
        * DK_INNING_PITCHED_POINTS
        + frame["projected_ks"]
        * DK_STRIKEOUT_POINTS
        + frame["projected_er"]
        * DK_EARNED_RUN_POINTS
        + frame["projected_hits_allowed"]
        * DK_HIT_ALLOWED_POINTS
        + frame["projected_walks"]
        * DK_WALK_POINTS
        + frame["projected_win_probability"]
        * DK_WIN_POINTS
    )


def calculate_fanduel_points(
    frame: pd.DataFrame,
) -> pd.Series:
    """Calculate expected FanDuel pitcher points."""
    quality_start_probability = frame[
        "projected_quality_start_probability"
    ].fillna(0.0)

    return (
        frame["projected_ip"]
        * FD_INNING_PITCHED_POINTS
        + frame["projected_ks"]
        * FD_STRIKEOUT_POINTS
        + frame["projected_er"]
        * FD_EARNED_RUN_POINTS
        + frame["projected_win_probability"]
        * FD_WIN_POINTS
        + quality_start_probability
        * FD_QUALITY_START_POINTS
    )


def estimate_fantasy_uncertainty(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create descriptive ranges from component uncertainty."""
    result = frame.copy()

    strikeout_std_column = choose_column(
        result,
        [
            "projected_ks_residual_std",
            "projected_ks_residual_std_strikeouts",
        ],
    )

    outs_std_column = choose_column(
        result,
        [
            "projected_outs_residual_std",
            "projected_outs_residual_std_outs",
        ],
    )

    if strikeout_std_column:
        strikeout_std = pd.to_numeric(
            result[strikeout_std_column],
            errors="coerce",
        )
    else:
        strikeout_std = pd.Series(
            np.nan,
            index=result.index,
        )

    if outs_std_column:
        outs_std = pd.to_numeric(
            result[outs_std_column],
            errors="coerce",
        )
    else:
        outs_std = pd.Series(
            np.nan,
            index=result.index,
        )

    innings_std = (
        outs_std
        / 3.0
    )

    estimated_std = np.sqrt(
        (
            DK_STRIKEOUT_POINTS
            * strikeout_std.fillna(0.0)
        )
        ** 2
        + (
            DK_INNING_PITCHED_POINTS
            * innings_std.fillna(0.0)
        )
        ** 2
        + 3.0**2
    )

    no_uncertainty = (
        strikeout_std.isna()
        & outs_std.isna()
    )

    estimated_std = estimated_std.mask(
        no_uncertainty,
        np.nan,
    )

    result[
        "projected_fantasy_score_residual_std"
    ] = estimated_std

    result[
        "projected_fantasy_score_lower_80"
    ] = np.clip(
        result["projected_fantasy_score"]
        - 1.2816 * estimated_std,
        a_min=0.0,
        a_max=None,
    )

    result[
        "projected_fantasy_score_upper_80"
    ] = (
        result["projected_fantasy_score"]
        + 1.2816 * estimated_std
    )

    return result


def build_output(
    frame: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    """Finalize the production fantasy projection schema."""
    output = frame.copy()

    output["date"] = (
        target_date.isoformat()
    )

    output["projected_ks"] = pd.to_numeric(
        output["projected_ks"],
        errors="coerce",
    )

    output["projected_outs"] = pd.to_numeric(
        output["projected_outs"],
        errors="coerce",
    )

    output["projected_ip"] = (
        output["projected_outs"]
        / 3.0
    )

    output[
        "draftkings_pitcher_points"
    ] = calculate_draftkings_points(
        output
    )

    output[
        "fanduel_pitcher_points"
    ] = calculate_fanduel_points(
        output
    )

    output["projected_fantasy_score"] = output[
        "draftkings_pitcher_points"
    ]

    output = estimate_fantasy_uncertainty(
        output
    )

    output["projection_method"] = (
        "strikeouts_outs_and_historical_components"
    )

    output["calibration_status"] = "UNCALIBRATED"

    output["uncertainty_method"] = (
        "component_dispersion_approximation"
    )

    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    numeric_columns = [
        "projected_ks",
        "projected_outs",
        "projected_ip",
        "projected_er",
        "projected_hits_allowed",
        "projected_walks",
        "projected_win_probability",
        "projected_quality_start_probability",
        "draftkings_pitcher_points",
        "fanduel_pitcher_points",
        "projected_fantasy_score",
        "projected_fantasy_score_lower_80",
        "projected_fantasy_score_upper_80",
        "projected_fantasy_score_residual_std",
        "history_games",
    ]

    for column in numeric_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).round(4)

    output = output.dropna(
        subset=[
            "pitcher_name",
            "projected_ks",
            "projected_outs",
            "projected_fantasy_score",
        ]
    ).copy()

    output = output.sort_values(
        [
            "projected_fantasy_score",
            "pitcher_name",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return output[
        OUTPUT_COLUMNS
    ].copy()


def save_output(
    frame: pd.DataFrame,
) -> None:
    """Write the output atomically."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.csv"
    )

    frame.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        OUTPUT_PATH
    )


def project_pitcher_fantasy() -> pd.DataFrame:
    """Generate and save current pitcher fantasy projections."""
    target_date = get_target_date()

    print("=" * 72)
    print("GENERATING PITCHER FANTASY PROJECTIONS")
    print(f"Slate date: {target_date.isoformat()}")
    print("=" * 72)

    strikeouts = load_strikeout_projections()
    outs = load_outs_projections()

    current = merge_current_projections(
        strikeouts,
        outs,
    )

    history = load_pitcher_history(
        target_date
    )

    current = add_historical_components(
        current,
        history,
    )

    output = build_output(
        current,
        target_date,
    )

    save_output(output)

    print(
        f"\nSaved {len(output):,} pitcher fantasy projections "
        f"to {OUTPUT_PATH}"
    )

    print(
        "\nImportant: these pitcher fantasy projections are "
        "not yet leakage-safe calibrated models."
    )

    fallback_counts = {
        "ER fallback": int(
            output["used_fallback_er"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "Hits fallback": int(
            output[
                "used_fallback_hits_allowed"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "Walks fallback": int(
            output["used_fallback_walks"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "Win fallback": int(
            output[
                "used_fallback_win_probability"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
    }

    print("\nFallback component counts:")

    for label, count in fallback_counts.items():
        print(f"{label}: {count}")

    preview_columns = [
        "pitcher_name",
        "team",
        "opponent",
        "projected_ks",
        "projected_outs",
        "projected_ip",
        "projected_er",
        "projected_hits_allowed",
        "projected_walks",
        "projected_win_probability",
        "draftkings_pitcher_points",
        "fanduel_pitcher_points",
        "calibration_status",
    ]

    print("\nProjection preview:")

    print(
        output[
            preview_columns
        ]
        .head(30)
        .to_string(index=False)
    )

    return output


if __name__ == "__main__":
    project_pitcher_fantasy()
