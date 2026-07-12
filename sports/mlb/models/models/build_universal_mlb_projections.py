"""Build one standardized MLB projection table for all supported markets.

Inputs may include:
    outputs/hitters/today_hitter_projections.csv
    outputs/calibrated_strikeout_projections.csv
    outputs/pitcher_outs_projections.csv
    outputs/pitcher_fantasy_projections.csv

Output:
    outputs/mlb_universal_projections.csv

The output uses one row per player, game, and market so downstream probability,
betting, and Streamlit code can consume a consistent schema.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIRECTORY / "mlb_universal_projections.csv"

HITTER_PROJECTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "hitters"
    / "today_hitter_projections.csv"
)

STRIKEOUT_PROJECTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "calibrated_strikeout_projections.csv"
)

PITCHER_OUTS_PATHS = [
    OUTPUT_DIRECTORY / "pitcher_outs_projections.csv",
    OUTPUT_DIRECTORY / "pitcher_outs.csv",
]

PITCHER_FANTASY_PATHS = [
    OUTPUT_DIRECTORY / "pitcher_fantasy_projections.csv",
    OUTPUT_DIRECTORY / "pitcher_fantasy.csv",
]

HITTER_MARKETS = {
    "hitter_hits": {
        "projection": "projected_hits",
        "lower": "projected_hits_lower_80",
        "upper": "projected_hits_upper_80",
        "residual_std": "projected_hits_residual_std",
        "model_name": "hits_model_name",
        "validation_mae": "hits_validation_mae",
    },
    "hitter_total_bases": {
        "projection": "projected_total_bases",
        "lower": "projected_total_bases_lower_80",
        "upper": "projected_total_bases_upper_80",
        "residual_std": "projected_total_bases_residual_std",
        "model_name": "total_bases_model_name",
        "validation_mae": "total_bases_validation_mae",
    },
    "hitter_runs": {
        "projection": "projected_runs",
        "lower": "projected_runs_lower_80",
        "upper": "projected_runs_upper_80",
        "residual_std": "projected_runs_residual_std",
        "model_name": "runs_model_name",
        "validation_mae": "runs_validation_mae",
    },
    "hitter_rbis": {
        "projection": "projected_rbi",
        "lower": "projected_rbi_lower_80",
        "upper": "projected_rbi_upper_80",
        "residual_std": "projected_rbi_residual_std",
        "model_name": "rbi_model_name",
        "validation_mae": "rbi_validation_mae",
    },
    "hitter_hits_runs_rbis": {
        "projection": "projected_hits_runs_rbis",
        "lower": "projected_hits_runs_rbis_lower_80",
        "upper": "projected_hits_runs_rbis_upper_80",
        "residual_std": "projected_hits_runs_rbis_residual_std",
        "model_name": "hits_runs_rbis_model_name",
        "validation_mae": "hits_runs_rbis_validation_mae",
    },
    "hitter_fantasy_score": {
        "projection": "projected_fantasy_score",
        "lower": "projected_fantasy_score_lower_80",
        "upper": "projected_fantasy_score_upper_80",
        "residual_std": "projected_fantasy_score_residual_std",
        "model_name": "fantasy_score_model_name",
        "validation_mae": "fantasy_score_validation_mae",
    },
}

OUTPUT_COLUMNS = [
    "sport",
    "slate_date",
    "game_id",
    "player_id",
    "player",
    "team",
    "opponent",
    "market",
    "projection",
    "projection_lower_80",
    "projection_upper_80",
    "residual_standard_deviation",
    "model_name",
    "validation_mae",
    "batting_order",
    "position",
    "expected_plate_appearances",
    "days_rest",
    "source_file",
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


def first_existing_path(
    paths: list[Path],
) -> Path | None:
    """Return the first existing file from a candidate list."""
    for path in paths:
        if path.exists():
            return path

    return None


def read_optional_csv(
    path: Path | None,
    label: str,
) -> pd.DataFrame:
    """Read an optional CSV without crashing on a missing source."""
    if path is None or not path.exists():
        print(f"Missing optional {label} file.")
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        print(f"Could not read {label} file {path}: {exc}")
        return pd.DataFrame()

    if frame.empty:
        print(f"{label} file is empty: {path}")

    return frame


def numeric_value(
    row: pd.Series,
    column: str | None,
) -> float:
    """Return a finite numeric value or NaN."""
    if not column:
        return float("nan")

    value = pd.to_numeric(
        row.get(column),
        errors="coerce",
    )

    if pd.isna(value) or not np.isfinite(value):
        return float("nan")

    return float(value)


def text_value(
    row: pd.Series,
    column: str | None,
) -> Any:
    """Return a row value when its source column exists."""
    if not column:
        return pd.NA

    value = row.get(column)

    if value is None or pd.isna(value):
        return pd.NA

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else pd.NA

    return value


def choose_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first candidate column present in a dataframe."""
    return next(
        (
            column
            for column in candidates
            if column in frame.columns
        ),
        None,
    )


def append_projection_rows(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    market: str,
    player_column: str,
    projection_column: str,
    source_file: Path,
    target_date: date,
    lower_column: str | None = None,
    upper_column: str | None = None,
    residual_std_column: str | None = None,
    model_name_column: str | None = None,
    validation_mae_column: str | None = None,
) -> None:
    """Append one market's rows using the standard output schema."""
    if frame.empty:
        return

    if player_column not in frame.columns:
        print(
            f"Skipping {market}: missing player column "
            f"{player_column!r}."
        )
        return

    if projection_column not in frame.columns:
        print(
            f"Skipping {market}: missing projection column "
            f"{projection_column!r}."
        )
        return

    added = 0

    for _, row in frame.iterrows():
        player = text_value(row, player_column)
        projection = numeric_value(
            row,
            projection_column,
        )

        if pd.isna(player) or not np.isfinite(projection):
            continue

        if projection < 0:
            projection = 0.0

        output_row = {
            "sport": "MLB",
            "slate_date": target_date.isoformat(),
            "game_id": text_value(row, "game_id"),
            "player_id": text_value(row, "player_id"),
            "player": player,
            "team": text_value(row, "team"),
            "opponent": text_value(row, "opponent"),
            "market": market,
            "projection": projection,
            "projection_lower_80": numeric_value(
                row,
                lower_column,
            ),
            "projection_upper_80": numeric_value(
                row,
                upper_column,
            ),
            "residual_standard_deviation": numeric_value(
                row,
                residual_std_column,
            ),
            "model_name": text_value(
                row,
                model_name_column,
            ),
            "validation_mae": numeric_value(
                row,
                validation_mae_column,
            ),
            "batting_order": numeric_value(
                row,
                "batting_order",
            ),
            "position": text_value(row, "position"),
            "expected_plate_appearances": numeric_value(
                row,
                "expected_plate_appearances",
            ),
            "days_rest": numeric_value(
                row,
                "days_rest",
            ),
            "source_file": str(
                source_file.relative_to(PROJECT_ROOT)
            ),
        }

        rows.append(output_row)
        added += 1

    print(f"Added {added} rows for {market}.")


def add_hitter_markets(
    rows: list[dict[str, Any]],
    target_date: date,
) -> None:
    """Add all supported hitter projections."""
    hitters = read_optional_csv(
        HITTER_PROJECTIONS_PATH,
        "hitter projections",
    )

    if hitters.empty:
        return

    if "date" in hitters.columns:
        hitter_dates = pd.to_datetime(
            hitters["date"],
            errors="coerce",
        ).dt.date

        valid_date_rows = hitter_dates.eq(target_date)

        if valid_date_rows.any():
            hitters = hitters.loc[
                valid_date_rows
            ].copy()

    for market, config in HITTER_MARKETS.items():
        append_projection_rows(
            rows,
            hitters,
            market=market,
            player_column="player_name",
            projection_column=config["projection"],
            lower_column=config["lower"],
            upper_column=config["upper"],
            residual_std_column=config["residual_std"],
            model_name_column=config["model_name"],
            validation_mae_column=config["validation_mae"],
            source_file=HITTER_PROJECTIONS_PATH,
            target_date=target_date,
        )


def add_pitcher_strikeouts(
    rows: list[dict[str, Any]],
    target_date: date,
) -> None:
    """Add calibrated pitcher-strikeout projections."""
    strikeouts = read_optional_csv(
        STRIKEOUT_PROJECTIONS_PATH,
        "pitcher strikeout projections",
    )

    if strikeouts.empty:
        return

    player_column = choose_column(
        strikeouts,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
        ],
    )

    projection_column = choose_column(
        strikeouts,
        [
            "calibrated_projected_ks",
            "projected_strikeouts",
            "projected_ks",
        ],
    )

    if player_column is None or projection_column is None:
        print(
            "Skipping pitcher strikeouts: projection schema "
            "was not recognized."
        )
        return

    append_projection_rows(
        rows,
        strikeouts,
        market="pitcher_strikeouts",
        player_column=player_column,
        projection_column=projection_column,
        lower_column=choose_column(
            strikeouts,
            [
                "projected_ks_lower_80",
                "calibrated_projected_ks_lower_80",
            ],
        ),
        upper_column=choose_column(
            strikeouts,
            [
                "projected_ks_upper_80",
                "calibrated_projected_ks_upper_80",
            ],
        ),
        residual_std_column=choose_column(
            strikeouts,
            [
                "projected_ks_residual_std",
                "residual_std",
            ],
        ),
        model_name_column=choose_column(
            strikeouts,
            [
                "model_name",
                "strikeout_model_name",
            ],
        ),
        validation_mae_column=choose_column(
            strikeouts,
            [
                "validation_mae",
                "strikeout_validation_mae",
            ],
        ),
        source_file=STRIKEOUT_PROJECTIONS_PATH,
        target_date=target_date,
    )


def add_pitcher_outs(
    rows: list[dict[str, Any]],
    target_date: date,
) -> None:
    """Add pitcher-outs projections."""
    path = first_existing_path(PITCHER_OUTS_PATHS)

    outs = read_optional_csv(
        path,
        "pitcher outs projections",
    )

    if outs.empty or path is None:
        return

    player_column = choose_column(
        outs,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
        ],
    )

    projection_column = choose_column(
        outs,
        [
            "projected_outs",
            "calibrated_projected_outs",
            "projection",
        ],
    )

    if player_column is None or projection_column is None:
        print(
            "Skipping pitcher outs: projection schema "
            "was not recognized."
        )
        return

    append_projection_rows(
        rows,
        outs,
        market="pitcher_outs",
        player_column=player_column,
        projection_column=projection_column,
        lower_column=choose_column(
            outs,
            [
                "projected_outs_lower_80",
                "projection_lower_80",
            ],
        ),
        upper_column=choose_column(
            outs,
            [
                "projected_outs_upper_80",
                "projection_upper_80",
            ],
        ),
        residual_std_column=choose_column(
            outs,
            [
                "projected_outs_residual_std",
                "residual_std",
                "projection_std",
            ],
        ),
        model_name_column=choose_column(
            outs,
            [
                "model_name",
                "outs_model_name",
            ],
        ),
        validation_mae_column=choose_column(
            outs,
            [
                "validation_mae",
                "outs_validation_mae",
            ],
        ),
        source_file=path,
        target_date=target_date,
    )


def add_pitcher_fantasy(
    rows: list[dict[str, Any]],
    target_date: date,
) -> None:
    """Add pitcher fantasy-score projections when available."""
    path = first_existing_path(PITCHER_FANTASY_PATHS)

    fantasy = read_optional_csv(
        path,
        "pitcher fantasy projections",
    )

    if fantasy.empty or path is None:
        return

    player_column = choose_column(
        fantasy,
        [
            "pitcher_name",
            "player_name",
            "pitcher",
        ],
    )

    projection_column = choose_column(
        fantasy,
        [
            "draftkings_pitcher_points",
            "projected_fantasy_score",
            "fantasy_projection",
            "projection",
        ],
    )

    if player_column is None or projection_column is None:
        print(
            "Skipping pitcher fantasy: projection schema "
            "was not recognized."
        )
        return

    append_projection_rows(
        rows,
        fantasy,
        market="pitcher_fantasy_score",
        player_column=player_column,
        projection_column=projection_column,
        lower_column=choose_column(
            fantasy,
            [
                "projected_fantasy_score_lower_80",
                "projection_lower_80",
            ],
        ),
        upper_column=choose_column(
            fantasy,
            [
                "projected_fantasy_score_upper_80",
                "projection_upper_80",
            ],
        ),
        residual_std_column=choose_column(
            fantasy,
            [
                "projected_fantasy_score_residual_std",
                "residual_std",
                "projection_std",
            ],
        ),
        model_name_column=choose_column(
            fantasy,
            [
                "model_name",
                "fantasy_model_name",
            ],
        ),
        validation_mae_column=choose_column(
            fantasy,
            [
                "validation_mae",
                "fantasy_validation_mae",
            ],
        ),
        source_file=path,
        target_date=target_date,
    )


def finalize_universal_table(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Clean and deduplicate the standardized projection table."""
    universal = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if universal.empty:
        raise RuntimeError(
            "No projection rows were created from any source file."
        )

    universal["projection"] = pd.to_numeric(
        universal["projection"],
        errors="coerce",
    )

    for column in [
        "projection_lower_80",
        "projection_upper_80",
        "residual_standard_deviation",
        "validation_mae",
        "batting_order",
        "expected_plate_appearances",
        "days_rest",
    ]:
        universal[column] = pd.to_numeric(
            universal[column],
            errors="coerce",
        )

    universal["player"] = (
        universal["player"]
        .astype("string")
        .str.strip()
    )

    universal["market"] = (
        universal["market"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    universal = universal.dropna(
        subset=[
            "player",
            "market",
            "projection",
        ]
    ).copy()

    universal = universal.loc[
        universal["player"].ne("")
        & universal["market"].ne("")
        & universal["projection"].ge(0)
    ].copy()

    duplicate_columns = [
        "slate_date",
        "game_id",
        "player",
        "market",
    ]

    universal = universal.sort_values(
        [
            "slate_date",
            "game_id",
            "player",
            "market",
            "validation_mae",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
        na_position="last",
    )

    universal = universal.drop_duplicates(
        subset=duplicate_columns,
        keep="first",
    )

    universal = universal.sort_values(
        [
            "market",
            "projection",
            "player",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    for column in [
        "projection",
        "projection_lower_80",
        "projection_upper_80",
        "residual_standard_deviation",
        "validation_mae",
        "expected_plate_appearances",
        "days_rest",
    ]:
        universal[column] = universal[
            column
        ].round(4)

    return universal[OUTPUT_COLUMNS]


def build_universal_mlb_projections() -> pd.DataFrame:
    """Create the unified current-slate MLB projection table."""
    target_date = get_target_date()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Building universal MLB projection table")
    print(f"Slate date: {target_date.isoformat()}")
    print("=" * 72)

    rows: list[dict[str, Any]] = []

    add_hitter_markets(rows, target_date)
    add_pitcher_strikeouts(rows, target_date)
    add_pitcher_outs(rows, target_date)
    add_pitcher_fantasy(rows, target_date)

    universal = finalize_universal_table(rows)

    temporary_path = OUTPUT_PATH.with_suffix(".tmp.csv")

    universal.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(OUTPUT_PATH)

    print("\nUniversal MLB projection table completed.")
    print(f"Projection rows: {len(universal):,}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nRows by market:")
    print(
        universal.groupby("market")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    preview_columns = [
        "player",
        "team",
        "opponent",
        "market",
        "projection",
        "projection_lower_80",
        "projection_upper_80",
        "model_name",
        "validation_mae",
    ]

    print("\nProjection preview:")
    print(
        universal[preview_columns]
        .head(50)
        .to_string(index=False)
    )

    return universal


if __name__ == "__main__":
    build_universal_mlb_projections()
