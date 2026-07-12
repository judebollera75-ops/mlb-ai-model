"""Log current MLB recommendations into permanent bet history.

Input:
    outputs/mlb_daily_card.csv

Output:
    outputs/history/mlb_bet_results.csv

The logger preserves game, platform, pricing, model, and grading metadata so
recommendations can later be matched to official MLB box scores and evaluated
without ambiguity.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DAILY_CARD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mlb_daily_card.csv"
)

HISTORY_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "history"
)

HISTORY_PATH = (
    HISTORY_DIRECTORY
    / "mlb_bet_results.csv"
)

HISTORY_COLUMNS = [
    "bet_id",
    "event_date",
    "slate_date",
    "event_id",
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "team",
    "opponent",
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
    "grade",
    "confidence_tier",
    "distribution_method",
    "calibration_sample_size",
    "validation_mae",
    "kelly_fraction",
    "recommended_bankroll_fraction",
    "fetched_at",
    "logged_at",
    "stake",
    "closing_line",
    "closing_odds",
    "actual_result",
    "outcome",
    "profit",
    "grading_status",
    "grading_note",
    "matched_game_pk",
    "graded_at",
]

UNIQUE_BET_COLUMNS = [
    "event_date",
    "event_id",
    "game_id",
    "platform",
    "player",
    "market",
    "direction",
    "line",
]


def get_target_date() -> str:
    """Return the active slate date."""
    raw_value = os.getenv(
        "MLB_TARGET_DATE",
        date.today().isoformat(),
    )

    try:
        parsed = datetime.strptime(
            raw_value,
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc

    return parsed.isoformat()


def empty_history() -> pd.DataFrame:
    """Return an empty history table with the production schema."""
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def load_history() -> pd.DataFrame:
    """Load the existing history while preserving old columns."""
    if not HISTORY_PATH.exists():
        return empty_history()

    try:
        history = pd.read_csv(HISTORY_PATH)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return empty_history()

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA

    extra_columns = [
        column
        for column in history.columns
        if column not in HISTORY_COLUMNS
    ]

    return history[
        HISTORY_COLUMNS + extra_columns
    ].copy()


def load_daily_card() -> pd.DataFrame:
    """Load the current daily card."""
    if not DAILY_CARD_PATH.exists():
        raise FileNotFoundError(
            f"Daily card was not found: {DAILY_CARD_PATH}"
        )

    try:
        card = pd.read_csv(DAILY_CARD_PATH)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()

    return card


def clean_text(value: Any) -> Any:
    """Strip text values without converting missing values to strings."""
    if value is None or pd.isna(value):
        return pd.NA

    cleaned = str(value).strip()

    return cleaned if cleaned else pd.NA


def normalize_direction(value: Any) -> Any:
    """Normalize platform direction names."""
    if value is None or pd.isna(value):
        return pd.NA

    cleaned = str(value).strip().casefold()

    if cleaned in {
        "over",
        "more",
        "yes",
        "higher",
        "more/yes",
    }:
        return "Over"

    if cleaned in {
        "under",
        "less",
        "no",
        "lower",
        "less/no",
    }:
        return "Under"

    return clean_text(value)


def normalize_market(value: Any) -> Any:
    """Normalize market names into the production naming scheme."""
    if value is None or pd.isna(value):
        return pd.NA

    cleaned = str(value).strip().casefold()

    aliases = {
        "hits": "hitter_hits",
        "hitter hits": "hitter_hits",
        "total bases": "hitter_total_bases",
        "hitter total bases": "hitter_total_bases",
        "runs": "hitter_runs",
        "hitter runs": "hitter_runs",
        "rbi": "hitter_rbis",
        "rbis": "hitter_rbis",
        "hitter rbi": "hitter_rbis",
        "hitter rbis": "hitter_rbis",
        "hits+runs+rbis": "hitter_hits_runs_rbis",
        "hits runs rbis": "hitter_hits_runs_rbis",
        "h+r+rbi": "hitter_hits_runs_rbis",
        "hitter fantasy score": "hitter_fantasy_score",
        "strikeouts": "pitcher_strikeouts",
        "pitcher strikeouts": "pitcher_strikeouts",
        "outs": "pitcher_outs",
        "pitcher outs": "pitcher_outs",
    }

    return aliases.get(
        cleaned,
        cleaned.replace(" ", "_"),
    )


def first_existing_value(
    row: pd.Series,
    candidates: list[str],
) -> Any:
    """Return the first non-empty value from candidate columns."""
    for column in candidates:
        if column not in row.index:
            continue

        value = row.get(column)

        if value is None or pd.isna(value):
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return pd.NA


def make_bet_id(row: pd.Series) -> str:
    """Create a stable ID for one exact recommendation."""
    values = [
        row.get("event_date"),
        row.get("event_id"),
        row.get("game_id"),
        row.get("platform"),
        row.get("player"),
        row.get("market"),
        row.get("direction"),
        row.get("line"),
    ]

    normalized = [
        "" if pd.isna(value) else str(value).strip().casefold()
        for value in values
    ]

    return "|".join(normalized)


def prepare_daily_rows(
    card: pd.DataFrame,
) -> pd.DataFrame:
    """Convert the daily card into the permanent history schema."""
    if card.empty:
        return empty_history()

    target_date = get_target_date()
    logged_at = pd.Timestamp.now(tz="UTC").isoformat()

    prepared_rows: list[dict[str, Any]] = []

    for _, row in card.iterrows():
        event_date = first_existing_value(
            row,
            [
                "event_date",
                "slate_date",
                "date",
            ],
        )

        if pd.isna(event_date):
            event_date = target_date

        prepared_row: dict[str, Any] = {
            column: row.get(column, pd.NA)
            for column in HISTORY_COLUMNS
        }

        prepared_row.update(
            {
                "event_date": event_date,
                "slate_date": first_existing_value(
                    row,
                    ["slate_date", "event_date", "date"],
                ),
                "event_id": first_existing_value(
                    row,
                    ["event_id", "game_id"],
                ),
                "game_id": first_existing_value(
                    row,
                    ["game_id", "event_id"],
                ),
                "commence_time": row.get(
                    "commence_time",
                    pd.NA,
                ),
                "home_team": row.get(
                    "home_team",
                    pd.NA,
                ),
                "away_team": row.get(
                    "away_team",
                    pd.NA,
                ),
                "team": row.get("team", pd.NA),
                "opponent": row.get(
                    "opponent",
                    pd.NA,
                ),
                "platform": clean_text(
                    row.get("platform")
                ),
                "platform_key": clean_text(
                    row.get("platform_key")
                ),
                "player": clean_text(
                    row.get("player")
                ),
                "market": normalize_market(
                    row.get("market")
                ),
                "direction": normalize_direction(
                    row.get("direction")
                ),
                "logged_at": logged_at,
                "grading_status": "UNRESOLVED",
            }
        )

        prepared_rows.append(prepared_row)

    prepared = pd.DataFrame(
        prepared_rows,
        columns=HISTORY_COLUMNS,
    )

    numeric_columns = [
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
        "calibration_sample_size",
        "validation_mae",
        "kelly_fraction",
        "recommended_bankroll_fraction",
        "stake",
        "closing_line",
        "closing_odds",
        "actual_result",
        "profit",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared["event_date"] = pd.to_datetime(
        prepared["event_date"],
        errors="coerce",
    ).dt.date.astype("string")

    prepared["slate_date"] = pd.to_datetime(
        prepared["slate_date"],
        errors="coerce",
    ).dt.date.astype("string")

    prepared = prepared.dropna(
        subset=[
            "event_date",
            "player",
            "market",
            "direction",
            "line",
            "platform",
        ]
    ).copy()

    prepared["bet_id"] = prepared.apply(
        make_bet_id,
        axis=1,
    )

    prepared = prepared.drop_duplicates(
        subset=["bet_id"],
        keep="last",
    ).reset_index(drop=True)

    return prepared


def preserve_existing_grading(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure later logging never erases settled grading fields."""
    grading_columns = [
        "actual_result",
        "outcome",
        "profit",
        "grading_status",
        "grading_note",
        "matched_game_pk",
        "graded_at",
        "closing_line",
        "closing_odds",
    ]

    if "bet_id" not in combined.columns:
        return combined

    combined = combined.sort_values(
        "logged_at",
        na_position="first",
    )

    rows: list[pd.Series] = []

    for _, group in combined.groupby(
        "bet_id",
        dropna=False,
        sort=False,
    ):
        base = group.iloc[-1].copy()

        for column in grading_columns:
            if column not in group.columns:
                continue

            existing_values = group[column].dropna()

            if not existing_values.empty:
                base[column] = existing_values.iloc[-1]

        rows.append(base)

    return pd.DataFrame(rows)


def save_history(history: pd.DataFrame) -> None:
    """Write history atomically."""
    HISTORY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA

    extra_columns = [
        column
        for column in history.columns
        if column not in HISTORY_COLUMNS
    ]

    history = history[
        HISTORY_COLUMNS + extra_columns
    ].copy()

    temporary_path = HISTORY_PATH.with_suffix(
        ".tmp.csv"
    )

    history.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(HISTORY_PATH)


def append_new_picks() -> pd.DataFrame:
    """Append current recommendations without duplicating prior bets."""
    history = load_history()
    card = load_daily_card()
    prepared = prepare_daily_rows(card)

    print("=" * 72)
    print("MLB HISTORICAL BET LOGGER")
    print("=" * 72)
    print(f"Existing history rows: {len(history):,}")
    print(f"Current daily-card rows: {len(card):,}")
    print(f"Prepared valid rows: {len(prepared):,}")

    if prepared.empty:
        print(
            "No valid current recommendations were available. "
            "The existing history file was preserved."
        )

        save_history(history)
        return history

    for frame in [history, prepared]:
        if "bet_id" not in frame.columns:
            frame["bet_id"] = frame.apply(
                make_bet_id,
                axis=1,
            )

    existing_bet_ids = set(
        history["bet_id"]
        .dropna()
        .astype(str)
    )

    new_rows = prepared.loc[
        ~prepared["bet_id"]
        .astype(str)
        .isin(existing_bet_ids)
    ].copy()

    combined = pd.concat(
        [history, new_rows],
        ignore_index=True,
        sort=False,
    )

    combined = preserve_existing_grading(
        combined
    )

    combined = combined.sort_values(
        [
            "event_date",
            "commence_time",
            "player",
            "market",
            "platform",
        ],
        na_position="last",
    ).reset_index(drop=True)

    save_history(combined)

    unresolved_count = int(
        pd.to_numeric(
            combined["actual_result"],
            errors="coerce",
        )
        .isna()
        .sum()
    )

    print("\nHistorical bet log updated.")
    print(f"New recommendations added: {len(new_rows):,}")
    print(f"Total historical rows: {len(combined):,}")
    print(f"Rows awaiting grading: {unresolved_count:,}")
    print(f"Saved to: {HISTORY_PATH}")

    if not new_rows.empty:
        print("\nNew rows by market:")
        print(
            new_rows["market"]
            .value_counts()
            .to_string()
        )

    return combined


if __name__ == "__main__":
    append_new_picks()
