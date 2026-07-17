"""Automatically grade historical MLB prop recommendations.

Input:
    outputs/history/mlb_bet_results.csv

Outputs:
    outputs/history/mlb_bet_results.csv
    outputs/backtesting/mlb_grading_audit.csv

The script:
1. Finds unresolved historical recommendations.
2. Downloads final MLB games for each unresolved date.
3. Reads official game box scores.
4. Matches players conservatively.
5. Calculates the actual result for each supported market.
6. Updates the historical results file without grading ambiguous matches.

Optional environment variable:
    MLB_GRADE_DATE=YYYY-MM-DD

When MLB_GRADE_DATE is set, only that date is processed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[3]

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "history"
    / "mlb_bet_results.csv"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
    / "mlb_grading_audit.csv"
)

BACKTEST_SCRIPT = (
    PROJECT_ROOT
    / "sports"
    / "mlb"
    / "backtesting"
    / "backtest_daily_card.py"
)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_PAUSE_SECONDS = 0.15

# Must match build_hitter_training_dataset.py.
SINGLE_PTS = 3.0
DOUBLE_PTS = 5.0
TRIPLE_PTS = 8.0
HOME_RUN_PTS = 10.0
RUN_PTS = 2.0
RBI_PTS = 2.0
WALK_PTS = 2.0
HBP_PTS = 2.0
STOLEN_BASE_PTS = 5.0

SUPPORTED_MARKETS = {
    "hitter_hits",
    "hitter_total_bases",
    "hitter_runs",
    "hitter_rbis",
    "hitter_hits_runs_rbis",
    "hitter_fantasy_score",
    "pitcher_strikeouts",
    "pitcher_outs",
}

FINAL_GAME_STATES = {
    "final",
    "game over",
    "completed early",
}

POSTPONED_GAME_STATES = {
    "postponed",
    "cancelled",
    "canceled",
    "suspended",
}

AUDIT_COLUMNS = [
    "event_date",
    "history_index",
    "player",
    "market",
    "direction",
    "line",
    "event_id",
    "matched_game_pk",
    "actual_result",
    "grading_status",
    "grading_note",
    "graded_at",
]


def build_session() -> requests.Session:
    """Create an HTTP session with retry protection."""
    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "mlb-ai-model-result-grader/1.0"
            ),
        }
    )

    return session


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Download and validate one MLB API response."""
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError(
            f"MLB API returned invalid JSON from {url}."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Unexpected MLB API response type from "
            f"{url}: {type(payload).__name__}"
        )

    return payload


def normalize_player_name(
    value: Any,
) -> str:
    """Normalize names for conservative cross-source matching."""
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
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


def normalize_market(
    value: Any,
) -> str:
    """Normalize known market aliases."""
    cleaned = str(value).strip().casefold()

    aliases = {
        "hits": "hitter_hits",
        "hitter hits": "hitter_hits",
        "batter_hits": "hitter_hits",
        "total bases": "hitter_total_bases",
        "hitter total bases": "hitter_total_bases",
        "batter_total_bases": "hitter_total_bases",
        "runs": "hitter_runs",
        "hitter runs": "hitter_runs",
        "batter_runs_scored": "hitter_runs",
        "rbis": "hitter_rbis",
        "rbi": "hitter_rbis",
        "hitter rbis": "hitter_rbis",
        "batter_rbis": "hitter_rbis",
        "hits+runs+rbis": (
            "hitter_hits_runs_rbis"
        ),
        "hits runs rbis": (
            "hitter_hits_runs_rbis"
        ),
        "h+r+rbi": (
            "hitter_hits_runs_rbis"
        ),
        "hitter fantasy score": (
            "hitter_fantasy_score"
        ),
        "batter_fantasy_score": (
            "hitter_fantasy_score"
        ),
        "strikeouts": (
            "pitcher_strikeouts"
        ),
        "pitcher strikeouts": (
            "pitcher_strikeouts"
        ),
        "outs": "pitcher_outs",
        "pitcher outs": "pitcher_outs",
    }

    return aliases.get(
        cleaned,
        cleaned.replace(
            " ",
            "_",
        ),
    )


def normalize_direction(
    value: Any,
) -> str:
    """Normalize Over/Under-style direction names."""
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

    return ""


def get_requested_grade_date() -> date | None:
    """Read an optional one-date grading restriction."""
    raw_value = os.getenv(
        "MLB_GRADE_DATE"
    )

    if not raw_value:
        return None

    try:
        return datetime.strptime(
            raw_value,
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ValueError(
            "MLB_GRADE_DATE must use YYYY-MM-DD format. "
            f"Received: {raw_value!r}"
        ) from exc


def ensure_history_columns(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Add grading metadata columns without deleting existing data."""
    result = history.copy()

    default_columns: dict[str, Any] = {
        "event_date": pd.NA,
        "player": pd.NA,
        "market": pd.NA,
        "direction": pd.NA,
        "line": pd.NA,
        "event_id": pd.NA,
        "game_id": pd.NA,
        "actual_result": pd.NA,
        "outcome": pd.NA,
        "grading_status": "UNRESOLVED",
        "grading_note": pd.NA,
        "matched_game_pk": pd.NA,
        "graded_at": pd.NA,
        "stake": 1.0,
        "profit": pd.NA,
    }

    for column, default_value in (
        default_columns.items()
    ):
        if column not in result.columns:
            result[column] = default_value

    return result


def load_history() -> pd.DataFrame:
    """Load the historical recommendations file."""
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            "Historical recommendations were not "
            f"found: {HISTORY_PATH}"
        )

    try:
        history = pd.read_csv(
            HISTORY_PATH
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError(
            "Could not read historical "
            f"recommendations: {HISTORY_PATH}"
        ) from exc

    required_columns = {
        "event_date",
        "player",
        "market",
        "direction",
        "line",
    }

    missing_columns = (
        required_columns
        - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "Historical recommendations are "
            "missing columns: "
            f"{sorted(missing_columns)}"
        )

    history = ensure_history_columns(
        history
    )

    history["event_date"] = pd.to_datetime(
        history["event_date"],
        errors="coerce",
    ).dt.date

    # Numeric columns
numeric_columns = [
    "line",
    "actual_result",
    "sportsbook_odds",
    "stake",
    "profit",
]

for column in numeric_columns:
    if column not in history.columns:
        history[column] = pd.NA

    history[column] = pd.to_numeric(
        history[column],
        errors="coerce",
    )

# Text columns
text_columns = [
    "outcome",
    "grading_status",
    "grading_note",
    "graded_at",
    "event_id",
    "game_id",
]

for column in text_columns:
    if column not in history.columns:
        history[column] = pd.NA

    history[column] = history[column].astype("object")

    history["market"] = history[
        "market"
    ].apply(normalize_market)

    history["direction"] = history[
        "direction"
    ].apply(normalize_direction)

    history["player_key"] = history[
        "player"
    ].apply(normalize_player_name)

    return history


def get_schedule(
    session: requests.Session,
    event_date: date,
) -> list[dict[str, Any]]:
    """Download all MLB games for one date."""
    payload = fetch_json(
        session,
        f"{MLB_API_BASE}/schedule",
        params={
            "sportId": 1,
            "date": event_date.isoformat(),
        },
    )

    games: list[dict[str, Any]] = []

    for date_record in payload.get(
        "dates",
        [],
    ):
        if not isinstance(
            date_record,
            dict,
        ):
            continue

        date_games = date_record.get(
            "games",
            [],
        )

        if not isinstance(
            date_games,
            list,
        ):
            continue

        for game in date_games:
            if isinstance(
                game,
                dict,
            ):
                games.append(game)

    return games


def game_status(
    game: dict[str, Any],
) -> str:
    """Return a normalized detailed game status."""
    status = game.get(
        "status",
        {},
    )

    if not isinstance(
        status,
        dict,
    ):
        return ""

    detailed_state = status.get(
        "detailedState"
    )

    abstract_state = status.get(
        "abstractGameState"
    )

    value = (
        detailed_state
        or abstract_state
        or ""
    )

    return str(value).strip().casefold()


def game_is_final(
    game: dict[str, Any],
) -> bool:
    """Return whether a game has a final box score."""
    status = game_status(game)

    return (
        status in FINAL_GAME_STATES
        or status.startswith("final")
    )


def game_is_postponed(
    game: dict[str, Any],
) -> bool:
    """Return whether a game will not have a normal final result."""
    status = game_status(game)

    return (
        status in POSTPONED_GAME_STATES
        or any(
            word in status
            for word in POSTPONED_GAME_STATES
        )
    )


def get_boxscore(
    session: requests.Session,
    game_pk: int,
) -> dict[str, Any]:
    """Download one game's complete box score."""
    return fetch_json(
        session,
        (
            f"{MLB_API_BASE}/game/"
            f"{game_pk}/boxscore"
        ),
    )


def safe_number(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to finite float."""
    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

    if not np.isfinite(number):
        return default

    return number


def innings_to_outs(
    value: Any,
) -> float:
    """Convert an innings-pitched string into recorded outs.

    Examples:
        5.0 -> 15 outs
        5.1 -> 16 outs
        5.2 -> 17 outs
    """
    if value is None or pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if not text:
        return float("nan")

    if "." in text:
        whole_text, partial_text = (
            text.split(
                ".",
                1,
            )
        )
    else:
        whole_text = text
        partial_text = "0"

    try:
        full_innings = int(
            whole_text
        )

        partial_outs = int(
            partial_text[:1]
            or "0"
        )
    except ValueError:
        return float("nan")

    if partial_outs not in {
        0,
        1,
        2,
    }:
        return float("nan")

    return float(
        full_innings * 3
        + partial_outs
    )


def calculate_total_bases(
    batting: dict[str, Any],
) -> float:
    """Calculate total bases if the direct field is unavailable."""
    direct_value = batting.get(
        "totalBases"
    )

    if direct_value is not None:
        return safe_number(
            direct_value
        )

    hits = safe_number(
        batting.get("hits")
    )

    doubles = safe_number(
        batting.get("doubles")
    )

    triples = safe_number(
        batting.get("triples")
    )

    home_runs = safe_number(
        batting.get("homeRuns")
    )

    singles = max(
        0.0,
        hits
        - doubles
        - triples
        - home_runs,
    )

    return (
        singles
        + (2.0 * doubles)
        + (3.0 * triples)
        + (4.0 * home_runs)
    )


def calculate_hitter_fantasy_score(
    batting: dict[str, Any],
) -> float:
    """Calculate hitter fantasy score using the model's scoring system."""
    hits = safe_number(
        batting.get("hits")
    )

    doubles = safe_number(
        batting.get("doubles")
    )

    triples = safe_number(
        batting.get("triples")
    )

    home_runs = safe_number(
        batting.get("homeRuns")
    )

    runs = safe_number(
        batting.get("runs")
    )

    rbi = safe_number(
        batting.get("rbi")
    )

    walks = safe_number(
        batting.get(
            "baseOnBalls",
            batting.get("walks"),
        )
    )

    hit_by_pitch = safe_number(
        batting.get(
            "hitByPitch"
        )
    )

    stolen_bases = safe_number(
        batting.get(
            "stolenBases"
        )
    )

    singles = max(
        0.0,
        hits
        - doubles
        - triples
        - home_runs,
    )

    return (
        singles * SINGLE_PTS
        + doubles * DOUBLE_PTS
        + triples * TRIPLE_PTS
        + home_runs * HOME_RUN_PTS
        + runs * RUN_PTS
        + rbi * RBI_PTS
        + walks * WALK_PTS
        + hit_by_pitch * HBP_PTS
        + stolen_bases
        * STOLEN_BASE_PTS
    )


def build_player_game_stats(
    boxscore: dict[str, Any],
    game_pk: int,
    event_date: date,
) -> list[dict[str, Any]]:
    """Extract hitter and pitcher stat lines from one final game."""
    rows: list[dict[str, Any]] = []

    teams = boxscore.get(
        "teams",
        {},
    )

    if not isinstance(
        teams,
        dict,
    ):
        return rows

    for side in (
        "away",
        "home",
    ):
        team_section = teams.get(
            side,
            {},
        )

        if not isinstance(
            team_section,
            dict,
        ):
            continue

        team_info = team_section.get(
            "team",
            {},
        )

        if not isinstance(
            team_info,
            dict,
        ):
            team_info = {}

        team_name = team_info.get(
            "name"
        )

        players = team_section.get(
            "players",
            {},
        )

        if not isinstance(
            players,
            dict,
        ):
            continue

        for player_record in (
            players.values()
        ):
            if not isinstance(
                player_record,
                dict,
            ):
                continue

            person = player_record.get(
                "person",
                {},
            )

            if not isinstance(
                person,
                dict,
            ):
                person = {}

            player_name = person.get(
                "fullName"
            )

            if not player_name:
                continue

            stats = player_record.get(
                "stats",
                {},
            )

            if not isinstance(
                stats,
                dict,
            ):
                stats = {}

            batting = stats.get(
                "batting",
                {},
            )

            pitching = stats.get(
                "pitching",
                {},
            )

            if not isinstance(
                batting,
                dict,
            ):
                batting = {}

            if not isinstance(
                pitching,
                dict,
            ):
                pitching = {}

            has_batting_appearance = (
                safe_number(
                    batting.get(
                        "plateAppearances"
                    )
                )
                > 0
                or safe_number(
                    batting.get(
                        "atBats"
                    )
                )
                > 0
            )

            has_pitching_appearance = (
                pitching.get(
                    "inningsPitched"
                )
                is not None
                or safe_number(
                    pitching.get(
                        "battersFaced"
                    )
                )
                > 0
            )

            rows.append(
                {
                    "event_date": (
                        event_date
                    ),
                    "game_pk": game_pk,
                    "player_id": (
                        person.get("id")
                    ),
                    "player": player_name,
                    "player_key": (
                        normalize_player_name(
                            player_name
                        )
                    ),
                    "team": team_name,
                    "side": side,
                    "has_batting_appearance": (
                        has_batting_appearance
                    ),
                    "has_pitching_appearance": (
                        has_pitching_appearance
                    ),
                    "hitter_hits": (
                        safe_number(
                            batting.get(
                                "hits"
                            )
                        )
                    ),
                    "hitter_total_bases": (
                        calculate_total_bases(
                            batting
                        )
                    ),
                    "hitter_runs": (
                        safe_number(
                            batting.get(
                                "runs"
                            )
                        )
                    ),
                    "hitter_rbis": (
                        safe_number(
                            batting.get(
                                "rbi"
                            )
                        )
                    ),
                    "hitter_hits_runs_rbis": (
                        safe_number(
                            batting.get(
                                "hits"
                            )
                        )
                        + safe_number(
                            batting.get(
                                "runs"
                            )
                        )
                        + safe_number(
                            batting.get(
                                "rbi"
                            )
                        )
                    ),
                    "hitter_fantasy_score": (
                        calculate_hitter_fantasy_score(
                            batting
                        )
                    ),
                    "pitcher_strikeouts": (
                        safe_number(
                            pitching.get(
                                "strikeOuts"
                            )
                        )
                    ),
                    "pitcher_outs": (
                        innings_to_outs(
                            pitching.get(
                                "inningsPitched"
                            )
                        )
                    ),
                }
            )

    return rows


def collect_date_stats(
    session: requests.Session,
    event_date: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect all final-game player stats for one date."""
    games = get_schedule(
        session,
        event_date,
    )

    metadata = {
        "scheduled_games": len(games),
        "final_games": 0,
        "unfinished_games": 0,
        "postponed_games": 0,
    }

    rows: list[
        dict[str, Any]
    ] = []

    for game in games:
        game_pk = game.get(
            "gamePk"
        )

        if game_pk is None:
            continue

        try:
            game_pk = int(game_pk)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if game_is_postponed(
            game
        ):
            metadata[
                "postponed_games"
            ] += 1

            continue

        if not game_is_final(
            game
        ):
            metadata[
                "unfinished_games"
            ] += 1

            continue

        metadata[
            "final_games"
        ] += 1

        try:
            boxscore = get_boxscore(
                session,
                game_pk,
            )
        except requests.RequestException as exc:
            print(
                "Could not download "
                f"box score {game_pk}: {exc}"
            )

            continue

        rows.extend(
            build_player_game_stats(
                boxscore=boxscore,
                game_pk=game_pk,
                event_date=event_date,
            )
        )

        time.sleep(
            REQUEST_PAUSE_SECONDS
        )

    return (
        pd.DataFrame(rows),
        metadata,
    )


def extract_requested_game_pk(
    history_row: pd.Series,
) -> int | None:
    """Read an MLB game identifier when the history contains one."""
    for column in [
        "matched_game_pk",
        "game_id",
        "event_id",
    ]:
        value = history_row.get(
            column
        )

        if value is None or pd.isna(value):
            continue

        text = str(value).strip()

        if text.isdigit():
            return int(text)

    return None


def find_matching_stat_row(
    history_row: pd.Series,
    date_stats: pd.DataFrame,
) -> tuple[
    pd.Series | None,
    str,
    str,
]:
    """Conservatively match one historical recommendation."""
    if date_stats.empty:
        return (
            None,
            "UNRESOLVED",
            (
                "No completed game box "
                "scores were available."
            ),
        )

    player_key = history_row.get(
        "player_key",
        "",
    )

    candidates = date_stats.loc[
        date_stats[
            "player_key"
        ].eq(player_key)
    ].copy()

    if candidates.empty:
        return (
            None,
            "UNMATCHED_PLAYER",
            (
                "Player was not found in "
                "any final box score."
            ),
        )

    market = history_row.get(
        "market",
        "",
    )

    requested_game_pk = (
        extract_requested_game_pk(
            history_row
        )
    )

    if requested_game_pk is not None:
        game_candidates = (
            candidates.loc[
                candidates[
                    "game_pk"
                ].eq(
                    requested_game_pk
                )
            ].copy()
        )

        if game_candidates.empty:
            return (
                None,
                "GAME_ID_MISMATCH",
                (
                    "Player was found, but "
                    "not in the saved game ID."
                ),
            )

        candidates = game_candidates

    if market.startswith(
        "hitter_"
    ):
        candidates = candidates.loc[
            candidates[
                "has_batting_appearance"
            ].eq(True)
        ].copy()

    if market.startswith(
        "pitcher_"
    ):
        candidates = candidates.loc[
            candidates[
                "has_pitching_appearance"
            ].eq(True)
        ].copy()

    if candidates.empty:
        return (
            None,
            "DID_NOT_APPEAR",
            (
                "Player was listed but did "
                "not record the required "
                "appearance."
            ),
        )

    game_count = candidates[
        "game_pk"
    ].nunique()

    if (
        game_count > 1
        and requested_game_pk is None
    ):
        return (
            None,
            "AMBIGUOUS_DOUBLEHEADER",
            (
                "Player appeared in multiple "
                "games and history lacks a "
                "game ID."
            ),
        )

    candidates = (
        candidates
        .sort_values("game_pk")
        .drop_duplicates(
            subset=[
                "game_pk",
                "player_key",
            ],
            keep="last",
        )
    )

    return (
        candidates.iloc[0],
        "MATCHED",
        "",
    )


def calculate_actual_result(
    stat_row: pd.Series,
    market: str,
) -> float:
    """Return the actual statistic for a supported market."""
    if market not in SUPPORTED_MARKETS:
        return float("nan")

    value = pd.to_numeric(
        stat_row.get(market),
        errors="coerce",
    )

    if (
        pd.isna(value)
        or not np.isfinite(value)
    ):
        return float("nan")

    return float(value)


def grade_bet_outcome(
    direction: str,
    line: float,
    actual_result: float,
) -> str:
    """Return WIN, LOSS, or PUSH."""
    if direction == "Over":
        if actual_result > line:
            return "WIN"

        if actual_result < line:
            return "LOSS"

        return "PUSH"

    if direction == "Under":
        if actual_result < line:
            return "WIN"

        if actual_result > line:
            return "LOSS"

        return "PUSH"

    return "UNRESOLVED"


def calculate_profit(
    outcome: str,
    odds: Any,
    stake: Any,
) -> float:
    """Calculate profit using American odds and the saved stake."""
    stake_value = pd.to_numeric(
        stake,
        errors="coerce",
    )

    odds_value = pd.to_numeric(
        odds,
        errors="coerce",
    )

    if pd.isna(stake_value):
        stake_value = 1.0

    stake_value = float(
        stake_value
    )

    if outcome == "PUSH":
        return 0.0

    if outcome == "LOSS":
        return -stake_value

    if outcome != "WIN":
        return float("nan")

    if (
        pd.isna(odds_value)
        or float(odds_value) == 0
    ):
        return stake_value

    odds_value = float(
        odds_value
    )

    if odds_value > 0:
        return stake_value * (
            odds_value / 100.0
        )

    return stake_value * (
        100.0 / abs(odds_value)
    )


def grade_history_row(
    history_row: pd.Series,
    date_stats: pd.DataFrame,
    graded_at: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Grade one historical recommendation."""
    market = history_row.get(
        "market",
        "",
    )

    history_index = int(
        history_row.name
    )

    audit = {
        "event_date": (
            history_row.get(
                "event_date"
            )
        ),
        "history_index": (
            history_index
        ),
        "player": (
            history_row.get(
                "player"
            )
        ),
        "market": market,
        "direction": (
            history_row.get(
                "direction"
            )
        ),
        "line": (
            history_row.get(
                "line"
            )
        ),
        "event_id": (
            history_row.get(
                "event_id"
            )
        ),
        "matched_game_pk": pd.NA,
        "actual_result": pd.NA,
        "grading_status": (
            "UNRESOLVED"
        ),
        "grading_note": "",
        "graded_at": graded_at,
    }

    if market not in SUPPORTED_MARKETS:
        audit[
            "grading_status"
        ] = "UNSUPPORTED_MARKET"

        audit[
            "grading_note"
        ] = (
            "Automatic grading is not "
            "configured for this market."
        )

        return (
            {},
            audit,
        )

    line = pd.to_numeric(
        history_row.get("line"),
        errors="coerce",
    )

    if pd.isna(line):
        audit[
            "grading_status"
        ] = "INVALID_LINE"

        audit[
            "grading_note"
        ] = (
            "The saved recommendation "
            "does not have a valid line."
        )

        return (
            {},
            audit,
        )

    direction = history_row.get(
        "direction",
        "",
    )

    if direction not in {
        "Over",
        "Under",
    }:
        audit[
            "grading_status"
        ] = "INVALID_DIRECTION"

        audit[
            "grading_note"
        ] = (
            "The saved direction is not "
            "Over or Under."
        )

        return (
            {},
            audit,
        )

    (
        stat_row,
        match_status,
        match_note,
    ) = find_matching_stat_row(
        history_row,
        date_stats,
    )

    if stat_row is None:
        audit[
            "grading_status"
        ] = match_status

        audit[
            "grading_note"
        ] = match_note

        return (
            {},
            audit,
        )

    actual_result = (
        calculate_actual_result(
            stat_row,
            market,
        )
    )

    if not np.isfinite(
        actual_result
    ):
        audit[
            "grading_status"
        ] = "MISSING_STAT"

        audit[
            "grading_note"
        ] = (
            "The player matched, but the "
            "required statistic was "
            "unavailable."
        )

        return (
            {},
            audit,
        )

    outcome = grade_bet_outcome(
        direction=direction,
        line=float(line),
        actual_result=actual_result,
    )

    game_pk = int(
        stat_row["game_pk"]
    )

    profit = calculate_profit(
        outcome=outcome,
        odds=history_row.get(
            "sportsbook_odds"
        ),
        stake=history_row.get(
            "stake",
            1.0,
        ),
    )

    updates = {
        "actual_result": (
            actual_result
        ),
        "outcome": outcome,
        "profit": profit,
        "grading_status": "GRADED",
        "grading_note": "",
        "matched_game_pk": game_pk,
        "graded_at": graded_at,
    }

    audit.update(
        {
            "matched_game_pk": (
                game_pk
            ),
            "actual_result": (
                actual_result
            ),
            "grading_status": (
                "GRADED"
            ),
            "grading_note": outcome,
        }
    )

    return (
        updates,
        audit,
    )


def save_history(
    history: pd.DataFrame,
) -> None:
    """Save history atomically without helper matching columns."""
    output = history.drop(
        columns=[
            "player_key"
        ],
        errors="ignore",
    ).copy()

    output[
        "event_date"
    ] = output[
        "event_date"
    ].apply(
        lambda value: (
            value.isoformat()
            if isinstance(
                value,
                date,
            )
            else value
        )
    )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        HISTORY_PATH.with_suffix(
            ".tmp.csv"
        )
    )

    output.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        HISTORY_PATH
    )


def save_audit(
    audit_rows: list[
        dict[str, Any]
    ],
) -> None:
    """Save a detailed report of every grading attempt."""
    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit = pd.DataFrame(
        audit_rows,
        columns=AUDIT_COLUMNS,
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )


def run_backtest_if_requested() -> None:
    """Optionally run the backtester after grading."""
    should_run = os.getenv(
        "MLB_RUN_BACKTEST_AFTER_GRADING",
        "0",
    ).strip().casefold()

    if should_run not in {
        "1",
        "true",
        "yes",
    }:
        return

    if not BACKTEST_SCRIPT.exists():
        print(
            "Backtest was requested, but "
            "the script was not found: "
            f"{BACKTEST_SCRIPT}"
        )

        return

    print(
        "\nRunning MLB backtest..."
    )

    subprocess.run(
        [
            sys.executable,
            str(BACKTEST_SCRIPT),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def grade_daily_results() -> pd.DataFrame:
    """Grade all eligible unresolved MLB recommendations."""
    history = load_history()

    requested_date = (
        get_requested_grade_date()
    )

    today = datetime.now(
        timezone.utc
    ).date()

    unresolved_mask = history[
        "actual_result"
    ].isna()

    eligible_mask = (
        unresolved_mask
        & history[
            "event_date"
        ].notna()
        & history[
            "event_date"
        ].le(today)
    )

    if requested_date is not None:
        eligible_mask &= history[
            "event_date"
        ].eq(requested_date)

    eligible = history.loc[
        eligible_mask
    ].copy()

    print("=" * 72)
    print(
        "MLB AUTOMATIC RESULT GRADER"
    )
    print("=" * 72)

    print(
        f"History rows: "
        f"{len(history):,}"
    )

    print(
        "Eligible unresolved rows: "
        f"{len(eligible):,}"
    )

    if requested_date is not None:
        print(
            "Restricted grading date: "
            f"{requested_date.isoformat()}"
        )

    if eligible.empty:
        print(
            "No eligible unresolved "
            "recommendations were found."
        )

        save_audit([])

        return history

    session = build_session()

    graded_at = datetime.now(
        timezone.utc
    ).isoformat()

    audit_rows: list[
        dict[str, Any]
    ] = []

    total_graded = 0

    unresolved_dates = sorted(
        eligible[
            "event_date"
        ]
        .dropna()
        .unique()
    )

    for event_date in unresolved_dates:
        print(
            "\nProcessing "
            f"{event_date.isoformat()}..."
        )

        try:
            (
                date_stats,
                metadata,
            ) = collect_date_stats(
                session,
                event_date,
            )
        except requests.RequestException as exc:
            print(
                "Could not download games "
                f"for {event_date}: {exc}"
            )

            continue

        print(
            "Games — "
            f"scheduled: "
            f"{metadata['scheduled_games']}, "
            f"final: "
            f"{metadata['final_games']}, "
            f"unfinished: "
            f"{metadata['unfinished_games']}, "
            f"postponed: "
            f"{metadata['postponed_games']}"
        )

        date_indices = (
            eligible.index[
                eligible[
                    "event_date"
                ].eq(event_date)
            ]
        )

        for history_index in date_indices:
            history_row = history.loc[
                history_index
            ]

            (
                updates,
                audit,
            ) = grade_history_row(
                history_row=history_row,
                date_stats=date_stats,
                graded_at=graded_at,
            )

            audit_rows.append(
                audit
            )

            if not updates:
                continue

            for column, value in (
                updates.items()
            ):
                history.at[
                    history_index,
                    column,
                ] = value

            total_graded += 1

        time.sleep(
            REQUEST_PAUSE_SECONDS
        )

    save_history(
        history
    )

    save_audit(
        audit_rows
    )

    audit_frame = pd.DataFrame(
        audit_rows,
        columns=AUDIT_COLUMNS,
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "MLB GRADING COMPLETE"
    )

    print("=" * 72)

    print(
        "Successfully graded: "
        f"{total_graded:,}"
    )

    print(
        "Updated history: "
        f"{HISTORY_PATH}"
    )

    print(
        "Saved grading audit: "
        f"{AUDIT_PATH}"
    )

    if not audit_frame.empty:
        print(
            "\nGrading status breakdown:"
        )

        print(
            audit_frame[
                "grading_status"
            ]
            .value_counts()
            .to_string()
        )

        graded_rows = (
            audit_frame.loc[
                audit_frame[
                    "grading_status"
                ].eq("GRADED")
            ]
        )

        if not graded_rows.empty:
            print(
                "\nGraded markets:"
            )

            print(
                graded_rows[
                    "market"
                ]
                .value_counts()
                .to_string()
            )

    remaining_unresolved = int(
        history[
            "actual_result"
        ]
        .isna()
        .sum()
    )

    print(
        "\nRemaining unresolved "
        "history rows: "
        f"{remaining_unresolved:,}"
    )

    run_backtest_if_requested()

    return history


if __name__ == "__main__":
    grade_daily_results()
