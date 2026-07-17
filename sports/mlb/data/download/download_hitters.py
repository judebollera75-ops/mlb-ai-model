from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SCHEDULE_DIRECTORY = Path("data/schedules")
OUTPUT_DIRECTORY = Path("data/hitters")

MLB_API_BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT_SECONDS = 30
RECENT_LOOKBACK_DAYS = 21
MAX_RECENT_GAMES = 7

NON_HITTER_POSITIONS = {
    "P",
    "SP",
    "RP",
}


def parse_target_date(
    target_date: str | None,
) -> str:
    """Return a validated YYYY-MM-DD target date."""
    if target_date is None:
        return date.today().isoformat()

    try:
        return datetime.strptime(
            target_date,
            "%Y-%m-%d",
        ).date().isoformat()
    except ValueError as exc:
        raise ValueError(
            "target_date must use YYYY-MM-DD format. "
            f"Received: {target_date!r}"
        ) from exc


def create_session() -> requests.Session:
    """Create a reusable HTTP session."""
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 MLB hitter projection downloader"
            ),
            "Accept": "application/json",
        }
    )

    return session


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request JSON from MLB StatsAPI."""
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected response format from {url}"
        )

    return data


def load_schedule(
    target_date: str,
) -> pd.DataFrame:
    """Load and validate the current MLB schedule."""
    schedule_path = (
        SCHEDULE_DIRECTORY
        / f"{target_date}.csv"
    )

    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Missing schedule file: {schedule_path}"
        )

    schedule = pd.read_csv(
        schedule_path
    )

    required_columns = {
        "game_id",
        "away_team",
        "home_team",
        "status",
    }

    missing_columns = (
        required_columns
        - set(schedule.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{schedule_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    schedule["game_id"] = pd.to_numeric(
        schedule["game_id"],
        errors="coerce",
    )

    schedule = schedule.dropna(
        subset=[
            "game_id",
            "away_team",
            "home_team",
        ]
    ).copy()

    schedule["game_id"] = (
        schedule["game_id"]
        .astype("int64")
    )

    schedule = schedule.drop_duplicates(
        subset=["game_id"],
        keep="first",
    ).reset_index(
        drop=True
    )

    return schedule


def get_game_context(
    session: requests.Session,
    game_id: int,
) -> dict[str, Any]:
    """Load game metadata such as team IDs and venue."""
    url = (
    "https://statsapi.mlb.com/api/v1.1/"
    f"game/{game_id}/feed/live"
)

    data = request_json(
        session=session,
        url=url,
    )

    game_data = data.get(
        "gameData",
        {},
    )

    teams = game_data.get(
        "teams",
        {},
    )

    venue = game_data.get(
        "venue",
        {},
    )

    datetime_data = game_data.get(
        "datetime",
        {},
    )

    return {
        "away_team_id": (
            teams.get("away", {})
            .get("id")
        ),
        "home_team_id": (
            teams.get("home", {})
            .get("id")
        ),
        "away_team_name": (
            teams.get("away", {})
            .get("name")
        ),
        "home_team_name": (
            teams.get("home", {})
            .get("name")
        ),
        "venue": venue.get(
            "name"
        ),
        "game_datetime": datetime_data.get(
            "dateTime"
        ),
    }


def get_boxscore(
    session: requests.Session,
    game_id: int,
) -> dict[str, Any]:
    """Download one MLB boxscore."""
    url = (
        f"{MLB_API_BASE_URL}/game/"
        f"{game_id}/boxscore"
    )

    return request_json(
        session=session,
        url=url,
    )


def extract_player_information(
    player_data: dict[str, Any],
) -> dict[str, Any]:
    """Extract stable player identity fields."""
    person = player_data.get(
        "person",
        {},
    )

    position = player_data.get(
        "position",
        {},
    )

    if not position:
        position = player_data.get(
            "allPositions",
            [{}],
        )

        if isinstance(position, list):
            position = (
                position[0]
                if position
                else {}
            )

    return {
        "player_id": person.get(
            "id"
        ),
        "player_name": person.get(
            "fullName"
        ),
        "position": (
            position.get("abbreviation")
            if isinstance(position, dict)
            else None
        ),
    }


def get_active_roster(
    session: requests.Session,
    team_id: int,
    target_date: str,
) -> dict[int, dict[str, Any]]:
    """Return active roster players keyed by player ID."""
    url = (
        f"{MLB_API_BASE_URL}/teams/"
        f"{team_id}/roster"
    )

    try:
        data = request_json(
            session=session,
            url=url,
            params={
                "rosterType": "active",
                "date": target_date,
            },
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        data = request_json(
            session=session,
            url=url,
            params={
                "rosterType": "active",
            },
        )

    roster: dict[
        int,
        dict[str, Any],
    ] = {}

    for entry in data.get(
        "roster",
        [],
    ):
        person = entry.get(
            "person",
            {},
        )

        position = entry.get(
            "position",
            {},
        )

        player_id = person.get(
            "id"
        )

        player_name = person.get(
            "fullName"
        )

        if player_id is None:
            continue

        roster[int(player_id)] = {
            "player_id": int(
                player_id
            ),
            "player_name": player_name,
            "position": position.get(
                "abbreviation"
            ),
        }

    return roster


def get_recent_team_games(
    session: requests.Session,
    team_id: int,
    target_date: str,
) -> list[int]:
    """Find recently completed games for one team."""
    target_day = datetime.strptime(
        target_date,
        "%Y-%m-%d",
    ).date()

    start_day = (
        target_day
        - timedelta(
            days=RECENT_LOOKBACK_DAYS
        )
    )

    end_day = (
        target_day
        - timedelta(days=1)
    )

    url = f"{MLB_API_BASE_URL}/schedule"

    data = request_json(
        session=session,
        url=url,
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_day.isoformat(),
            "endDate": end_day.isoformat(),
            "hydrate": "status",
        },
    )

    completed_games: list[
        tuple[str, int]
    ] = []

    for date_entry in data.get(
        "dates",
        [],
    ):
        game_date = date_entry.get(
            "date",
            "",
        )

        for game in date_entry.get(
            "games",
            [],
        ):
            status = game.get(
                "status",
                {},
            )

            coded_state = status.get(
                "codedGameState"
            )

            abstract_state = status.get(
                "abstractGameState"
            )

            if (
                coded_state != "F"
                and abstract_state != "Final"
            ):
                continue

            game_id = game.get(
                "gamePk"
            )

            if game_id is None:
                continue

            completed_games.append(
                (
                    game_date,
                    int(game_id),
                )
            )

    completed_games.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        game_id
        for _, game_id
        in completed_games[
            :MAX_RECENT_GAMES
        ]
    ]


def determine_team_side(
    boxscore: dict[str, Any],
    team_id: int,
) -> str | None:
    """Determine whether a team was home or away."""
    teams = boxscore.get(
        "teams",
        {},
    )

    for side in [
        "away",
        "home",
    ]:
        team_data = teams.get(
            side,
            {},
        )

        current_team_id = (
            team_data.get(
                "team",
                {},
            )
            .get("id")
        )

        if current_team_id == team_id:
            return side

    return None


def build_projected_lineup(
    session: requests.Session,
    team_id: int,
    target_date: str,
) -> list[dict[str, Any]]:
    """Build an expected lineup from recent confirmed batting orders.

    Players are ranked using:
    - recent lineup appearances;
    - average batting position;
    - recency weighting;
    - active-roster status.

    This is a projected lineup, not a confirmed lineup.
    """
    active_roster = get_active_roster(
        session=session,
        team_id=team_id,
        target_date=target_date,
    )

    if not active_roster:
        return []

    recent_game_ids = get_recent_team_games(
        session=session,
        team_id=team_id,
        target_date=target_date,
    )

    lineup_appearances: defaultdict[
        int,
        float,
    ] = defaultdict(float)

    weighted_position_sum: defaultdict[
        int,
        float,
    ] = defaultdict(float)

    position_weight_sum: defaultdict[
        int,
        float,
    ] = defaultdict(float)

    known_players: dict[
        int,
        dict[str, Any],
    ] = dict(
        active_roster
    )

    for game_number, recent_game_id in enumerate(
        recent_game_ids
    ):
        try:
            boxscore = get_boxscore(
                session=session,
                game_id=recent_game_id,
            )
        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            print(
                f"Could not inspect recent game "
                f"{recent_game_id}: {exc}"
            )
            continue

        team_side = determine_team_side(
            boxscore=boxscore,
            team_id=team_id,
        )

        if team_side is None:
            continue

        team_data = (
            boxscore.get(
                "teams",
                {},
            )
            .get(
                team_side,
                {},
            )
        )

        batting_order = team_data.get(
            "battingOrder",
            [],
        )

        players = team_data.get(
            "players",
            {},
        )

        if not batting_order:
            continue

        recency_weight = max(
            1.0,
            float(
                MAX_RECENT_GAMES
                - game_number
            ),
        )

        for batting_position, raw_player_id in enumerate(
            batting_order,
            start=1,
        ):
            try:
                player_id = int(
                    raw_player_id
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if player_id not in active_roster:
                continue

            player_data = players.get(
                f"ID{player_id}",
                {},
            )

            player_info = (
                extract_player_information(
                    player_data
                )
            )

            roster_info = active_roster.get(
                player_id,
                {},
            )

            known_players[player_id] = {
                "player_id": player_id,
                "player_name": (
                    player_info.get(
                        "player_name"
                    )
                    or roster_info.get(
                        "player_name"
                    )
                ),
                "position": (
                    player_info.get(
                        "position"
                    )
                    or roster_info.get(
                        "position"
                    )
                ),
            }

            lineup_appearances[
                player_id
            ] += recency_weight

            weighted_position_sum[
                player_id
            ] += (
                batting_position
                * recency_weight
            )

            position_weight_sum[
                player_id
            ] += recency_weight

    ranked_players: list[
        dict[str, Any]
    ] = []

    for player_id, appearance_score in (
        lineup_appearances.items()
    ):
        position_weight = (
            position_weight_sum[
                player_id
            ]
        )

        if position_weight <= 0:
            continue

        average_position = (
            weighted_position_sum[
                player_id
            ]
            / position_weight
        )

        player_info = known_players.get(
            player_id,
            {},
        )

        ranked_players.append(
            {
                "player_id": player_id,
                "player_name": (
                    player_info.get(
                        "player_name"
                    )
                ),
                "position": (
                    player_info.get(
                        "position"
                    )
                ),
                "appearance_score": (
                    appearance_score
                ),
                "average_position": (
                    average_position
                ),
            }
        )

    ranked_players.sort(
        key=lambda player: (
            -player[
                "appearance_score"
            ],
            player[
                "average_position"
            ],
        )
    )

    selected_players = (
        ranked_players[:9]
    )

    selected_ids = {
        player["player_id"]
        for player in selected_players
    }

    if len(selected_players) < 9:
        fallback_players = []

        for player_id, player_info in (
            active_roster.items()
        ):
            if player_id in selected_ids:
                continue

            position = (
                player_info.get(
                    "position"
                )
                or ""
            ).upper()

            if position in NON_HITTER_POSITIONS:
                continue

            fallback_players.append(
                {
                    "player_id": player_id,
                    "player_name": (
                        player_info.get(
                            "player_name"
                        )
                    ),
                    "position": position,
                    "appearance_score": 0.0,
                    "average_position": 99.0,
                }
            )

        fallback_players.sort(
            key=lambda player: (
                player.get(
                    "player_name"
                )
                or ""
            )
        )

        needed = (
            9
            - len(selected_players)
        )

        selected_players.extend(
            fallback_players[:needed]
        )

    selected_players.sort(
        key=lambda player: (
            player[
                "average_position"
            ],
            -player[
                "appearance_score"
            ],
        )
    )

    projected_lineup = []

    for batting_position, player in enumerate(
        selected_players[:9],
        start=1,
    ):
        projected_lineup.append(
            {
                "player_id": player[
                    "player_id"
                ],
                "player_name": player[
                    "player_name"
                ],
                "position": player[
                    "position"
                ],
                "batting_order": (
                    batting_position
                ),
            }
        )

    return projected_lineup


def extract_confirmed_lineup(
    team_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract a confirmed batting order from the current boxscore."""
    batting_order = team_data.get(
        "battingOrder",
        [],
    )

    players = team_data.get(
        "players",
        {},
    )

    confirmed_lineup = []

    for batting_position, raw_player_id in enumerate(
        batting_order,
        start=1,
    ):
        try:
            player_id = int(
                raw_player_id
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        player_data = players.get(
            f"ID{player_id}",
            {},
        )

        player_info = (
            extract_player_information(
                player_data
            )
        )

        player_name = (
            player_info.get(
                "player_name"
            )
        )

        if not player_name:
            continue

        confirmed_lineup.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "position": (
                    player_info.get(
                        "position"
                    )
                ),
                "batting_order": (
                    batting_position
                ),
            }
        )

    return confirmed_lineup


def download_hitters(
    target_date: str | None = None,
) -> pd.DataFrame:
    """Download confirmed or projected hitters for the slate."""
    target_date = parse_target_date(
        target_date
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule = load_schedule(
        target_date
    )

    session = create_session()

    rows: list[
        dict[str, Any]
    ] = []

    confirmed_teams = 0
    projected_teams = 0
    failed_teams = 0

    projected_lineup_cache: dict[
        tuple[int, str],
        list[dict[str, Any]],
    ] = {}

    for game in schedule.itertuples(
        index=False
    ):
        game_id = int(
            game.game_id
        )

        try:
            boxscore = get_boxscore(
                session=session,
                game_id=game_id,
            )

            game_context = get_game_context(
                session=session,
                game_id=game_id,
            )
        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            print(
                f"Skipped game {game_id}: "
                f"{exc}"
            )
            continue

        boxscore_teams = boxscore.get(
            "teams",
            {},
        )

        home_team = (
            game_context.get(
                "home_team_name"
            )
            or getattr(
                game,
                "home_team",
                "",
            )
        )

        away_team = (
            game_context.get(
                "away_team_name"
            )
            or getattr(
                game,
                "away_team",
                "",
            )
        )

        venue = game_context.get(
            "venue"
        )

        for side in [
            "away",
            "home",
        ]:
            team_data = (
                boxscore_teams.get(
                    side,
                    {},
                )
            )

            team_id = game_context.get(
                f"{side}_team_id"
            )

            team_name = (
                away_team
                if side == "away"
                else home_team
            )

            opponent = (
                home_team
                if side == "away"
                else away_team
            )

            confirmed_lineup = (
                extract_confirmed_lineup(
                    team_data
                )
            )

            if confirmed_lineup:
                lineup = confirmed_lineup
                lineup_status = "CONFIRMED"
                lineup_source = (
                    "CURRENT_BOX_SCORE"
                )
                confirmed_teams += 1
            else:
                if team_id is None:
                    print(
                        f"Could not identify team ID "
                        f"for game {game_id}, "
                        f"side {side}."
                    )

                    failed_teams += 1
                    continue

                cache_key = (
                    int(team_id),
                    target_date,
                )

                if (
                    cache_key
                    not in projected_lineup_cache
                ):
                    try:
                        projected_lineup_cache[
                            cache_key
                        ] = build_projected_lineup(
                            session=session,
                            team_id=int(
                                team_id
                            ),
                            target_date=target_date,
                        )
                    except (
                        requests.RequestException,
                        ValueError,
                    ) as exc:
                        print(
                            f"Could not build projected "
                            f"lineup for {team_name}: "
                            f"{exc}"
                        )

                        projected_lineup_cache[
                            cache_key
                        ] = []

                lineup = (
                    projected_lineup_cache[
                        cache_key
                    ]
                )

                lineup_status = "PROJECTED"
                lineup_source = (
                    "RECENT_CONFIRMED_LINEUPS"
                )

                if lineup:
                    projected_teams += 1
                else:
                    failed_teams += 1

                    print(
                        f"No confirmed or projected "
                        f"lineup could be created for "
                        f"{team_name}."
                    )

                    continue

            for player in lineup:
                player_id = player.get(
                    "player_id"
                )

                player_name = player.get(
                    "player_name"
                )

                if (
                    player_id is None
                    or not player_name
                ):
                    continue

                rows.append(
                    {
                        "date": target_date,
                        "game_id": game_id,
                        "team": team_name,
                        "opponent": opponent,
                        "side": side,
                        "player_id": int(
                            player_id
                        ),
                        "player_name": (
                            player_name
                        ),
                        "batting_order": (
                            player.get(
                                "batting_order"
                            )
                        ),
                        "position": (
                            player.get(
                                "position"
                            )
                        ),
                        "status": getattr(
                            game,
                            "status",
                            None,
                        ),
                        "lineup_status": (
                            lineup_status
                        ),
                        "lineup_source": (
                            lineup_source
                        ),
                        "home_team": home_team,
                        "away_team": away_team,
                        "venue": venue,
                        "game_datetime": (
                            game_context.get(
                                "game_datetime"
                            )
                        ),
                    }
                )

    columns = [
        "date",
        "game_id",
        "team",
        "opponent",
        "side",
        "player_id",
        "player_name",
        "batting_order",
        "position",
        "status",
        "lineup_status",
        "lineup_source",
        "home_team",
        "away_team",
        "venue",
        "game_datetime",
    ]

    hitters = pd.DataFrame(
        rows,
        columns=columns,
    )

    if not hitters.empty:
        hitters["game_id"] = pd.to_numeric(
            hitters["game_id"],
            errors="coerce",
        )

        hitters["player_id"] = pd.to_numeric(
            hitters["player_id"],
            errors="coerce",
        )

        hitters["batting_order"] = pd.to_numeric(
            hitters["batting_order"],
            errors="coerce",
        )

        hitters = hitters.dropna(
            subset=[
                "game_id",
                "player_id",
                "player_name",
            ]
        ).copy()

        hitters["game_id"] = (
            hitters["game_id"]
            .astype("int64")
        )

        hitters["player_id"] = (
            hitters["player_id"]
            .astype("int64")
        )

        hitters = hitters.drop_duplicates(
            subset=[
                "game_id",
                "player_id",
            ],
            keep="first",
        )

        hitters = hitters.sort_values(
            [
                "game_id",
                "side",
                "batting_order",
            ]
        ).reset_index(
            drop=True
        )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    hitters.to_csv(
        output_path,
        index=False,
    )

    print(
        "=" * 72
    )

    print(
        "HITTER DOWNLOAD COMPLETE"
    )

    print(
        f"Slate date: {target_date}"
    )

    print(
        f"Confirmed team lineups: "
        f"{confirmed_teams}"
    )

    print(
        f"Projected team lineups: "
        f"{projected_teams}"
    )

    print(
        f"Teams without usable lineups: "
        f"{failed_teams}"
    )

    print(
        f"Hitters saved: "
        f"{len(hitters)}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        "=" * 72
    )

    if hitters.empty:
        print(
            "No confirmed or projected hitters "
            "could be created."
        )
    else:
        print(
            "\nRows by lineup status:"
        )

        print(
            hitters[
                "lineup_status"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

        preview_columns = [
            "game_id",
            "team",
            "opponent",
            "player_name",
            "batting_order",
            "position",
            "lineup_status",
        ]

        print(
            "\nHitter preview:"
        )

        print(
            hitters[
                preview_columns
            ]
            .head(40)
            .to_string(
                index=False
            )
        )

    return hitters


if __name__ == "__main__":
    download_hitters()
