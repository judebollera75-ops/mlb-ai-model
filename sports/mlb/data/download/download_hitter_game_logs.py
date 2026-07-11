from datetime import date
from pathlib import Path
import time

import pandas as pd
import requests


HITTERS_DIRECTORY = Path("data/hitters")
OUTPUT_DIRECTORY = Path("data/hitter_game_logs")


def download_hitter_game_logs(target_date=None, season=None):
    if target_date is None:
        target_date = date.today().isoformat()

    if season is None:
        season = str(date.fromisoformat(target_date).year)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    hitters_path = (
        HITTERS_DIRECTORY
        / f"{target_date}.csv"
    )

    if not hitters_path.exists():
        raise FileNotFoundError(
            f"Missing hitters file: {hitters_path}"
        )

    hitters = pd.read_csv(hitters_path)

    if hitters.empty:
        print(
            f"No hitters are available for {target_date}. "
            "Skipping hitter game logs."
        )
        return pd.DataFrame()

    required_columns = {
        "player_id",
        "player_name",
        "team",
    }

    missing_columns = required_columns - set(hitters.columns)

    if missing_columns:
        raise KeyError(
            f"{hitters_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    hitters["player_id"] = pd.to_numeric(
        hitters["player_id"],
        errors="coerce",
    )

    hitters = hitters.dropna(
        subset=[
            "player_id",
            "player_name",
        ]
    ).copy()

    hitters["player_id"] = hitters["player_id"].astype(int)

    hitters = hitters.drop_duplicates(
        subset=["player_id"],
        keep="first",
    )

    rows = []

    for count, (_, hitter) in enumerate(
        hitters.iterrows(),
        start=1,
    ):
        player_id = int(hitter["player_id"])

        url = (
            f"https://statsapi.mlb.com/api/v1/"
            f"people/{player_id}/stats"
        )

        params = {
            "stats": "gameLog",
            "group": "hitting",
            "season": season,
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=30,
            )

            response.raise_for_status()
            data = response.json()

            stats_blocks = data.get("stats", [])

            if not stats_blocks:
                print(
                    f"No game logs returned for "
                    f"{hitter['player_name']}"
                )
                continue

            splits = stats_blocks[0].get("splits", [])

            if not splits:
                print(
                    f"No game-log splits returned for "
                    f"{hitter['player_name']}"
                )
                continue

            for split in splits:
                stat = split.get("stat", {})
                game = split.get("game", {})
                opponent = split.get("opponent", {})
                game_date = split.get("date")

                hits = pd.to_numeric(
                    stat.get("hits", 0),
                    errors="coerce",
                )

                doubles = pd.to_numeric(
                    stat.get("doubles", 0),
                    errors="coerce",
                )

                triples = pd.to_numeric(
                    stat.get("triples", 0),
                    errors="coerce",
                )

                home_runs = pd.to_numeric(
                    stat.get("homeRuns", 0),
                    errors="coerce",
                )

                hits = 0 if pd.isna(hits) else hits
                doubles = 0 if pd.isna(doubles) else doubles
                triples = 0 if pd.isna(triples) else triples
                home_runs = 0 if pd.isna(home_runs) else home_runs

                total_bases = (
                    hits
                    + doubles
                    + (2 * triples)
                    + (3 * home_runs)
                )

                rows.append(
                    {
                        "date": game_date,
                        "game_id": game.get("gamePk"),
                        "player_id": player_id,
                        "player_name": hitter.get("player_name"),
                        "team": hitter.get("team"),
                        "opponent": opponent.get("name"),
                        "plate_appearances": stat.get(
                            "plateAppearances"
                        ),
                        "at_bats": stat.get("atBats"),
                        "hits": hits,
                        "doubles": doubles,
                        "triples": triples,
                        "home_runs": home_runs,
                        "total_bases": total_bases,
                        "runs": stat.get("runs"),
                        "rbi": stat.get("rbi"),
                        "walks": stat.get("baseOnBalls"),
                        "strikeouts": stat.get("strikeOuts"),
                        "stolen_bases": stat.get("stolenBases"),
                        "caught_stealing": stat.get(
                            "caughtStealing"
                        ),
                        "hit_by_pitch": stat.get("hitByPitch"),
                    }
                )

        except requests.RequestException as exc:
            print(
                f"Skipped {hitter['player_name']}: {exc}"
            )

        if count % 25 == 0:
            print(
                f"Processed {count} hitters"
            )

        time.sleep(0.05)

    new_logs = pd.DataFrame(
        rows,
        columns=[
            "date",
            "game_id",
            "player_id",
            "player_name",
            "team",
            "opponent",
            "plate_appearances",
            "at_bats",
            "hits",
            "doubles",
            "triples",
            "home_runs",
            "total_bases",
            "runs",
            "rbi",
            "walks",
            "strikeouts",
            "stolen_bases",
            "caught_stealing",
            "hit_by_pitch",
        ],
    )

    if not new_logs.empty:
        new_logs["date"] = pd.to_datetime(
            new_logs["date"],
            errors="coerce",
        )

        new_logs["game_id"] = pd.to_numeric(
            new_logs["game_id"],
            errors="coerce",
        )

        new_logs["player_id"] = pd.to_numeric(
            new_logs["player_id"],
            errors="coerce",
        )

        new_logs = new_logs.dropna(
            subset=[
                "date",
                "player_id",
            ]
        ).copy()

        new_logs["player_id"] = (
            new_logs["player_id"].astype(int)
        )

        cutoff = pd.to_datetime(target_date)

        # Prevent the current prediction date or future games
        # from entering the training data.
        new_logs = new_logs[
            new_logs["date"] < cutoff
        ].copy()

    output_path = (
        OUTPUT_DIRECTORY
        / f"{season}.csv"
    )

    # Preserve previously downloaded players instead of replacing
    # the entire season file with only today's lineup.
    if output_path.exists():
        try:
            existing_logs = pd.read_csv(output_path)

            if not existing_logs.empty:
                existing_logs["date"] = pd.to_datetime(
                    existing_logs["date"],
                    errors="coerce",
                )

                combined_logs = pd.concat(
                    [
                        existing_logs,
                        new_logs,
                    ],
                    ignore_index=True,
                )
            else:
                combined_logs = new_logs.copy()

        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            combined_logs = new_logs.copy()
    else:
        combined_logs = new_logs.copy()

    if not combined_logs.empty:
        combined_logs["date"] = pd.to_datetime(
            combined_logs["date"],
            errors="coerce",
        )

        combined_logs["player_id"] = pd.to_numeric(
            combined_logs["player_id"],
            errors="coerce",
        )

        combined_logs["game_id"] = pd.to_numeric(
            combined_logs["game_id"],
            errors="coerce",
        )

        combined_logs = combined_logs.dropna(
            subset=[
                "date",
                "player_id",
            ]
        ).copy()

        combined_logs["player_id"] = (
            combined_logs["player_id"].astype(int)
        )

        duplicate_columns = [
            "player_id",
            "game_id",
            "date",
        ]

        combined_logs = combined_logs.drop_duplicates(
            subset=duplicate_columns,
            keep="last",
        )

        combined_logs = combined_logs.sort_values(
            [
                "player_id",
                "date",
            ],
            ascending=[
                True,
                True,
            ],
        ).reset_index(drop=True)

        combined_logs["date"] = (
            combined_logs["date"]
            .dt.strftime("%Y-%m-%d")
        )

    combined_logs.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nDownloaded {len(new_logs)} current hitter "
        f"game-log rows."
    )

    print(
        f"Saved {len(combined_logs)} total hitter "
        f"game-log rows to {output_path}"
    )

    if not combined_logs.empty:
        display_columns = [
            "date",
            "player_name",
            "opponent",
            "at_bats",
            "hits",
            "total_bases",
            "home_runs",
            "runs",
            "rbi",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in combined_logs.columns
        ]

        print(
            combined_logs[display_columns]
            .tail(30)
            .to_string(index=False)
        )

    return combined_logs


if __name__ == "__main__":
    download_hitter_game_logs()
