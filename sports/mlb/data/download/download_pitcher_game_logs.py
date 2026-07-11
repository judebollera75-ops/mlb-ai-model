from datetime import date
from pathlib import Path

import pandas as pd
import requests


PITCHERS_DIRECTORY = Path("data/pitchers")
OUTPUT_DIRECTORY = Path("data/game_logs")


def download_game_logs(target_date=None, season=None):
    if target_date is None:
        target_date = date.today().isoformat()

    if season is None:
        season = str(date.fromisoformat(target_date).year)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pitchers_path = (
        PITCHERS_DIRECTORY
        / f"{target_date}.csv"
    )

    if not pitchers_path.exists():
        raise FileNotFoundError(
            f"Missing pitcher file: {pitchers_path}"
        )

    pitchers = pd.read_csv(pitchers_path)

    required_columns = {
        "pitcher_id",
        "pitcher_name",
        "team",
    }

    missing_columns = required_columns - set(pitchers.columns)

    if missing_columns:
        raise KeyError(
            f"{pitchers_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    for _, pitcher in pitchers.iterrows():
        pitcher_id = pd.to_numeric(
            pitcher.get("pitcher_id"),
            errors="coerce",
        )

        if pd.isna(pitcher_id):
            continue

        pitcher_id = int(pitcher_id)

        url = (
            f"https://statsapi.mlb.com/api/v1/"
            f"people/{pitcher_id}/stats"
        )

        params = {
            "stats": "gameLog",
            "group": "pitching",
            "season": season,
        }

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
                f"{pitcher.get('pitcher_name')}"
            )
            continue

        splits = stats_blocks[0].get("splits", [])

        if not splits:
            print(
                f"No game-log splits returned for "
                f"{pitcher.get('pitcher_name')}"
            )
            continue

        # Sort newest games first, then keep the most recent five.
        splits = sorted(
            splits,
            key=lambda game: game.get("date", ""),
            reverse=True,
        )[:5]

        for game in splits:
            stat = game.get("stat", {})

            rows.append(
                {
                    "pitcher_id": pitcher_id,
                    "pitcher_name": pitcher.get("pitcher_name"),
                    "team": pitcher.get("team"),
                    "game_date": game.get("date"),
                    "innings": stat.get("inningsPitched"),
                    "strikeouts": stat.get("strikeOuts"),
                    "walks": stat.get("baseOnBalls"),
                    "hits": stat.get("hits"),
                    "earned_runs": stat.get("earnedRuns"),
                    "home_runs": stat.get("homeRuns"),
                    "era": stat.get("era"),
                }
            )

    logs = pd.DataFrame(
        rows,
        columns=[
            "pitcher_id",
            "pitcher_name",
            "team",
            "game_date",
            "innings",
            "strikeouts",
            "walks",
            "hits",
            "earned_runs",
            "home_runs",
            "era",
        ],
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    logs.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(logs)} pitcher game-log rows "
        f"to {output_path}"
    )

    if not logs.empty:
        print(logs.head(30).to_string(index=False))

    return logs


if __name__ == "__main__":
    download_game_logs()
