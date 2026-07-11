from datetime import date
from pathlib import Path
import time

import pandas as pd
import requests


HITTERS_DIRECTORY = Path("data/hitters")
OUTPUT_DIRECTORY = Path("data/hitter_stats")


def safe_divide(numerator, denominator):
    if numerator is None or pd.isna(numerator):
        return None

    if denominator is None or pd.isna(denominator) or denominator == 0:
        return None

    return numerator / denominator


def download_hitter_stats(target_date=None, season=None):
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
            "Skipping hitter stats."
        )

        empty_output = pd.DataFrame()
        output_path = (
            OUTPUT_DIRECTORY
            / f"{target_date}.csv"
        )

        empty_output.to_csv(
            output_path,
            index=False,
        )

        return empty_output

    required_columns = {
        "player_id",
        "player_name",
        "team",
        "opponent",
        "batting_order",
        "position",
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

    for index, hitter in hitters.reset_index(drop=True).iterrows():
        player_id = int(hitter["player_id"])

        url = (
            f"https://statsapi.mlb.com/api/v1/"
            f"people/{player_id}/stats"
        )

        params = {
            "stats": "season",
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
                    f"No season stats returned for "
                    f"{hitter['player_name']}"
                )
                continue

            splits = stats_blocks[0].get("splits", [])

            if not splits:
                print(
                    f"No season split returned for "
                    f"{hitter['player_name']}"
                )
                continue

            stat = splits[0].get("stat", {})

            games = pd.to_numeric(
                stat.get("gamesPlayed"),
                errors="coerce",
            )

            plate_appearances = pd.to_numeric(
                stat.get("plateAppearances"),
                errors="coerce",
            )

            at_bats = pd.to_numeric(
                stat.get("atBats"),
                errors="coerce",
            )

            hits = pd.to_numeric(
                stat.get("hits"),
                errors="coerce",
            )

            doubles = pd.to_numeric(
                stat.get("doubles"),
                errors="coerce",
            )

            triples = pd.to_numeric(
                stat.get("triples"),
                errors="coerce",
            )

            home_runs = pd.to_numeric(
                stat.get("homeRuns"),
                errors="coerce",
            )

            runs = pd.to_numeric(
                stat.get("runs"),
                errors="coerce",
            )

            rbi = pd.to_numeric(
                stat.get("rbi"),
                errors="coerce",
            )

            walks = pd.to_numeric(
                stat.get("baseOnBalls"),
                errors="coerce",
            )

            strikeouts = pd.to_numeric(
                stat.get("strikeOuts"),
                errors="coerce",
            )

            stolen_bases = pd.to_numeric(
                stat.get("stolenBases"),
                errors="coerce",
            )

            caught_stealing = pd.to_numeric(
                stat.get("caughtStealing"),
                errors="coerce",
            )

            total_bases = (
                (0 if pd.isna(hits) else hits)
                + (0 if pd.isna(doubles) else doubles)
                + 2 * (0 if pd.isna(triples) else triples)
                + 3 * (0 if pd.isna(home_runs) else home_runs)
            )

            rows.append(
                {
                    "date": target_date,
                    "season": season,
                    "player_id": player_id,
                    "player_name": hitter.get("player_name"),
                    "team": hitter.get("team"),
                    "opponent": hitter.get("opponent"),
                    "batting_order": hitter.get("batting_order"),
                    "position": hitter.get("position"),
                    "games": games,
                    "plate_appearances": plate_appearances,
                    "at_bats": at_bats,
                    "hits": hits,
                    "doubles": doubles,
                    "triples": triples,
                    "home_runs": home_runs,
                    "runs": runs,
                    "rbi": rbi,
                    "walks": walks,
                    "strikeouts": strikeouts,
                    "stolen_bases": stolen_bases,
                    "caught_stealing": caught_stealing,
                    "total_bases": total_bases,
                    "avg": pd.to_numeric(
                        stat.get("avg"),
                        errors="coerce",
                    ),
                    "obp": pd.to_numeric(
                        stat.get("obp"),
                        errors="coerce",
                    ),
                    "slg": pd.to_numeric(
                        stat.get("slg"),
                        errors="coerce",
                    ),
                    "ops": pd.to_numeric(
                        stat.get("ops"),
                        errors="coerce",
                    ),
                    "hits_per_game": safe_divide(
                        hits,
                        games,
                    ),
                    "total_bases_per_game": safe_divide(
                        total_bases,
                        games,
                    ),
                    "runs_per_game": safe_divide(
                        runs,
                        games,
                    ),
                    "rbi_per_game": safe_divide(
                        rbi,
                        games,
                    ),
                    "walks_per_game": safe_divide(
                        walks,
                        games,
                    ),
                    "strikeouts_per_game": safe_divide(
                        strikeouts,
                        games,
                    ),
                    "home_runs_per_game": safe_divide(
                        home_runs,
                        games,
                    ),
                    "stolen_bases_per_game": safe_divide(
                        stolen_bases,
                        games,
                    ),
                    "walk_rate": safe_divide(
                        walks,
                        plate_appearances,
                    ),
                    "strikeout_rate": safe_divide(
                        strikeouts,
                        plate_appearances,
                    ),
                    "home_run_rate": safe_divide(
                        home_runs,
                        plate_appearances,
                    ),
                }
            )

        except requests.RequestException as exc:
            print(
                f"Skipped {hitter['player_name']}: {exc}"
            )

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1} hitters"
            )

        time.sleep(0.05)

    stats = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIRECTORY
        / f"{target_date}.csv"
    )

    stats.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved {len(stats)} hitter stat rows "
        f"to {output_path}"
    )

    if not stats.empty:
        display_columns = [
            "player_name",
            "team",
            "games",
            "avg",
            "obp",
            "slg",
            "ops",
            "hits_per_game",
            "total_bases_per_game",
            "home_runs_per_game",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in stats.columns
        ]

        print(
            stats[display_columns]
            .head(30)
            .to_string(index=False)
        )

    return stats


if __name__ == "__main__":
    download_hitter_stats()
