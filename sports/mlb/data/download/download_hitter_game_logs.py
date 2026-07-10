import os
import time
import requests
import pandas as pd

TARGET_DATE = "2026-07-09"
SEASON = "2026"


def download_hitter_game_logs(target_date=TARGET_DATE, season=SEASON):
    os.makedirs("data/hitter_game_logs", exist_ok=True)

    hitters = pd.read_csv(f"data/hitters/{target_date}.csv")
    hitters = hitters.drop_duplicates(subset=["player_id"])

    rows = []

    for count, (_, hitter) in enumerate(hitters.iterrows(), start=1):
        player_id = int(hitter["player_id"])

        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        params = {
            "stats": "gameLog",
            "group": "hitting",
            "season": season,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            stats_blocks = data.get("stats", [])
            if not stats_blocks:
                continue

            splits = stats_blocks[0].get("splits", [])

            for split in splits:
                stat = split.get("stat", {})
                game = split.get("game", {})
                opponent = split.get("opponent", {})
                game_date = split.get("date")

                hits = stat.get("hits", 0) or 0
                doubles = stat.get("doubles", 0) or 0
                triples = stat.get("triples", 0) or 0
                home_runs = stat.get("homeRuns", 0) or 0

                total_bases = (
                    hits
                    + doubles
                    + (2 * triples)
                    + (3 * home_runs)
                )

                rows.append({
                    "date": game_date,
                    "game_id": game.get("gamePk"),
                    "player_id": player_id,
                    "player_name": hitter["player_name"],
                    "team": hitter["team"],
                    "opponent": opponent.get("name"),

                    "plate_appearances": stat.get("plateAppearances"),
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
                    "caught_stealing": stat.get("caughtStealing"),
                    "hit_by_pitch": stat.get("hitByPitch"),
                })

        except requests.RequestException as exc:
            print(f"Skipped {hitter['player_name']}: {exc}")

        if count % 25 == 0:
            print(f"Processed {count} hitters")

        time.sleep(0.05)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values(["player_id", "date"])

        # Remove games on or after the prediction date.
        # This prevents future information from entering training data.
        cutoff = pd.to_datetime(target_date)
        df = df[df["date"] < cutoff]

    output_path = f"data/hitter_game_logs/{season}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} hitter game-log rows to {output_path}")

    if not df.empty:
        print(
            df[[
                "date",
                "player_name",
                "opponent",
                "at_bats",
                "hits",
                "total_bases",
                "home_runs",
                "runs",
                "rbi",
            ]].head(30).to_string(index=False)
        )

    return df


if __name__ == "__main__":
    download_hitter_game_logs()
