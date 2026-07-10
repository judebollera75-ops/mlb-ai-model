import os
import time
import requests
import pandas as pd

TARGET_DATE = "2026-07-09"
SEASON = "2026"


def safe_divide(numerator, denominator):
    if denominator in (None, 0) or pd.isna(denominator):
        return None
    return numerator / denominator


def download_hitter_stats(target_date=TARGET_DATE, season=SEASON):
    os.makedirs("data/hitter_stats", exist_ok=True)

    hitters = pd.read_csv(f"data/hitters/{target_date}.csv")
    hitters = hitters.drop_duplicates(subset=["player_id"])

    rows = []

    for i, hitter in hitters.iterrows():
        player_id = int(hitter["player_id"])

        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        params = {
            "stats": "season",
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
            if not splits:
                continue

            stat = splits[0].get("stat", {})

            games = stat.get("gamesPlayed")
            plate_appearances = stat.get("plateAppearances")
            at_bats = stat.get("atBats")
            hits = stat.get("hits")
            doubles = stat.get("doubles")
            triples = stat.get("triples")
            home_runs = stat.get("homeRuns")
            runs = stat.get("runs")
            rbi = stat.get("rbi")
            walks = stat.get("baseOnBalls")
            strikeouts = stat.get("strikeOuts")
            stolen_bases = stat.get("stolenBases")
            caught_stealing = stat.get("caughtStealing")

            total_bases = (
                (hits or 0)
                + (doubles or 0)
                + 2 * (triples or 0)
                + 3 * (home_runs or 0)
            )

            rows.append({
                "date": target_date,
                "season": season,
                "player_id": player_id,
                "player_name": hitter["player_name"],
                "team": hitter["team"],
                "opponent": hitter["opponent"],
                "batting_order": hitter["batting_order"],
                "position": hitter["position"],

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

                "avg": pd.to_numeric(stat.get("avg"), errors="coerce"),
                "obp": pd.to_numeric(stat.get("obp"), errors="coerce"),
                "slg": pd.to_numeric(stat.get("slg"), errors="coerce"),
                "ops": pd.to_numeric(stat.get("ops"), errors="coerce"),

                "hits_per_game": safe_divide(hits, games),
                "total_bases_per_game": safe_divide(total_bases, games),
                "runs_per_game": safe_divide(runs, games),
                "rbi_per_game": safe_divide(rbi, games),
                "walks_per_game": safe_divide(walks, games),
                "strikeouts_per_game": safe_divide(strikeouts, games),
                "home_runs_per_game": safe_divide(home_runs, games),
                "stolen_bases_per_game": safe_divide(stolen_bases, games),

                "walk_rate": safe_divide(walks, plate_appearances),
                "strikeout_rate": safe_divide(strikeouts, plate_appearances),
                "home_run_rate": safe_divide(home_runs, plate_appearances),
            })

        except requests.RequestException as exc:
            print(f"Skipped {hitter['player_name']}: {exc}")

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1} hitters")

        time.sleep(0.05)

    df = pd.DataFrame(rows)

    output_path = f"data/hitter_stats/{target_date}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} hitter stat rows to {output_path}")

    if not df.empty:
        print(
            df[[
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
            ]].head(30).to_string(index=False)
        )

    return df


if __name__ == "__main__":
    download_hitter_stats()
