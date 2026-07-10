import os
import requests
import pandas as pd

SLATE_DATE = "2026-07-09"


def download_actual_results(slate_date=SLATE_DATE):
    os.makedirs("data/results", exist_ok=True)

    schedule = pd.read_csv(f"data/schedules/{slate_date}.csv")

    hitter_rows = []
    pitcher_rows = []

    for game_id in schedule["game_id"].dropna().astype(int).unique():
        url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        for side in ["away", "home"]:
            team_data = data.get("teams", {}).get(side, {})
            players = team_data.get("players", {})

            for player_data in players.values():
                person = player_data.get("person", {})
                stats = player_data.get("stats", {})

                player_id = person.get("id")
                player_name = person.get("fullName")

                batting = stats.get("batting", {})
                pitching = stats.get("pitching", {})

                if batting:
                    hits = batting.get("hits", 0) or 0
                    doubles = batting.get("doubles", 0) or 0
                    triples = batting.get("triples", 0) or 0
                    home_runs = batting.get("homeRuns", 0) or 0

                    total_bases = (
                        hits
                        + doubles
                        + (2 * triples)
                        + (3 * home_runs)
                    )

                    hitter_rows.append({
                        "date": slate_date,
                        "game_id": game_id,
                        "player_id": player_id,
                        "player_name": player_name,
                        "hits": hits,
                        "total_bases": total_bases,
                        "home_runs": home_runs,
                        "runs": batting.get("runs"),
                        "rbi": batting.get("rbi"),
                    })

                if pitching:
                    innings = pitching.get("inningsPitched")

                    pitcher_rows.append({
                        "date": slate_date,
                        "game_id": game_id,
                        "player_id": player_id,
                        "pitcher_name": player_name,
                        "strikeouts": pitching.get("strikeOuts"),
                        "innings_pitched": innings,
                    })

    hitters = pd.DataFrame(hitter_rows).drop_duplicates(
        subset=["game_id", "player_id"]
    )

    pitchers = pd.DataFrame(pitcher_rows).drop_duplicates(
        subset=["game_id", "player_id"]
    )

    hitter_path = f"data/results/{slate_date}_hitters.csv"
    pitcher_path = f"data/results/{slate_date}_pitchers.csv"

    hitters.to_csv(hitter_path, index=False)
    pitchers.to_csv(pitcher_path, index=False)

    print(f"Saved {len(hitters)} hitter results to {hitter_path}")
    print(f"Saved {len(pitchers)} pitcher results to {pitcher_path}")

    print("\nRequested hitter results:")
    print(
        hitters[
            hitters["player_name"].isin([
                "Matt Olson",
                "Michael Harris II",
            ])
        ].to_string(index=False)
    )

    print("\nRequested pitcher results:")
    print(
        pitchers[
            pitchers["pitcher_name"].isin([
                "Gavin Williams",
                "Bryce Miller",
            ])
        ].to_string(index=False)
    )


if __name__ == "__main__":
    download_actual_results()
