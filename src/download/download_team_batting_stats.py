import os
import requests
import pandas as pd

def download_team_batting_stats(season="2026"):
    os.makedirs("data/team_stats", exist_ok=True)

    url = "https://statsapi.mlb.com/api/v1/teams/stats"
    params = {
        "sportIds": 1,
        "stats": "season",
        "group": "hitting",
        "season": season
    }

    data = requests.get(url, params=params).json()

    rows = []

    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            team = split["team"]
            stat = split["stat"]

            rows.append({
                "season": season,
                "team_id": team["id"],
                "team": team["name"],
                "games": stat.get("gamesPlayed"),
                "runs": stat.get("runs"),
                "hits": stat.get("hits"),
                "doubles": stat.get("doubles"),
                "triples": stat.get("triples"),
                "home_runs": stat.get("homeRuns"),
                "strikeouts": stat.get("strikeOuts"),
                "walks": stat.get("baseOnBalls"),
                "avg": stat.get("avg"),
                "obp": stat.get("obp"),
                "slg": stat.get("slg"),
                "ops": stat.get("ops")
            })

    df = pd.DataFrame(rows)

    df["games"] = pd.to_numeric(df["games"], errors="coerce")
    df["strikeouts"] = pd.to_numeric(df["strikeouts"], errors="coerce")
    df["runs"] = pd.to_numeric(df["runs"], errors="coerce")
    df["hits"] = pd.to_numeric(df["hits"], errors="coerce")
    df["walks"] = pd.to_numeric(df["walks"], errors="coerce")

    df["team_k_per_game"] = df["strikeouts"] / df["games"]
    df["runs_per_game"] = df["runs"] / df["games"]
    df["hits_per_game"] = df["hits"] / df["games"]
    df["walks_per_game"] = df["walks"] / df["games"]

    df.to_csv(f"data/team_stats/team_batting_{season}.csv", index=False)

    return df

if __name__ == "__main__":
    print(download_team_batting_stats())
