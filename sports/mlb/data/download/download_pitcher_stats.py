import os
import requests
import pandas as pd


def baseball_ip_to_decimal(ip_value):
    """
    Converts MLB innings format:
    5.0 -> 5.0
    5.1 -> 5.333...
    5.2 -> 5.666...
    """
    if ip_value is None or pd.isna(ip_value):
        return None

    text = str(ip_value)

    if "." not in text:
        return float(text)

    whole, partial = text.split(".", 1)

    whole = int(whole)
    partial = int(partial[0]) if partial else 0

    if partial == 1:
        return whole + (1 / 3)

    if partial == 2:
        return whole + (2 / 3)

    return float(whole)


def safe_divide(numerator, denominator):
    if denominator in [None, 0] or pd.isna(denominator):
        return None

    return numerator / denominator


def download_pitcher_stats(target_date="2026-07-09", season="2026"):
    os.makedirs("data/pitcher_stats", exist_ok=True)

    pitchers = pd.read_csv(f"data/pitchers/{target_date}.csv")
    rows = []

    for _, pitcher in pitchers.iterrows():
        pitcher_id = int(pitcher["pitcher_id"])

        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        params = {
            "stats": "season",
            "group": "pitching",
            "season": season,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        splits = data.get("stats", [{}])[0].get("splits", [])

        if not splits:
            continue

        stat = splits[0]["stat"]

        games_started = stat.get("gamesStarted")
        innings_pitched = stat.get("inningsPitched")
        strikeouts = stat.get("strikeOuts")
        walks = stat.get("baseOnBalls")
        hits_allowed = stat.get("hits")
        earned_runs = stat.get("earnedRuns")
        batters_faced = stat.get("battersFaced")
        wins = stat.get("wins")

        innings_decimal = baseball_ip_to_decimal(innings_pitched)

        season_k_per_start = safe_divide(strikeouts, games_started)
        avg_ip_per_start = safe_divide(innings_decimal, games_started)
        walks_per_start = safe_divide(walks, games_started)
        hits_per_start = safe_divide(hits_allowed, games_started)
        earned_runs_per_start = safe_divide(earned_runs, games_started)
        win_rate = safe_divide(wins, games_started)

        k_per_9 = (
            safe_divide(strikeouts * 9, innings_decimal)
            if innings_decimal
            else None
        )

        bb_per_9 = (
            safe_divide(walks * 9, innings_decimal)
            if innings_decimal
            else None
        )

        hits_per_9 = (
            safe_divide(hits_allowed * 9, innings_decimal)
            if innings_decimal
            else None
        )

        k_minus_bb_rate = (
            safe_divide(strikeouts - walks, batters_faced)
            if batters_faced
            else None
        )

        rows.append({
            "date": target_date,
            "season": season,
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher["pitcher_name"],
            "team": pitcher["team"],

            "games": stat.get("gamesPlayed"),
            "games_started": games_started,
            "innings_pitched": innings_pitched,
            "innings_decimal": innings_decimal,

            "era": pd.to_numeric(stat.get("era"), errors="coerce"),
            "whip": pd.to_numeric(stat.get("whip"), errors="coerce"),

            "strikeouts": strikeouts,
            "walks": walks,
            "hits_allowed": hits_allowed,
            "earned_runs": earned_runs,
            "home_runs_allowed": stat.get("homeRuns"),
            "batters_faced": batters_faced,
            "wins": wins,
            "losses": stat.get("losses"),

            "season_k_per_start": season_k_per_start,
            "avg_ip_per_start": avg_ip_per_start,
            "walks_per_start": walks_per_start,
            "hits_per_start": hits_per_start,
            "earned_runs_per_start": earned_runs_per_start,
            "win_rate": win_rate,

            "k_per_9": k_per_9,
            "bb_per_9": bb_per_9,
            "hits_per_9": hits_per_9,
            "k_minus_bb_rate": k_minus_bb_rate,
        })

    df = pd.DataFrame(rows)

    output_path = f"data/pitcher_stats/{target_date}.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} pitcher rows to {output_path}")

    return df


if __name__ == "__main__":
    print(download_pitcher_stats())
