from datetime import date
from pathlib import Path

import pandas as pd
import requests


PITCHERS_DIRECTORY = Path("data/pitchers")
OUTPUT_DIRECTORY = Path("data/pitcher_stats")


def baseball_ip_to_decimal(ip_value):
    """
    Convert MLB innings notation into decimal innings.

    Examples:
        5.0 -> 5.0
        5.1 -> 5.333...
        5.2 -> 5.666...
    """
    if ip_value is None or pd.isna(ip_value):
        return None

    text = str(ip_value).strip()

    if not text:
        return None

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
    if numerator is None or pd.isna(numerator):
        return None

    if denominator is None or pd.isna(denominator) or denominator == 0:
        return None

    return numerator / denominator


def download_pitcher_stats(target_date=None, season=None):
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
            "stats": "season",
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
                f"No season stats returned for "
                f"{pitcher.get('pitcher_name')}"
            )
            continue

        splits = stats_blocks[0].get("splits", [])

        if not splits:
            print(
                f"No season split returned for "
                f"{pitcher.get('pitcher_name')}"
            )
            continue

        stat = splits[0].get("stat", {})

        games_started = pd.to_numeric(
            stat.get("gamesStarted"),
            errors="coerce",
        )

        innings_pitched = stat.get("inningsPitched")

        strikeouts = pd.to_numeric(
            stat.get("strikeOuts"),
            errors="coerce",
        )

        walks = pd.to_numeric(
            stat.get("baseOnBalls"),
            errors="coerce",
        )

        hits_allowed = pd.to_numeric(
            stat.get("hits"),
            errors="coerce",
        )

        earned_runs = pd.to_numeric(
            stat.get("earnedRuns"),
            errors="coerce",
        )

        batters_faced = pd.to_numeric(
            stat.get("battersFaced"),
            errors="coerce",
        )

        wins = pd.to_numeric(
            stat.get("wins"),
            errors="coerce",
        )

        innings_decimal = baseball_ip_to_decimal(
            innings_pitched
        )

        season_k_per_start = safe_divide(
            strikeouts,
            games_started,
        )

        avg_ip_per_start = safe_divide(
            innings_decimal,
            games_started,
        )

        walks_per_start = safe_divide(
            walks,
            games_started,
        )

        hits_per_start = safe_divide(
            hits_allowed,
            games_started,
        )

        earned_runs_per_start = safe_divide(
            earned_runs,
            games_started,
        )

        win_rate = safe_divide(
            wins,
            games_started,
        )

        k_per_9 = safe_divide(
            strikeouts * 9
            if pd.notna(strikeouts)
            else None,
            innings_decimal,
        )

        bb_per_9 = safe_divide(
            walks * 9
            if pd.notna(walks)
            else None,
            innings_decimal,
        )

        hits_per_9 = safe_divide(
            hits_allowed * 9
            if pd.notna(hits_allowed)
            else None,
            innings_decimal,
        )

        k_minus_bb_rate = safe_divide(
            strikeouts - walks
            if pd.notna(strikeouts) and pd.notna(walks)
            else None,
            batters_faced,
        )

        rows.append(
            {
                "date": target_date,
                "season": season,
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher.get("pitcher_name"),
                "team": pitcher.get("team"),
                "games": pd.to_numeric(
                    stat.get("gamesPlayed"),
                    errors="coerce",
                ),
                "games_started": games_started,
                "innings_pitched": innings_pitched,
                "innings_decimal": innings_decimal,
                "era": pd.to_numeric(
                    stat.get("era"),
                    errors="coerce",
                ),
                "whip": pd.to_numeric(
                    stat.get("whip"),
                    errors="coerce",
                ),
                "strikeouts": strikeouts,
                "walks": walks,
                "hits_allowed": hits_allowed,
                "earned_runs": earned_runs,
                "home_runs_allowed": pd.to_numeric(
                    stat.get("homeRuns"),
                    errors="coerce",
                ),
                "batters_faced": batters_faced,
                "wins": wins,
                "losses": pd.to_numeric(
                    stat.get("losses"),
                    errors="coerce",
                ),
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
            }
        )

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
        f"Saved {len(stats)} pitcher rows "
        f"to {output_path}"
    )

    if not stats.empty:
        print(stats.to_string(index=False))

    return stats


if __name__ == "__main__":
    download_pitcher_stats()
