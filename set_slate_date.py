import argparse
import re
from pathlib import Path


FILES = [
    "sports/mlb/features/features/build_master_dataset.py",
    "sports/mlb/features/features/add_recent_form.py",
    "sports/mlb/features/features/build_pitcher_features.py",
    "sports/mlb/models/models/project_today_hitters.py",
    "sports/mlb/data/download/download_pitcher_game_logs.py",
    "sports/mlb/data/download/download_actual_results.py",
    "sports/mlb/data/download/download_pitchers.py",
    "sports/mlb/data/download/download_hitter_stats.py",
    "sports/mlb/data/download/download_schedule.py",
    "sports/mlb/data/download/download_hitters.py",
    "sports/mlb/data/download/download_hitter_game_logs.py",
    "sports/mlb/data/download/download_pitcher_stats.py",
    "sports/mlb/betting/grade_daily_results.py",
]


def update_file(file_path: str, new_date: str) -> None:
    path = Path(file_path)

    if not path.exists():
        print(f"Skipped missing file: {file_path}")
        return

    original = path.read_text()

    # Replace slate dates formatted as YYYY-MM-DD.
    updated = re.sub(
        r"2026-07-\d{2}",
        new_date,
        original,
    )

    if updated == original:
        print(f"No date changed: {file_path}")
        return

    path.write_text(updated)
    print(f"Updated: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "date",
        help="Slate date in YYYY-MM-DD format",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        raise ValueError("Date must use YYYY-MM-DD format.")

    for file_path in FILES:
        update_file(file_path, args.date)

    print(f"\nSlate date set to {args.date}")


if __name__ == "__main__":
    main()
