"""Run the production MLB pipeline for one slate.

The slate defaults to today's date. To run another date, set:

    MLB_TARGET_DATE=2026-07-11 python daily_run.py

This script intentionally does not retrain models. Daily inference and model
training should remain separate so a live slate cannot accidentally overwrite
production models.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
HITTER_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "hitters"

TARGET_DATE = os.getenv("MLB_TARGET_DATE", date.today().isoformat())
HITTERS_PATH = PROJECT_ROOT / "data" / "hitters" / f"{TARGET_DATE}.csv"
HITTER_PROJECTIONS_PATH = (
    HITTER_OUTPUT_DIRECTORY / "today_hitter_projections.csv"
)


def validate_target_date(value: str) -> str:
    """Require an ISO date so malformed paths cannot be created."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {value!r}"
        ) from exc

    return parsed.isoformat()


def run_script(script: str, required: bool = True) -> bool:
    """Run one repository script with the current Python interpreter."""
    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        message = f"Missing script: {script}"

        if required:
            raise FileNotFoundError(message)

        print(f"\nSkipped optional script. {message}")
        return False

    environment = os.environ.copy()
    environment["MLB_TARGET_DATE"] = TARGET_DATE

    print(f"\nRunning {script} for {TARGET_DATE}...")

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError:
        if required:
            raise

        print(f"Optional script failed: {script}")
        return False

    return True


def confirmed_hitters_are_available() -> tuple[bool, pd.DataFrame]:
    """Return whether a usable confirmed-lineup file exists."""
    if not HITTERS_PATH.exists():
        return False, pd.DataFrame()

    try:
        hitters = pd.read_csv(HITTERS_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Could not read {HITTERS_PATH}: {exc}")
        return False, pd.DataFrame()

    required_columns = {
        "game_id",
        "player_id",
        "player_name",
        "batting_order",
    }

    if hitters.empty:
        return False, hitters

    missing_columns = required_columns - set(hitters.columns)

    if missing_columns:
        print(
            f"{HITTERS_PATH} is missing required columns: "
            f"{sorted(missing_columns)}"
        )
        return False, hitters

    valid_hitters = hitters.dropna(
        subset=[
            "game_id",
            "player_id",
            "player_name",
            "batting_order",
        ]
    )

    return not valid_hitters.empty, valid_hitters


def remove_stale_hitter_projections() -> None:
    """Prevent yesterday's hitter projections from appearing as current."""
    if HITTER_PROJECTIONS_PATH.exists():
        HITTER_PROJECTIONS_PATH.unlink()
        print(
            "Removed stale hitter projections because confirmed "
            "batting orders are not available."
        )


def main() -> None:
    global TARGET_DATE, HITTERS_PATH

    TARGET_DATE = validate_target_date(TARGET_DATE)
    HITTERS_PATH = (
        PROJECT_ROOT / "data" / "hitters" / f"{TARGET_DATE}.csv"
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    HITTER_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MLB production pipeline — slate date: {TARGET_DATE}")
    print("=" * 70)

    # Download the current schedule and pitcher data first.
    pitcher_and_schedule_scripts = [
        "sports/mlb/data/download/download_schedule.py",
        "sports/mlb/data/download/download_pitchers.py",
        "sports/mlb/data/download/download_pitcher_stats.py",
        "sports/mlb/data/download/download_pitcher_game_logs.py",
        "sports/mlb/features/features/build_pitcher_features.py",
        "sports/mlb/features/features/build_master_dataset.py",
        "sports/mlb/features/features/add_opponent_features.py",
        "sports/mlb/features/features/add_recent_form.py",
        "sports/mlb/models/models/project_pitcher_strikeouts.py",
        "sports/mlb/models/models/calibrate_projections.py",
        "sports/mlb/models/models/probability_engine.py",
        "sports/mlb/models/models/project_pitcher_fantasy.py",
        "sports/mlb/models/models/project_pitcher_outs.py",
    ]

    for script in pitcher_and_schedule_scripts:
        run_script(script)

    # Download confirmed batting orders. An empty file is normal early in the day.
    run_script("sports/mlb/data/download/download_hitters.py")

    hitters_available, hitters = confirmed_hitters_are_available()

    if hitters_available:
        print(f"\nConfirmed lineups found: {len(hitters)} hitters")

        hitter_scripts = [
            "sports/mlb/data/download/download_hitter_stats.py",
            "sports/mlb/data/download/download_hitter_game_logs.py",
            "sports/mlb/models/models/project_today_hitters.py",
        ]

        for script in hitter_scripts:
            run_script(script)
    else:
        print("\nNo confirmed batting orders are available yet.")
        print("Hitter projections will be skipped for this run.")
        remove_stale_hitter_projections()

    # Combine whichever current projections are available.
    run_script(
        "sports/mlb/models/models/build_universal_mlb_projections.py"
    )

    # Match projections only against currently supplied platform lines.
    run_script("sports/mlb/betting/build_daily_card.py")

    # Preserve the generated recommendations for later grading.
    run_script("sports/mlb/betting/log_daily_card.py")

    print("\nMLB daily run completed successfully.")
    print(f"Slate date: {TARGET_DATE}")
    print("Final card: outputs/mlb_daily_card.csv")

    if not hitters_available:
        print(
            "This run contains current pitcher markets only. "
            "Run it again after official batting orders are released."
        )


if __name__ == "__main__":
    main()
