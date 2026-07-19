"""Run the production MLB pipeline for one slate."""

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

TARGET_DATE = os.getenv(
    "MLB_TARGET_DATE",
    date.today().isoformat(),
)


def validate_target_date(value: str) -> str:
    """Require YYYY-MM-DD format."""
    try:
        parsed = datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {value!r}"
        ) from exc

    return parsed.isoformat()


def run_script(
    script: str,
    required: bool = True,
) -> bool:
    """Run one repository script."""
    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        message = f"Missing script: {script}"

        if required:
            raise FileNotFoundError(message)

        print(
            f"Skipped optional script: {message}",
            flush=True,
        )
        return False

    environment = os.environ.copy()
    environment["MLB_TARGET_DATE"] = TARGET_DATE

    print(
        "\n" + "-" * 72,
        flush=True,
    )
    print(
        f"Running {script} for {TARGET_DATE}...",
        flush=True,
    )
    print(
        "-" * 72,
        flush=True,
    )

    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError:
        if required:
            raise

        print(
            f"Optional script failed: {script}",
            flush=True,
        )
        return False

    return True


def hitter_file_path() -> Path:
    """Return the current-slate hitter file."""
    return (
        PROJECT_ROOT
        / "data"
        / "hitters"
        / f"{TARGET_DATE}.csv"
    )


def confirmed_hitters_are_available() -> bool:
    """Check whether a usable lineup file exists."""
    path = hitter_file_path()

    if not path.exists():
        return False

    try:
        hitters = pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return False

    required_columns = {
        "game_id",
        "player_id",
        "player_name",
        "batting_order",
    }

    if hitters.empty:
        return False

    if not required_columns.issubset(
        hitters.columns
    ):
        return False

    valid = hitters.dropna(
        subset=[
            "game_id",
            "player_id",
            "player_name",
            "batting_order",
        ]
    )

    return not valid.empty


def remove_stale_hitter_output() -> None:
    """Remove an old hitter projection file."""
    path = (
        HITTER_OUTPUT_DIRECTORY
        / "today_hitter_projections.csv"
    )

    if path.exists():
        path.unlink()

        print(
            "Removed stale hitter projection output.",
            flush=True,
        )


def remove_stale_pitcher_outputs() -> None:
    """Remove outputs that must be rebuilt for the current slate."""
    stale_paths = [
        OUTPUT_DIRECTORY
        / "pitcher_outs_projections.csv",
        OUTPUT_DIRECTORY
        / "pitcher_fantasy_projections.csv",
        OUTPUT_DIRECTORY
        / "mlb_universal_projections.csv",
        OUTPUT_DIRECTORY
        / "probability_table.csv",
        OUTPUT_DIRECTORY
        / "mlb_daily_card.csv",
    ]

    for path in stale_paths:
        if path.exists():
            path.unlink()

            print(
                f"Removed stale output: {path}",
                flush=True,
            )


def run_pitcher_pipeline() -> None:
    """Generate all current pitcher projections."""
    scripts = [
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

        # These two must run before pitcher fantasy.
        "sports/mlb/models/models/project_pitcher_outs.py",
        "sports/mlb/models/models/project_pitcher_fantasy.py",
    ]

    for script in scripts:
        run_script(script)


def run_hitter_pipeline() -> bool:
    """Generate current hitter projections."""
    run_script(
        "sports/mlb/data/download/download_hitters.py"
    )

    if not confirmed_hitters_are_available():
        print(
            "\nConfirmed batting orders are unavailable. "
            "Skipping hitter projections.",
            flush=True,
        )

        remove_stale_hitter_output()
        return False

    scripts = [
        "sports/mlb/data/download/download_hitter_stats.py",
        "sports/mlb/data/download/download_hitter_game_logs.py",
        "sports/mlb/models/models/project_today_hitters.py",
    ]

    for script in scripts:
        run_script(script)

    return True


def run_betting_pipeline() -> None:
    """Build projections, probabilities, card, historical market analytics, and history."""
    scripts = [
        "sports/mlb/models/models/build_universal_mlb_projections.py",
        "sports/mlb/models/models/probability_engine.py",

        # NEW: Build historical market performance report
        "sports/mlb/models/models/market_performance.py",

        "sports/mlb/betting/build_daily_card.py",
        "sports/mlb/backtesting/log_daily_results.py",
    ]

    for script in scripts:
        run_script(script)

def main() -> None:
    """Run the complete daily production pipeline."""
    global TARGET_DATE

    TARGET_DATE = validate_target_date(
        TARGET_DATE
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    HITTER_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72,
        flush=True,
    )
    print(
        f"MLB production pipeline — slate date: {TARGET_DATE}",
        flush=True,
    )
    print(
        "=" * 72,
        flush=True,
    )

    # Prevent yesterday's files from entering today's pipeline.
    remove_stale_pitcher_outputs()

    run_pitcher_pipeline()
    hitters_available = run_hitter_pipeline()
    run_betting_pipeline()

    print(
        "\n" + "=" * 72,
        flush=True,
    )
    print(
        "MLB DAILY PIPELINE COMPLETED",
        flush=True,
    )
    print(
        "=" * 72,
        flush=True,
    )
    print(
        "Pitcher outs: outputs/pitcher_outs_projections.csv",
        flush=True,
    )
    print(
        "Pitcher fantasy: outputs/pitcher_fantasy_projections.csv",
        flush=True,
    )
    print(
        "Universal projections: outputs/mlb_universal_projections.csv",
        flush=True,
    )
    print(
        "Probability table: outputs/probability_table.csv",
        flush=True,
    )
    print(
        "Daily card: outputs/mlb_daily_card.csv",
        flush=True,
    )

    if not hitters_available:
        print(
            "\nHitter markets were skipped because confirmed "
            "lineups were unavailable.",
            flush=True,
        )


if __name__ == "__main__":
    main()
