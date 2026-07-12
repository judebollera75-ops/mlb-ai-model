"""Launch the production MLB daily pipeline.

This file is kept as a stable entry point for GitHub Actions and local runs.
The actual pipeline logic lives in daily_run.py.

Optional historical-date example:

    MLB_TARGET_DATE=2026-07-11 python run_mlb_daily.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DAILY_PIPELINE = PROJECT_ROOT / "daily_run.py"


def get_target_date() -> str:
    """Return and validate the requested MLB slate date."""
    target_date = os.getenv("MLB_TARGET_DATE", date.today().isoformat())

    try:
        parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            "MLB_TARGET_DATE must use YYYY-MM-DD format. "
            f"Received: {target_date!r}"
        ) from exc

    return parsed_date.isoformat()


def main() -> None:
    """Run the production daily pipeline with the active Python interpreter."""
    if not DAILY_PIPELINE.exists():
        raise FileNotFoundError(
            f"Production pipeline was not found: {DAILY_PIPELINE}"
        )

    target_date = get_target_date()

    environment = os.environ.copy()
    environment["MLB_TARGET_DATE"] = target_date

    print("=" * 70)
    print("Starting MLB production daily run")
    print(f"Slate date: {target_date}")
    print(f"Python: {sys.executable}")
    print("=" * 70)

    subprocess.run(
        [sys.executable, str(DAILY_PIPELINE)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    print("\nMLB production run completed.")
    print("Check: outputs/mlb_daily_card.csv")


if __name__ == "__main__":
    main()
