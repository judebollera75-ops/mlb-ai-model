import os
import subprocess
import pandas as pd


HITTERS_PATH = "data/hitters/2026-07-10.csv"
HITTER_PROJECTIONS_PATH = "outputs/hitters/today_hitter_projections.csv"


def run_script(script):
    print(f"\nRunning {script}...")
    subprocess.run(["python", script], check=True)


# Always run these
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


# Download today's confirmed hitters
run_script("sports/mlb/data/download/download_hitters.py")


# Check whether official batting orders exist
hitters_available = False

if os.path.exists(HITTERS_PATH):
    try:
        hitters = pd.read_csv(HITTERS_PATH)
        hitters_available = not hitters.empty
    except pd.errors.EmptyDataError:
        hitters_available = False


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
    print("\nNo confirmed batting orders yet.")
    print("Skipping hitter stats and hitter projections.")

    # Remove stale hitter projections from a previous slate
    if os.path.exists(HITTER_PROJECTIONS_PATH):
        os.remove(HITTER_PROJECTIONS_PATH)
        print("Removed stale hitter projection file.")


# Build the combined projection file from whatever is available
run_script(
    "sports/mlb/models/models/build_universal_mlb_projections.py"
)

# Compare projections against your entered platform lines
run_script("sports/mlb/betting/build_daily_card.py")

# Save today's card to the historical results log
run_script("sports/mlb/betting/log_daily_card.py")


print("\nFast MLB daily run complete.")
print("Check outputs/mlb_daily_card.csv")

if not hitters_available:
    print(
        "Only pitcher markets are current. "
        "Rerun closer to first pitch for hitter projections."
    )
