import subprocess

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
    "sports/mlb/models/models/probability_engine.py",
    "sports/mlb/models/models/project_pitcher_fantasy.py",
    "sports/mlb/models/models/project_pitcher_outs.py",
    "sports/mlb/models/models/rank_platform_props.py",

    "sports/mlb/models/models/create_daily_report.py",
]

for script in scripts:
    print(f"\nRunning {script}...")
    subprocess.run(["python", script], check=True)

print("\nMLB daily run complete.")
print("Check outputs/universal_daily_report.csv")
