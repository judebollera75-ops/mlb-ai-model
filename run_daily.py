import subprocess

scripts = [
    "src/download/download_schedule.py",
    "src/download/download_pitchers.py",
    "src/download/download_pitcher_stats.py",
    "src/download/download_pitcher_game_logs.py",
    "src/features/build_pitcher_features.py",
    "src/features/build_master_dataset.py",
    "src/features/add_opponent_features.py",
    "src/features/add_recent_form.py",
    "src/models/project_pitcher_strikeouts.py",
    "src/models/calibrate_projections.py",
    "src/models/probability_engine.py",
    "src/models/project_pitcher_fantasy.py",
    "src/models/project_pitcher_outs.py",
    "src/models/rank_platform_props.py",
]

for script in scripts:
    print(f"\nRunning {script}...")
    subprocess.run(["python", script], check=True)

print("\n✅ Daily MLB model run complete.")
print("Check outputs/best_platform_props.csv")
