import subprocess

scripts = [
    "sports/mlb/data/download/download_schedule.py",
    "sports/mlb/data/download/download_pitchers.py",
    "sports/mlb/data/download/download_pitcher_stats.py",
    "sports/mlb/data/download/download_pitcher_game_logs.py",
    "sports/mlb/data/download/download_hitters.py",
    "sports/mlb/data/download/download_hitter_stats.py",
    "sports/mlb/data/download/download_hitter_game_logs.py",

    "sports/mlb/features/features/build_pitcher_features.py",
    "sports/mlb/features/features/build_master_dataset.py",
    "sports/mlb/features/features/add_opponent_features.py",
    "sports/mlb/features/features/add_recent_form.py",
    "sports/mlb/features/features/build_hitter_training_dataset.py",

    "sports/mlb/models/models/project_pitcher_strikeouts.py",
    "sports/mlb/models/models/calibrate_projections.py",
    "sports/mlb/models/models/probability_engine.py",
    "sports/mlb/models/models/project_pitcher_fantasy.py",
    "sports/mlb/models/models/project_pitcher_outs.py",
    "sports/mlb/models/models/train_hitter_models.py",
    "sports/mlb/models/models/project_today_hitters.py",
    "sports/mlb/models/models/build_universal_mlb_projections.py",

    "sports/mlb/betting/build_daily_card.py",
]

for script in scripts:
    print(f"\nRunning {script}...")
    subprocess.run(["python", script], check=True)

print("\nMLB test pipeline complete.")
print("Check outputs/mlb_daily_card.csv")
