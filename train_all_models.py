import subprocess

scripts = [
    "sports/mlb/features/features/build_hitter_training_dataset.py",
    "sports/mlb/models/models/train_hitter_models.py",
    "sports/mlb/models/models/train_leakage_free_strikeouts.py",
]

for script in scripts:
    print(f"\nTraining with {script}...")
    subprocess.run(["python", script], check=True)

print("\nAll MLB models trained successfully.")
