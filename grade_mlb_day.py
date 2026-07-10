import subprocess

scripts = [
    "sports/mlb/data/download/download_actual_results.py",
    "sports/mlb/betting/grade_daily_results.py",
]

for script in scripts:
    print(f"\nRunning {script}...")
    subprocess.run(["python", script], check=True)

print("\nMLB slate grading complete.")
print("Check data/model_results_log.csv")
