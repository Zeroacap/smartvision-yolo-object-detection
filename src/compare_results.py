import pandas as pd
from pathlib import Path

BASE = Path("runs/detect/runs/smartvision")

baseline_file = BASE / "baseline-4" / "results.csv"
tuning_file = BASE / "tuning_adam_5epoch" / "results.csv"

if not baseline_file.exists():
    raise FileNotFoundError(f"Baseline results tak jumpa: {baseline_file}")

if not tuning_file.exists():
    raise FileNotFoundError(f"Tuning results tak jumpa: {tuning_file}")

baseline = pd.read_csv(baseline_file)
tuning = pd.read_csv(tuning_file)

# Buang ruang pada nama column
baseline.columns = baseline.columns.str.strip()
tuning.columns = tuning.columns.str.strip()

metrics = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
]

baseline_last = baseline.iloc[-1]
tuning_last = tuning.iloc[-1]

print("\nSMARTVISION YOLO - HYPERPARAMETER COMPARISON")
print("=" * 65)

print(f"{'Metric':<25}{'Baseline SGD':>18}{'Tuning Adam':>18}")
print("-" * 65)

for metric in metrics:
    b = baseline_last[metric]
    t = tuning_last[metric]

    print(f"{metric:<25}{b:>18.4f}{t:>18.4f}")

print("-" * 65)

baseline_map = baseline_last["metrics/mAP50-95(B)"]
tuning_map = tuning_last["metrics/mAP50-95(B)"]

if tuning_map > baseline_map:
    print("\nModel terbaik: TUNING ADAM")
    print("Optimizer: Adam")
    print("Learning rate: 0.001")
else:
    print("\nModel terbaik: BASELINE SGD")
    print("Optimizer: SGD")
    print("Learning rate: 0.01")

print("\nBaseline:")
print("Epochs: 5")
print("Optimizer: SGD")
print("Learning rate: 0.01")

print("\nTuning:")
print("Epochs: 5")
print("Optimizer: Adam")
print("Learning rate: 0.001")