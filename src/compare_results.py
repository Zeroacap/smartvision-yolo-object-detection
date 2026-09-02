import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("runs/detect/runs/smartvision")
DOCS = Path("docs")

baseline_file = BASE / "baseline-4" / "results.csv"
tuning_file = BASE / "tuning_adam_5epoch" / "results.csv"

if not baseline_file.exists():
    raise FileNotFoundError(f"Baseline results tak jumpa: {baseline_file}")

if not tuning_file.exists():
    raise FileNotFoundError(f"Tuning results tak jumpa: {tuning_file}")

# Pastikan folder docs wujud
DOCS.mkdir(exist_ok=True)

# Baca results menggunakan pandas
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

display_names = [
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95"
]

baseline_last = baseline.iloc[-1]
tuning_last = tuning.iloc[-1]

# Simpan nilai metric dalam NumPy array
baseline_values = np.array(
    [baseline_last[metric] for metric in metrics],
    dtype=float
)

tuning_values = np.array(
    [tuning_last[metric] for metric in metrics],
    dtype=float
)

print("\nSMARTVISION YOLO - HYPERPARAMETER COMPARISON")
print("=" * 65)

print(f"{'Metric':<25}{'Baseline SGD':>18}{'Tuning Adam':>18}")
print("-" * 65)

for name, b, t in zip(
    display_names,
    baseline_values,
    tuning_values
):
    print(f"{name:<25}{b:>18.4f}{t:>18.4f}")

print("-" * 65)

baseline_map = baseline_values[3]
tuning_map = tuning_values[3]

if tuning_map > baseline_map:
    best_model = "TUNING ADAM"
    best_optimizer = "Adam"
    best_lr = 0.001
else:
    best_model = "BASELINE SGD"
    best_optimizer = "SGD"
    best_lr = 0.01

print(f"\nModel terbaik: {best_model}")
print(f"Optimizer: {best_optimizer}")
print(f"Learning rate: {best_lr}")

print("\nBaseline:")
print("Epochs: 5")
print("Optimizer: SGD")
print("Learning rate: 0.01")

print("\nTuning:")
print("Epochs: 5")
print("Optimizer: Adam")
print("Learning rate: 0.001")

# ==============================
# SIMPAN SUMMARY DALAM CSV
# ==============================

comparison_df = pd.DataFrame({
    "Metric": display_names,
    "Baseline_SGD": baseline_values,
    "Tuning_Adam": tuning_values
})

csv_output = DOCS / "hyperparameter_comparison.csv"
comparison_df.to_csv(csv_output, index=False)

print(f"\nComparison CSV disimpan di:")
print(csv_output)

# ==============================
# GENERATE BAR CHART
# ==============================

x = np.arange(len(display_names))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline SGD"
)

plt.bar(
    x + width / 2,
    tuning_values,
    width,
    label="Tuning Adam"
)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.title("SmartVision YOLO Hyperparameter Comparison")

plt.xticks(x, display_names)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(baseline_values):
    plt.text(
        i - width / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

for i, value in enumerate(tuning_values):
    plt.text(
        i + width / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

graph_output = DOCS / "hyperparameter_comparison.png"

plt.savefig(
    graph_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nComparison graph disimpan di:")
print(graph_output)

print("\nSelesai.")