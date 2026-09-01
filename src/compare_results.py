from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUNS = ["baseline", "tune_lr", "tune_adamw"]

def load_last_row(run_name):
    path = Path("runs/detect") / run_name / "results.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    row = df.iloc[-1]

    candidates = {
        "precision": ["metrics/precision(B)", "metrics/precision"],
        "recall": ["metrics/recall(B)", "metrics/recall"],
        "mAP50": ["metrics/mAP50(B)", "metrics/mAP50"],
        "mAP50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
    }

    out = {"run": run_name}
    for key, names in candidates.items():
        out[key] = None
        for n in names:
            if n in row:
                out[key] = float(row[n])
                break
    return out

def main():
    rows = [load_last_row(r) for r in RUNS]
    rows = [r for r in rows if r is not None]

    if not rows:
        raise SystemExit("Tiada results.csv ditemui. Jalankan training dahulu.")

    df = pd.DataFrame(rows)
    df.to_csv("experiment_comparison.csv", index=False)
    print(df)

    plot_cols = [c for c in ["precision", "recall", "mAP50", "mAP50_95"] if df[c].notna().any()]
    ax = df.set_index("run")[plot_cols].plot(kind="bar", figsize=(10, 6))
    ax.set_title("Perbandingan Prestasi Eksperimen YOLO")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("experiment_comparison.png", dpi=200)
    print("Saved experiment_comparison.csv and experiment_comparison.png")

if __name__ == "__main__":
    main()
