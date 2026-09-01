import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, plots=True)

    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    out = Path(args.weights).parents[1] / f"metrics_{args.split}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Saved:", out)

if __name__ == "__main__":
    main()
