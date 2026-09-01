from ultralytics import YOLO

EXPERIMENTS = [
    {
        "name": "baseline",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.01,
        "optimizer": "SGD",
    },
    {
        "name": "tune_lr",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.005,
        "optimizer": "SGD",
    },
    {
        "name": "tune_adamw",
        "epochs": 70,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.001,
        "optimizer": "AdamW",
    },
]

def main():
    for exp in EXPERIMENTS:
        print("\nRunning:", exp["name"])
        model = YOLO("yolo11n.pt")
        model.train(
            data="data.yaml",
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            lr0=exp["lr0"],
            optimizer=exp["optimizer"],
            project="runs/detect",
            name=exp["name"],
            plots=True,
            save=True,
            pretrained=True,
            seed=42,
        )

if __name__ == "__main__":
    main()
