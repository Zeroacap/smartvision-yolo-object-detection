from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train baseline model
results = model.train(
    data="roboflow_dataset/data.yaml",
    epochs=5,
    imgsz=512,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    project="runs/smartvision",
    name="baseline"
)

print("Training selesai.")