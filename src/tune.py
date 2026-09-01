from ultralytics import YOLO

# Load pretrained YOLO11 model
model = YOLO("yolo11n.pt")

# Hyperparameter tuning
model.train(
    data="roboflow_dataset/data.yaml",
    epochs=5,
    imgsz=512,
    batch=16,
    optimizer="Adam",
    lr0=0.001,
    project="runs/smartvision",
    name="tuning_adam_5epoch"
)

print("Tuning 5 epoch selesai.")