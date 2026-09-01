from ultralytics import YOLO

# Load model terbaik hasil training 20 epochs
model = YOLO(
    "runs/detect/runs/smartvision/baseline-3/weights/best.pt"
)

# Real-time webcam detection
model.predict(
    source=0,
    show=True,
    conf=0.5
)