from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from pathlib import Path
from typing import List
import shutil
import uuid

app = FastAPI(
    title="SmartVision YOLO Object Detection API",
    description="Object Detection API for Book, Bottle and Phone",
    version="1.0"
)

# Path project
BASE_DIR = Path(__file__).resolve().parent.parent

# Model terbaik daripada hyperparameter tuning
MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "smartvision"
    / "tuning_adam_5epoch"
    / "weights"
    / "best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak dijumpai: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


# =========================
# PYDANTIC RESPONSE MODELS
# =========================

class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictionResponse(BaseModel):
    filename: str
    total_detections: int
    detections: List[Detection]


class HealthResponse(BaseModel):
    status: str
    model: str


# =========================
# API ENDPOINTS
# =========================

@app.get("/")
def root():
    return {
        "message": "SmartVision YOLO Object Detection API",
        "classes": ["Book", "Bottle", "Phone"]
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="online",
        model="YOLO11n - Tuned Adam"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Fail mesti dalam format JPG, JPEG atau PNG."
        )

    temp_dir = BASE_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)

    suffix = Path(file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{suffix}"
    temp_path = temp_dir / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model.predict(
            source=str(temp_path),
            conf=0.5,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    Detection(
                        class_name=model.names[class_id],
                        confidence=round(confidence, 4),
                        x1=round(x1, 2),
                        y1=round(y1, 2),
                        x2=round(x2, 2),
                        y2=round(y2, 2)
                    )
                )

        return PredictionResponse(
            filename=file.filename,
            total_detections=len(detections),
            detections=detections
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()