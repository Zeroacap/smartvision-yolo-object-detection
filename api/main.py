from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/tune_adamw/weights/best.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("runs/detect/baseline/weights/best.pt")

app = FastAPI(
    title="SmartVision YOLO API",
    version="1.0.0",
    description="Real-time object detection API for book, phone and bottle.",
)

model = None

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    x1: float
    y1: float
    x2: float
    y2: float

class PredictionResponse(BaseModel):
    filename: str
    detections: List[Detection]
    count: int

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum ditemui. Train model dahulu.",
        )

    if not (0.0 < conf <= 1.0):
        raise HTTPException(status_code=400, detail="conf mesti antara 0 dan 1.")

    try:
        data = await file.read()
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Fail imej tidak sah.")

    results = model.predict(image, conf=conf, verbose=False)
    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=result.names[cls_id],
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return PredictionResponse(
        filename=file.filename or "uploaded_image",
        detections=detections,
        count=len(detections),
    )
