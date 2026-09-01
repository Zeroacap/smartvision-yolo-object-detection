# SmartVision YOLO Object Detection Project

Projek ini ialah contoh lengkap aliran kerja Object Detection menggunakan YOLO:
dataset -> annotation -> preprocessing -> training -> evaluation -> hyperparameter tuning -> comparison -> FastAPI deployment.

## Kelas objek
- book
- phone
- bottle

## Struktur dataset
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

Setiap imej mesti mempunyai fail label `.txt` dengan nama yang sama.
Format YOLO:
`class_id x_center y_center width height`
Semua koordinat dinormalisasi antara 0 dan 1.

Class ID:
- 0 = book
- 1 = phone
- 2 = bottle

## 1. Cipta virtual environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

## 2. Install
```bash
pip install -r requirements.txt
```

## 3. Kumpul imej
Guna:
```bash
python src/capture_images.py --class_name book --count 150
python src/capture_images.py --class_name phone --count 150
python src/capture_images.py --class_name bottle --count 150
```

Atau ambil imej sendiri menggunakan telefon/kamera dan masukkan ke folder `raw_images/`.

## 4. Annotation
Gunakan CVAT, LabelImg atau Roboflow dan eksport dalam format YOLO.
Letakkan imej dan label ke struktur dataset yang dinyatakan di atas.

## 5. Semak dataset
```bash
python src/check_dataset.py
```

## 6. Train baseline
```bash
python src/train.py --name baseline --epochs 50 --imgsz 640 --batch 16 --lr0 0.01
```

## 7. Evaluation
```bash
python src/evaluate.py --weights runs/detect/baseline/weights/best.pt
```

## 8. Hyperparameter tuning
```bash
python src/tune.py
```

## 9. Bandingkan eksperimen
```bash
python src/compare_results.py
```

## 10. FastAPI deployment
```bash
uvicorn api.main:app --reload
```

Buka:
- Swagger UI: http://127.0.0.1:8000/docs

Endpoint:
- GET `/health`
- POST `/predict`

## 11. Uji API
```bash
python src/test_api.py --image path/to/test_image.jpg
```
