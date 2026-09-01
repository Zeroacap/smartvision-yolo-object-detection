# Rekod Penggunaan AI Code Assistant

## Penggunaan 1 — Penjanaan fungsi semakan dataset
**Tujuan:** Membantu menghasilkan fungsi untuk mengesan label hilang dan format anotasi YOLO yang salah.

**Prompt contoh:**
> Bantu saya tulis fungsi Python untuk semak setiap imej dalam dataset YOLO mempunyai fail label .txt dan semua koordinat berada antara 0 hingga 1.

**Hasil:**
Kod `src/check_dataset.py` digunakan dan kemudian disemak semula secara manual.

## Penggunaan 2 — Debugging FastAPI
**Tujuan:** Membantu menyelesaikan isu pembacaan UploadFile dan menukar imej kepada PIL sebelum inference.

**Prompt contoh:**
> Kenapa endpoint FastAPI saya gagal membaca imej UploadFile untuk model Ultralytics YOLO? Betulkan fungsi supaya imej boleh ditukar kepada PIL RGB dan dihantar ke model.predict().

**Hasil:**
Endpoint `/predict` dibina menggunakan `UploadFile`, `BytesIO`, PIL dan Pydantic response model.
