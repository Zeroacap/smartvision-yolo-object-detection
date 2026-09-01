from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "roboflow_dataset"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"

TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

# Ambil semua gambar dari train + valid
all_images = list(TRAIN_IMAGES.glob("*")) + list(VALID_IMAGES.glob("*"))

# Pastikan hanya fail gambar
all_images = [
    p for p in all_images
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

random.shuffle(all_images)

split_index = int(len(all_images) * TRAIN_RATIO)

new_train = all_images[:split_index]
new_valid = all_images[split_index:]

print(f"Total images: {len(all_images)}")
print(f"Train: {len(new_train)}")
print(f"Valid: {len(new_valid)}")

# Folder sementara
temp_dir = DATASET_DIR / "temp_split"

temp_train_images = temp_dir / "train" / "images"
temp_train_labels = temp_dir / "train" / "labels"

temp_valid_images = temp_dir / "valid" / "images"
temp_valid_labels = temp_dir / "valid" / "labels"

for folder in [
    temp_train_images,
    temp_train_labels,
    temp_valid_images,
    temp_valid_labels
]:
    folder.mkdir(parents=True, exist_ok=True)


def copy_pair(image_path, image_dest, label_dest):
    label_name = image_path.stem + ".txt"

    # Cari label sama ada dari train atau valid asal
    possible_labels = [
        TRAIN_LABELS / label_name,
        VALID_LABELS / label_name
    ]

    label_path = None

    for p in possible_labels:
        if p.exists():
            label_path = p
            break

    shutil.copy2(image_path, image_dest / image_path.name)

    if label_path:
        shutil.copy2(label_path, label_dest / label_name)
    else:
        print(f"WARNING: Label tak jumpa untuk {image_path.name}")


for img in new_train:
    copy_pair(img, temp_train_images, temp_train_labels)

for img in new_valid:
    copy_pair(img, temp_valid_images, temp_valid_labels)


# Buang train/valid lama
shutil.rmtree(DATASET_DIR / "train")
shutil.rmtree(DATASET_DIR / "valid")

# Pindahkan split baru
shutil.move(str(temp_dir / "train"), str(DATASET_DIR / "train"))
shutil.move(str(temp_dir / "valid"), str(DATASET_DIR / "valid"))

# Buang folder sementara
shutil.rmtree(temp_dir)

print("\nDataset split selesai.")
print("Ratio: 80% Train / 20% Valid")