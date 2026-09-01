from pathlib import Path
from PIL import Image

CLASSES = {0: "book", 1: "phone", 2: "bottle"}

def check_split(split):
    image_dir = Path("dataset/images") / split
    label_dir = Path("dataset/labels") / split

    images = list(image_dir.glob("*.*"))
    missing_labels = []
    invalid_labels = []

    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue

        try:
            Image.open(img_path).verify()
        except Exception as e:
            print(f"Invalid image {img_path}: {e}")

        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_labels.append((label_path.name, line_no, "format"))
                continue

            try:
                class_id = int(parts[0])
                vals = list(map(float, parts[1:]))
                if class_id not in CLASSES:
                    invalid_labels.append((label_path.name, line_no, "class_id"))
                if not all(0 <= v <= 1 for v in vals):
                    invalid_labels.append((label_path.name, line_no, "coordinate"))
            except ValueError:
                invalid_labels.append((label_path.name, line_no, "numeric"))

    print(f"\n[{split}] images = {len(images)}")
    print(f"Missing labels = {len(missing_labels)}")
    print(f"Invalid labels = {len(invalid_labels)}")

    if missing_labels:
        print("Contoh missing:", missing_labels[:10])
    if invalid_labels:
        print("Contoh invalid:", invalid_labels[:10])

def main():
    for split in ["train", "val", "test"]:
        check_split(split)

if __name__ == "__main__":
    main()
