import argparse
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", required=True, choices=["book", "phone", "bottle"])
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path("raw_images") / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    saved = 0

    print("Tekan SPACE untuk simpan imej, Q untuk keluar.")

    while cap.isOpened() and saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(
            frame,
            f"{args.class_name}: {saved}/{args.count} | SPACE=simpan | Q=keluar",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Dataset Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            filename = out_dir / f"{args.class_name}_{saved:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            print("Saved:", filename)
            saved += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
