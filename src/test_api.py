import argparse
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        response = requests.post(
            args.url,
            params={"conf": args.conf},
            files={"file": f},
            timeout=60,
        )

    print("Status:", response.status_code)
    print(response.json())

if __name__ == "__main__":
    main()
