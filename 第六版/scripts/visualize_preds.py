"""把预测框叠加到 RGB 图上, 用于目视判断哪些类别在误报。"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.constants import CLASS_NAMES, IMAGE_SUFFIXES

PALETTE = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0),
    (255, 0, 255), (128, 0, 255), (0, 128, 255), (128, 255, 0), (255, 128, 0), (0, 0, 128),
]


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def find_image(directory, image_id):
    for suffix in IMAGE_SUFFIXES:
        path = directory / f"{image_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(image_id)


def load_predictions(directory, image_id, threshold):
    rows = []
    path = Path(directory) / f"{image_id}.txt"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        cls, cx, cy, w, h, conf = [float(v) for v in fields]
        if conf > threshold:
            rows.append((int(cls), cx, cy, w, h, conf))
    return rows


def draw(image, rows, title, min_conf_label=0.3):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for cls, cx, cy, w, h, conf in sorted(rows, key=lambda r: -r[5]):
        x1, y1 = int((cx - w / 2) * width), int((cy - h / 2) * height)
        x2, y2 = int((cx + w / 2) * width), int((cy + h / 2) * height)
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if conf >= min_conf_label:
            text = f"{CLASS_NAMES[cls]} {conf:.2f}"
            cv2.putText(canvas, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(canvas, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"{title}  n={len(rows)}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.001, 0.33])
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1400)
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    for image_id in arguments.ids:
        image = imread(find_image(Path(arguments.images), image_id))
        panels = [draw(image, load_predictions(arguments.predictions, image_id, threshold), f">{threshold}")
                  for threshold in arguments.thresholds]
        strip = np.hstack(panels)
        scale = arguments.width / strip.shape[1]
        strip = cv2.resize(strip, (arguments.width, int(strip.shape[0] * scale)))
        path = output / f"{image_id}.jpg"
        cv2.imwrite(str(path), strip)
        print(f"{image_id}: {path}")


if __name__ == "__main__":
    main()
