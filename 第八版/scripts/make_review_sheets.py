"""生成「候选框裁片审查表」: 把每个待审框放大裁出并编号, 供逐框目视判定是否为背景错框。

为什么用裁片而不是全图缩略图: 城市场景目标很小, 1920x1080 缩到 1400 宽后 10px 的框
只剩 7px, 无法判断框底下有没有物体。裁片把每个框放到足够大, 才判得准。

用法:
  python scripts/make_review_sheets.py --predictions <txt目录> --images <visible目录> \
      --output <审查表目录> --min-conf 0.10 --per-sheet 12
产出:
  sheet_000.png ... 以及 sheets.json: 每个编号 -> {image_id, class, conf, bbox}
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.constants import CLASS_NAMES, IMAGE_SUFFIXES

CELL = 340          # 每个裁片在审查表里的边长
BORDER = 22         # 裁片顶部留给编号/类别/置信度的文字带
PADDING = 1.6       # 裁片相对框的放大倍数 (留出上下文)
MIN_CROP = 96       # 最小裁片边长 (原图像素)


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def find_image(directory, image_id):
    for suffix in IMAGE_SUFFIXES:
        path = directory / f"{image_id}{suffix}"
        if path.exists():
            return path
    return None


def make_crop(image, box, index, cls, conf):
    height, width = image.shape[:2]
    cx, cy, bw, bh = box
    side = max(bw * width, bh * height) * PADDING
    side = max(side, MIN_CROP)
    x1 = int(round(cx * width - side / 2))
    y1 = int(round(cy * height - side / 2))
    x2 = x1 + int(round(side))
    y2 = y1 + int(round(side))
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - width), max(0, y2 - height)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=(40, 40, 40))
    # 框在裁片内的坐标
    bx1 = int(round((cx - bw / 2) * width)) - x1 + pad_left
    by1 = int(round((cy - bh / 2) * height)) - y1 + pad_top
    bx2 = int(round((cx + bw / 2) * width)) - x1 + pad_left
    by2 = int(round((cy + bh / 2) * height)) - y1 + pad_top
    scale = (CELL - BORDER) / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), CELL - BORDER),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    if crop.shape[1] < CELL:  # 补足宽度, 保持网格整齐
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, CELL - crop.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    elif crop.shape[1] > CELL:
        crop = crop[:, :CELL]
    cv2.rectangle(crop, (int(bx1 * scale), int(by1 * scale)), (int(bx2 * scale), int(by2 * scale)),
                  (0, 255, 255), 2)
    cell = np.full((CELL, CELL, 3), 40, np.uint8)
    cell[BORDER:, :, :] = crop
    cv2.putText(cell, f"#{index} {CLASS_NAMES[cls][:9]} {conf:.2f}", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-conf", type=float, default=0.10)
    parser.add_argument("--max-conf", type=float, default=1.01)
    parser.add_argument("--per-sheet", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 张图")
    parser.add_argument("--ids-file", default=None, help="只处理这些 id (每行一个)")
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    if arguments.ids_file:
        ids = [line.strip() for line in Path(arguments.ids_file).read_text(encoding="utf-8").split() if line.strip()]
    else:
        ids = sorted(path.stem for path in Path(arguments.predictions).glob("*.txt"))
    if arguments.limit:
        ids = ids[: arguments.limit]

    mapping = {}
    cells = []
    for image_id in ids:
        prediction_path = Path(arguments.predictions) / f"{image_id}.txt"
        image_path = find_image(Path(arguments.images), image_id)
        if not prediction_path.exists() or image_path is None:
            continue
        rows = []
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 6:
                continue
            cls, cx, cy, w, h, conf = [float(v) for v in fields]
            if arguments.min_conf <= conf < arguments.max_conf:
                rows.append((int(cls), cx, cy, w, h, conf))
        if not rows:
            continue
        image = imread(image_path)
        for cls, cx, cy, w, h, conf in sorted(rows, key=lambda r: -r[5]):
            index = len(mapping)
            cell = make_crop(image, (cx, cy, w, h), index, cls, conf)
            if cell is None:
                continue
            mapping[str(index)] = {"image_id": image_id, "class": cls, "class_name": CLASS_NAMES[cls],
                                   "confidence": round(conf, 6), "bbox": [cx, cy, w, h]}
            cells.append(cell)

    columns = 4
    sheets = 0
    for start in range(0, len(cells), arguments.per_sheet):
        chunk = cells[start:start + arguments.per_sheet]
        rows_count = math.ceil(len(chunk) / columns)
        sheet = np.full((rows_count * CELL, columns * CELL, 3), 20, np.uint8)
        for position, cell in enumerate(chunk):
            row, column = divmod(position, columns)
            sheet[row * CELL:(row + 1) * CELL, column * CELL:(column + 1) * CELL] = cell
        cv2.imwrite(str(output / f"sheet_{sheets:04d}.png"), sheet)
        sheets += 1

    (output / "sheets.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    print(f"待审框 {len(cells)} 个, 覆盖 {len({v['image_id'] for v in mapping.values()})} 张图")
    print(f"审查表 {sheets} 张 -> {output}")
    print(f"其中 conf>=0.33 的待审框: {sum(1 for v in mapping.values() if v['confidence'] >= 0.33)}")


if __name__ == "__main__":
    main()
