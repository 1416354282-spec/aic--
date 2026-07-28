from __future__ import annotations

import math
import zipfile
from pathlib import Path


def write_prediction(path, boxes, scores, labels):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(x) for x in box]
        cx, cy, width, height = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        lines.append(f"{int(label)} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f} {float(score):.8f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_submission(directory, expected_ids, num_classes=12, max_detections=100):
    directory = Path(directory); errors = []
    actual = {p.stem for p in directory.glob("*.txt")}
    if missing := set(expected_ids) - actual: errors.append(f"缺少 {len(missing)} 个文件")
    if extra := actual - set(expected_ids): errors.append(f"多出 {len(extra)} 个文件")
    for image_id in expected_ids:
        path = directory / f"{image_id}.txt"
        if not path.exists(): continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) > max_detections: errors.append(f"{image_id}: 超过 {max_detections} 框")
        for no, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 6: errors.append(f"{image_id}:{no}: 列数"); continue
            try: cls, *values = [float(x) for x in fields]
            except ValueError: errors.append(f"{image_id}:{no}: 非数值"); continue
            if not all(math.isfinite(x) for x in [cls, *values]): errors.append(f"{image_id}:{no}: NaN/Inf")
            if cls != int(cls) or not 0 <= cls < num_classes: errors.append(f"{image_id}:{no}: 类别")
            if not all(0 <= x <= 1 for x in values): errors.append(f"{image_id}:{no}: 坐标或置信度")
    return errors


def make_zip(directory, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(directory).glob("*.txt")): archive.write(path, path.name)
