from __future__ import annotations

import math
from pathlib import Path


def write_prediction(path, boxes, scores, labels):
    """boxes 为原图归一化 cxcywh, 直接按赛题格式写出: cls cx cy w h conf。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box, score, label in zip(boxes, scores, labels):
        center_x, center_y, box_width, box_height = [float(value) for value in box]
        lines.append(f"{int(label)} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f} {float(score):.8f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_submission(directory, expected_ids, num_classes=12, max_detections=100):
    directory = Path(directory)
    errors = []
    actual = {path.stem for path in directory.glob("*.txt")}
    if missing := set(expected_ids) - actual:
        errors.append(f"缺少 {len(missing)} 个文件")
    if extra := actual - set(expected_ids):
        errors.append(f"多出 {len(extra)} 个文件")
    for image_id in expected_ids:
        path = directory / f"{image_id}.txt"
        if not path.exists():
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) > max_detections:
            errors.append(f"{image_id}: 超过 {max_detections} 框")
        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 6:
                errors.append(f"{image_id}:{line_number}: 列数")
                continue
            try:
                cls, *values = [float(value) for value in fields]
            except ValueError:
                errors.append(f"{image_id}:{line_number}: 非数值")
                continue
            if not all(math.isfinite(value) for value in [cls, *values]):
                errors.append(f"{image_id}:{line_number}: NaN/Inf")
            if cls != int(cls) or not 0 <= cls < num_classes:
                errors.append(f"{image_id}:{line_number}: 类别")
            if not all(0 <= value <= 1 for value in values):
                errors.append(f"{image_id}:{line_number}: 坐标或置信度")
    return errors
