from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from aic_v2.constants import IMAGE_SUFFIXES, MODALITY_ORDER, NUM_CLASSES
from .split import infer_group_id


def find_image(directory: Path, image_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = directory / f"{image_id}{suffix}"
        if path.exists(): return path
    return None


def file_sha256(path: Path, chunk=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk): digest.update(data)
    return digest.hexdigest()


def read_labels(path: Path) -> tuple[np.ndarray, list[str]]:
    labels, errors = [], []
    if not path.exists(): return np.zeros((0, 5), np.float32), ["missing_label"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line_{line_no}:columns={len(fields)}"); continue
        try: row = [float(v) for v in fields]
        except ValueError:
            errors.append(f"line_{line_no}:non_numeric"); continue
        if not 0 <= int(row[0]) < NUM_CLASSES: errors.append(f"line_{line_no}:class")
        labels.append(row)
    return np.asarray(labels, np.float32).reshape(-1, 5), errors


def audit_dataset(root: str | Path, include_hashes: bool = True) -> list[dict]:
    root = Path(root)
    label_dir = root / "labels"
    ids = sorted(p.stem for p in label_dir.glob("*.txt")) if label_dir.exists() else sorted(p.stem for p in (root / "visible").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    records = []
    for image_id in ids:
        paths = {m: find_image(root / m, image_id) for m in MODALITY_ORDER}
        labels, errors = read_labels(label_dir / f"{image_id}.txt") if label_dir.exists() else (np.zeros((0, 5), np.float32), [])
        visible = cv2.imread(str(paths["visible"]), cv2.IMREAD_UNCHANGED) if paths["visible"] else None
        x1 = labels[:, 1] - labels[:, 3] / 2 if len(labels) else np.array([])
        y1 = labels[:, 2] - labels[:, 4] / 2 if len(labels) else np.array([])
        x2 = labels[:, 1] + labels[:, 3] / 2 if len(labels) else np.array([])
        y2 = labels[:, 2] + labels[:, 4] / 2 if len(labels) else np.array([])
        row = {
            "image_id": image_id, "group_id": infer_group_id(image_id),
            "width": int(visible.shape[1]) if visible is not None else None,
            "height": int(visible.shape[0]) if visible is not None else None,
            "extension": paths["visible"].suffix.lower() if paths["visible"] else None,
            "num_boxes": len(labels), "classes": sorted(set(labels[:, 0].astype(int).tolist())),
            "strict_out_of_bounds": int(((x1 < 0) | (y1 < 0) | (x2 > 1) | (y2 > 1)).sum()),
            "touch_or_out_of_bounds": int(((x1 <= 0) | (y1 <= 0) | (x2 >= 1) | (y2 >= 1)).sum()),
            "errors": errors + [f"missing_{m}" for m, path in paths.items() if path is None],
        }
        for modality, path in paths.items():
            row[f"{modality}_path"] = str(path) if path else None
            if include_hashes and path: row[f"{modality}_sha256"] = file_sha256(path)
        records.append(row)
    return records


def save_records(records: list[dict], output: str | Path) -> None:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")


def summarize(records: list[dict]) -> dict:
    classes = Counter(c for r in records for c in r["classes"])
    return {"images": len(records), "boxes": sum(r["num_boxes"] for r in records), "strict_out_of_bounds": sum(r["strict_out_of_bounds"] for r in records), "touch_or_out_of_bounds": sum(r["touch_or_out_of_bounds"] for r in records), "images_per_class": dict(classes)}
