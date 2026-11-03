from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import IMAGE_SUFFIXES, NUM_CLASSES


def find_image(directory: Path, image_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = directory / f"{image_id}{suffix}"
        if path.exists():
            return path
    return None


def read_labels(path: Path) -> tuple[np.ndarray, list[str]]:
    labels, errors = [], []
    if not path.exists():
        return np.zeros((0, 5), np.float32), ["missing_label"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line_{line_number}:columns={len(fields)}")
            continue
        try:
            row = [float(value) for value in fields]
        except ValueError:
            errors.append(f"line_{line_number}:non_numeric")
            continue
        if not 0 <= int(row[0]) < NUM_CLASSES:
            errors.append(f"line_{line_number}:class")
            continue
        labels.append(row)
    return np.asarray(labels, np.float32).reshape(-1, 5), errors
