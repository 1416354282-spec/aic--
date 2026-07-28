from __future__ import annotations

from collections import defaultdict

import numpy as np

from aic_v2.data.boxes import box_iou, cxcywh_to_xyxy


def detection_map(predictions, ground_truth, num_classes=12, thresholds=None):
    thresholds = np.linspace(0.50, 0.95, 10) if thresholds is None else np.asarray(thresholds)
    per_class = {}
    for cls in range(num_classes):
        preds = sorted([p for p in predictions if p["class_id"] == cls], key=lambda p: p["confidence"], reverse=True)
        gts = [g for g in ground_truth if g["class_id"] == cls]
        aps = []
        for threshold in thresholds:
            matched = defaultdict(set); tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
            for i, pred in enumerate(preds):
                candidates = [(j, gt) for j, gt in enumerate(gts) if gt["image_id"] == pred["image_id"] and j not in matched[pred["image_id"]]]
                if candidates:
                    pbox = cxcywh_to_xyxy(np.asarray([pred["bbox"]], np.float32))
                    gboxes = cxcywh_to_xyxy(np.asarray([gt["bbox"] for _, gt in candidates], np.float32))
                    ious = box_iou(pbox, gboxes)[0]; best = int(np.argmax(ious))
                    if ious[best] >= threshold: tp[i] = 1; matched[pred["image_id"]].add(candidates[best][0]); continue
                fp[i] = 1
            recall = np.cumsum(tp) / max(len(gts), 1); precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1e-12)
            aps.append(float(np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0 for r in np.linspace(0, 1, 101)])))
        per_class[cls] = aps
    values = [v for v in per_class.values() if v]
    return {"map50": float(np.mean([v[0] for v in values])) if values else 0.0, "map75": float(np.mean([v[5] for v in values])) if values else 0.0, "map50_95": float(np.mean([np.mean(v) for v in values])) if values else 0.0, "per_class": per_class}
