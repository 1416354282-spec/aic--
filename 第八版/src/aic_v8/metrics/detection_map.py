from __future__ import annotations

from collections import defaultdict

import numpy as np

from aic_v8.data.boxes import box_iou, cxcywh_to_xyxy


def detection_map(predictions, ground_truth, num_classes=12, thresholds=None):
    thresholds = np.linspace(0.50, 0.95, 10) if thresholds is None else np.asarray(thresholds)
    per_class = {}
    for cls in range(num_classes):
        predictions_for_class = sorted(
            [item for item in predictions if item["class_id"] == cls], key=lambda item: item["confidence"], reverse=True
        )
        truth_for_class = [item for item in ground_truth if item["class_id"] == cls]
        average_precisions = []
        for threshold in thresholds:
            matched = defaultdict(set)
            true_positive = np.zeros(len(predictions_for_class))
            false_positive = np.zeros(len(predictions_for_class))
            for prediction_index, prediction in enumerate(predictions_for_class):
                candidates = [
                    (index, truth) for index, truth in enumerate(truth_for_class)
                    if truth["image_id"] == prediction["image_id"] and index not in matched[prediction["image_id"]]
                ]
                if candidates:
                    predicted_box = cxcywh_to_xyxy(np.asarray([prediction["bbox"]], np.float32))
                    target_boxes = cxcywh_to_xyxy(np.asarray([truth["bbox"] for _, truth in candidates], np.float32))
                    overlaps = box_iou(predicted_box, target_boxes)[0]
                    best = int(np.argmax(overlaps))
                    if overlaps[best] >= threshold:
                        true_positive[prediction_index] = 1
                        matched[prediction["image_id"]].add(candidates[best][0])
                        continue
                false_positive[prediction_index] = 1
            recall = np.cumsum(true_positive) / max(len(truth_for_class), 1)
            precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive + false_positive), 1e-12)
            average_precisions.append(float(np.mean([
                precision[recall >= point].max() if np.any(recall >= point) else 0 for point in np.linspace(0, 1, 101)
            ])))
        per_class[cls] = average_precisions
    values = list(per_class.values())
    return {
        "map50": float(np.mean([value[0] for value in values])),
        "map75": float(np.mean([value[5] for value in values])),
        "map50_95": float(np.mean([np.mean(value) for value in values])),
        "per_class": per_class,
    }
