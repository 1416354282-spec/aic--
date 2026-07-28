from __future__ import annotations

import torch
from torchvision.ops import batched_nms


def class_aware_postprocess(boxes, scores, confidence=0.001, iou=0.7, max_detections=100, multi_label=True):
    results = []
    for image_boxes, image_scores in zip(boxes, scores):
        if multi_label:
            anchors, labels = torch.where(image_scores > confidence)
            candidate_boxes = image_boxes[anchors]
            candidate_scores = image_scores[anchors, labels]
        else:
            candidate_scores, labels = image_scores.max(1)
            mask = candidate_scores > confidence
            candidate_boxes, candidate_scores, labels = image_boxes[mask], candidate_scores[mask], labels[mask]
        if not len(candidate_boxes):
            results.append({"boxes": image_boxes.new_zeros((0, 4)), "scores": image_scores.new_zeros(0), "labels": torch.zeros(0, dtype=torch.long, device=image_scores.device)}); continue
        order = candidate_scores.argsort(descending=True)[:30000]
        candidate_boxes, candidate_scores, labels = candidate_boxes[order], candidate_scores[order], labels[order]
        keep = batched_nms(candidate_boxes, candidate_scores, labels, iou)[:max_detections]
        results.append({"boxes": candidate_boxes[keep], "scores": candidate_scores[keep], "labels": labels[keep]})
    return results
