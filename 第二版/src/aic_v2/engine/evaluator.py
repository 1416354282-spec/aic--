from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from aic_v2.inference.decode import decode_dfl, make_anchors, unpack_raw_predictions
from aic_v2.inference.postprocess import class_aware_postprocess
from aic_v2.metrics.detection_map import detection_map


def _head_metadata(model):
    head = model.model[-1]
    strides = [int(x) for x in head.stride.detach().cpu().tolist()]
    return int(getattr(head, "reg_max", 16)), int(head.nc), strides


def _feature_shapes(image_size, strides):
    return [(image_size // stride, image_size // stride) for stride in strides]


def _prediction_to_original(box, meta):
    """letterbox 像素 xyxy -> 原图归一化 xyxy。"""
    h, w = meta.original_hw
    left, top = meta.pad_xy
    output = box.clone()
    output[[0, 2]] = (output[[0, 2]] - left) / meta.ratio / w
    output[[1, 3]] = (output[[1, 3]] - top) / meta.ratio / h
    return output.clamp(0, 1)


@torch.no_grad()
def predict_batch(model, images, metas, image_size, postprocess_config):
    raw = model(images)
    reg_max, num_classes, strides = _head_metadata(model)
    anchors, stride_tensor = make_anchors(_feature_shapes(image_size, strides), strides, images.device)
    boxes_raw, scores_raw = unpack_raw_predictions(raw, reg_max, num_classes)
    boxes, scores = decode_dfl(boxes_raw, scores_raw, anchors, stride_tensor, reg_max)
    detections = class_aware_postprocess(
        boxes, scores,
        confidence=postprocess_config.get("confidence", 0.001),
        iou=postprocess_config.get("nms_iou", 0.7),
        max_detections=postprocess_config.get("max_detections", 100),
        multi_label=postprocess_config.get("multi_label", True),
    )
    for result, meta in zip(detections, metas):
        if len(result["boxes"]):
            result["boxes"] = torch.stack([_prediction_to_original(box, meta) for box in result["boxes"]])
    return detections


@torch.no_grad()
def evaluate(model, dataset, config, device):
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=config["train"].get("workers", 0), collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None)
    model.eval(); predictions, ground_truth = [], []
    for batch in loader:
        images = batch["img"].to(device)
        detections = predict_batch(model, images, batch["meta"], config["data"]["image_size"], config["validation"])
        for index, image_id in enumerate(batch["image_ids"]):
            mask = batch["batch_idx"] == index
            for cls, box in zip(batch["cls"][mask].flatten().tolist(), batch["bboxes"][mask].tolist()):
                ground_truth.append({"image_id": image_id, "class_id": int(cls), "bbox": box})
            result = detections[index]
            for box, score, cls in zip(result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({"image_id": image_id, "class_id": int(cls), "confidence": float(score), "bbox": [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]})
    return detection_map(predictions, ground_truth, config["model"]["num_classes"])
