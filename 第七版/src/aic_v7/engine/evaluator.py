"""v7 evaluator: NMS-free decoding of D-FINE outputs back to original coords.

The D-FINE core returns ``(B, 300, 6)`` normalized [cx, cy, w, h, score, cls]
in letterbox space (score sigmoid, top-300, no NMS) — the same contract as v6's
RTDETRDecoder eval output, so this module is migrated verbatim from v6.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..metrics.detection_map import detection_map


def _prediction_to_original(box, meta):
    """letterbox 图归一化 cxcywh → 原图归一化 cxcywh。

    步骤: 归一化 → letterbox 像素 (乘 output_hw) → 去 pad → 除 ratio → 原图像素 → 归一化。
    """
    height, width = meta.original_hw
    out_w, out_h = meta.output_hw  # letterbox 构造时 output_hw=(size_w, size_h)
    left, top = meta.pad_xy
    output = box.clone()
    output[0] = (output[0] * out_w - left) / meta.ratio / width
    output[1] = (output[1] * out_h - top) / meta.ratio / height
    output[2] = output[2] * out_w / meta.ratio / width
    output[3] = output[3] * out_h / meta.ratio / height
    return output.clamp(0, 1)


def _target_to_original(box, meta, device):
    """GT 标签本来就是原图归一化 cxcywh, 无需坐标转换。"""
    return box


def _class_thresholds(scores, labels, confidence, class_confidence):
    """按类别置信度阈值: 默认全局 confidence, class_confidence {cls: thr} 覆盖个别类别。

    CLI --set 传入的 key 是字符串 ("7"), YAML 配置是 int (7), 统一 int 归一化。
    返回与 scores 同形状的阈值张量。
    """
    thresholds = torch.full_like(scores, confidence)
    for cls, thr in (class_confidence or {}).items():
        thresholds[labels == int(cls)] = float(thr)
    return thresholds


@torch.no_grad()
def _decode_outputs(outputs, batch_size, device, confidence=0.001, max_detections=100, class_confidence=None):
    """D-FINE core 输出 (B, Q, 6) [cx,cy,w,h,score,cls] → per-image detections."""
    y = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    boxes = y[..., :4]    # (B, Q, 4) normalized cxcywh in letterbox space
    scores = y[..., 4]    # 已 sigmoid
    labels = y[..., 5].long()
    thresholds = _class_thresholds(scores, labels, confidence, class_confidence)
    detections = []
    for i in range(batch_size):
        valid = scores[i] > thresholds[i]
        score_i, label_i, box_i = scores[i][valid], labels[i][valid].float(), boxes[i][valid]
        if len(score_i):
            order = score_i.argsort(descending=True)[:max_detections]
            detections.append({
                "boxes": box_i[order],
                "scores": score_i[order],
                "labels": label_i[order],
            })
        else:
            detections.append({"boxes": torch.zeros(0, 4, device=device), "scores": torch.zeros(0, device=device), "labels": torch.zeros(0, device=device)})
    return detections


@torch.no_grad()
def predict_batch(model, batch, metas, postprocess_config):
    outputs = model.predict_raw(batch)
    detections = _decode_outputs(
        outputs, batch["img"].shape[0], batch["img"].device,
        confidence=postprocess_config.get("confidence", 0.001),
        max_detections=postprocess_config.get("max_detections", 100),
        class_confidence=postprocess_config.get("class_confidence"),
    )
    for result, meta in zip(detections, metas):
        if len(result["boxes"]):
            result["boxes"] = torch.stack([_prediction_to_original(box, meta) for box in result["boxes"]])
    return detections


@torch.no_grad()
def predict_batch_tta(model, batch, metas, postprocess_config):
    """TTA: original + horizontal flip, merge with batched NMS."""
    outputs = model.predict_raw(batch)
    detections = _decode_outputs(
        outputs, batch["img"].shape[0], batch["img"].device,
        confidence=postprocess_config.get("confidence", 0.001),
        max_detections=postprocess_config.get("max_detections", 100),
        class_confidence=postprocess_config.get("class_confidence"),
    )

    flipped_batch = dict(batch)
    flipped_batch["img"] = torch.flip(batch["img"], [3])
    if "ir" in flipped_batch:
        flipped_batch["ir"] = torch.flip(batch["ir"], [3])
        flipped_batch["depth"] = torch.flip(batch["depth"], [3])
        flipped_batch["depth_valid"] = torch.flip(batch["depth_valid"], [3])
    outputs_flip = model.predict_raw(flipped_batch, flip_x=True)  # 空间先验 μ_c.x 随翻转坐标系镜像
    det_flip = _decode_outputs(
        outputs_flip, batch["img"].shape[0], batch["img"].device,
        confidence=postprocess_config.get("confidence", 0.001),
        max_detections=postprocess_config.get("max_detections", 100),
        class_confidence=postprocess_config.get("class_confidence"),
    )
    for result in det_flip:
        if len(result["boxes"]):
            result["boxes"][:, 0] = 1 - result["boxes"][:, 0]  # cx 翻转到原空间, w/h 不变

    merged = []
    for orig, flip in zip(detections, det_flip):
        if len(orig["boxes"]) and len(flip["boxes"]):
            boxes = torch.cat([orig["boxes"], flip["boxes"]])
            scores = torch.cat([orig["scores"], flip["scores"]])
            labels = torch.cat([orig["labels"], flip["labels"]])
            order = scores.argsort(descending=True)[:30000]
            boxes, scores, labels = boxes[order], scores[order], labels[order]
            from torchvision.ops import batched_nms
            # batched_nms 需要 xyxy 坐标; cxcywh → xyxy 后计算, 返回索引仍作用于原数组
            xyxy = torch.stack((boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                                boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2), dim=1)
            keep = batched_nms(xyxy, scores, labels, postprocess_config.get("nms_iou", 0.7))
            keep = keep[:postprocess_config.get("max_detections", 100)]
            merged.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        else:
            merged.append(orig if len(orig["boxes"]) else flip)
    for result, meta in zip(merged, metas):
        if len(result["boxes"]):
            result["boxes"] = torch.stack([_prediction_to_original(box, meta) for box in result["boxes"]])
    return merged


@torch.no_grad()
def evaluate(model, dataset, config, device):
    phases = config["train"]["phases"]
    # 验证用大分辨率 (P3 尺寸) 前向, batch 限制 ≤4 避免与训练显存叠加爆峰
    batch_size = min(int(phases[-1]["batch_size"]), 4) if phases else 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["train"].get("workers", 0),
        collate_fn=dataset.collate_fn,
    )
    model.eval()
    predictions, ground_truth = [], []
    for batch in loader:
        for key in ("img", "ir", "depth", "depth_valid", "boxes", "labels", "num_boxes"):
            if torch.is_tensor(batch.get(key)):
                batch[key] = batch[key].to(device, non_blocking=True)
        detections = predict_batch(model, batch, batch["meta"], config["validation"])
        for image_index, image_id in enumerate(batch["image_ids"]):
            num_boxes = int(batch["num_boxes"][image_index])
            for j in range(num_boxes):
                ground_truth.append({
                    "image_id": image_id,
                    "class_id": int(batch["labels"][image_index, j]),
                    "bbox": _target_to_original(batch["boxes"][image_index, j].tolist(), batch["meta"][image_index], batch["img"].device),
                })
            result = detections[image_index]
            for box, score, cls in zip(result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()):
                predictions.append({
                    "image_id": image_id,
                    "class_id": int(cls),
                    "confidence": float(score),
                    "bbox": box.tolist(),  # 已是原图归一化 cxcywh
                })
    return detection_map(predictions, ground_truth, config["model"]["num_classes"])
