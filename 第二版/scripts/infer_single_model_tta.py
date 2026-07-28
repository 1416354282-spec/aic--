#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aic_v2.config import load_config
from aic_v2.data.dataset import MultiModalDataset, collate_fn
from aic_v2.inference.decode import decode_dfl, make_anchors, unpack_raw_predictions
from aic_v2.inference.submission import write_prediction
from aic_v2.models.early_fusion import EarlyFusionYOLO


# This script is single-model inference only: one checkpoint, optional multi-scale
# and horizontal-flip TTA, then one final class-aware NMS. It does not ensemble
# multiple checkpoints or model architectures.


def _head_metadata(model):
    head = model.model[-1]
    strides = [int(x) for x in head.stride.detach().cpu().tolist()]
    return int(getattr(head, "reg_max", 16)), int(head.nc), strides


def _feature_shapes(image_size: int, strides: list[int]):
    return [(image_size // stride, image_size // stride) for stride in strides]


def _boxes_to_original(boxes: torch.Tensor, meta) -> torch.Tensor:
    h, w = meta.original_hw
    left, top = meta.pad_xy
    output = boxes.clone()
    output[:, [0, 2]] = (output[:, [0, 2]] - left) / meta.ratio / w
    output[:, [1, 3]] = (output[:, [1, 3]] - top) / meta.ratio / h
    return output.clamp(0, 1)


def _unflip_boxes(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    output = boxes.clone()
    output[:, 0] = image_size - boxes[:, 2]
    output[:, 2] = image_size - boxes[:, 0]
    return output


def _valid_box_mask(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])


@torch.no_grad()
def predict_candidates(
    model,
    images: torch.Tensor,
    metas,
    image_size: int,
    confidence: float,
    multi_label: bool,
    pre_topk: int,
    flipped: bool,
):
    raw = model(images)
    reg_max, num_classes, strides = _head_metadata(model)
    anchors, stride_tensor = make_anchors(_feature_shapes(image_size, strides), strides, images.device)
    boxes_raw, scores_raw = unpack_raw_predictions(raw, reg_max, num_classes)
    boxes, scores = decode_dfl(boxes_raw, scores_raw, anchors, stride_tensor, reg_max)

    batch_results = []
    for image_boxes, image_scores, meta in zip(boxes, scores, metas):
        if flipped:
            image_boxes = _unflip_boxes(image_boxes, image_size)
        if multi_label:
            anchors_idx, labels = torch.where(image_scores > confidence)
            candidate_boxes = image_boxes[anchors_idx]
            candidate_scores = image_scores[anchors_idx, labels]
        else:
            candidate_scores, labels = image_scores.max(1)
            mask = candidate_scores > confidence
            candidate_boxes = image_boxes[mask]
            candidate_scores = candidate_scores[mask]
            labels = labels[mask]

        if len(candidate_boxes):
            order = candidate_scores.argsort(descending=True)[:pre_topk]
            candidate_boxes = _boxes_to_original(candidate_boxes[order], meta)
            candidate_scores = candidate_scores[order]
            labels = labels[order]
            valid = _valid_box_mask(candidate_boxes)
            candidate_boxes = candidate_boxes[valid]
            candidate_scores = candidate_scores[valid]
            labels = labels[valid]
        else:
            candidate_boxes = image_boxes.new_zeros((0, 4))
            candidate_scores = image_scores.new_zeros(0)
            labels = torch.zeros(0, dtype=torch.long, device=image_scores.device)

        batch_results.append({"boxes": candidate_boxes, "scores": candidate_scores, "labels": labels})
    return batch_results


def final_nms(candidate_sets, iou: float, max_detections: int):
    boxes = torch.cat([x["boxes"] for x in candidate_sets if len(x["boxes"])], 0) if candidate_sets else None
    if boxes is None or not len(boxes):
        device = candidate_sets[0]["boxes"].device if candidate_sets else torch.device("cpu")
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros(0, device=device),
            "labels": torch.zeros(0, dtype=torch.long, device=device),
        }
    scores = torch.cat([x["scores"] for x in candidate_sets if len(x["boxes"])], 0)
    labels = torch.cat([x["labels"] for x in candidate_sets if len(x["boxes"])], 0)
    keep = batched_nms(boxes, scores, labels, iou)[:max_detections]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def parse_scales(raw_scales, config):
    if raw_scales:
        scales = raw_scales
    else:
        base_size = int(config["data"]["image_size"])
        tta_sizes = [int(x) for x in config.get("inference", {}).get("multiscale_tta", [])]
        scales = [base_size, *tta_sizes]
    output = []
    for scale in scales:
        if scale % 32 != 0:
            raise ValueError(f"输入尺寸必须是 32 的倍数，建议把 {scale} 改为 {round(scale / 32) * 32}")
        if scale not in output:
            output.append(scale)
    return output


def main():
    parser = argparse.ArgumentParser(description="Single-checkpoint YOLOv8m inference with optional TTA")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", default=str(Path(__file__).parents[1] / "outputs/submissions/predictions_tta"))
    parser.add_argument("--device")
    parser.add_argument("--scales", nargs="*", type=int, help="TTA input sizes, e.g. --scales 1280 1504 1664")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal flip TTA")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--nms-iou", type=float)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--pre-topk", type=int, default=5000)
    parser.add_argument("--single-label", action="store_true", help="Use best class per anchor instead of multi-label candidates")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overlay, args.set)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    scales = parse_scales(args.scales, cfg)
    flip_tta = not args.no_flip
    confidence = args.confidence if args.confidence is not None else cfg["validation"].get("confidence", 0.001)
    nms_iou = args.nms_iou if args.nms_iou is not None else cfg["validation"].get("nms_iou", 0.7)
    max_detections = args.max_detections if args.max_detections is not None else cfg["validation"].get("max_detections", 100)
    multi_label = not args.single_label and cfg["validation"].get("multi_label", True)
    batch_size = args.batch_size or cfg["train"].get("batch_size", 1)
    workers = args.workers if args.workers is not None else cfg["train"].get("workers", 0)

    probe_dataset = MultiModalDataset(cfg["data"]["test_root"], None, scales[0], False, cfg["data"], {})
    channels = probe_dataset[0]["img"].shape[0]
    model = EarlyFusionYOLO(cfg["model"]["weights"], channels, cfg["model"]["num_classes"], str(device), False).model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
    model.eval()

    predictions = {image_id: [] for image_id in probe_dataset.ids}
    variants = [(False, "orig")]
    if flip_tta:
        variants.append((True, "hflip"))

    for scale in scales:
        dataset = MultiModalDataset(cfg["data"]["test_root"], probe_dataset.ids, scale, False, cfg["data"], {})
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)
        for batch in loader:
            images = batch["img"].to(device)
            for flipped, _name in variants:
                variant_images = torch.flip(images, dims=[3]) if flipped else images
                detections = predict_candidates(
                    model,
                    variant_images,
                    batch["meta"],
                    scale,
                    confidence,
                    multi_label,
                    args.pre_topk,
                    flipped,
                )
                for image_id, result in zip(batch["image_ids"], detections):
                    predictions[image_id].append({k: v.detach().cpu() for k, v in result.items()})
        print(f"完成尺度 {scale}: variants={len(variants)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id in probe_dataset.ids:
        result = final_nms(predictions[image_id], nms_iou, max_detections)
        write_prediction(
            output_dir / f"{image_id}.txt",
            result["boxes"].cpu(),
            result["scores"].cpu(),
            result["labels"].cpu(),
        )

    meta = {
        "single_model_only": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": cfg["model"]["weights"],
        "scales": scales,
        "horizontal_flip_tta": flip_tta,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "max_detections": max_detections,
        "multi_label": multi_label,
        "pre_topk": args.pre_topk,
    }
    (output_dir.parent / "single_model_tta_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: {len(probe_dataset)} 张 -> {output_dir}")
    print("single_model_only=True; 未加载多个 checkpoint，未做模型集成")


if __name__ == "__main__":
    main()

