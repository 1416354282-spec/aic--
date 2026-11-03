#!/usr/bin/env python3
"""冒烟验证: 用指定权重对若干测试帧推理 (1920x1088 + TTA, 原始 conf=0.001),
与已有原始预测目录逐框对比, 确认权重与提交一致。

用法:
  python scripts/smoke_repro.py /root/aic/v6/runs_x_last_local.pt \
      --frames 00000286 00000287 --reference outputs/submissions/x_last_1920_tta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.config import load_config
from aic_v8.data.dataset import MultiModalDataset, collate_fn
from aic_v8.engine.evaluator import predict_batch_tta
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR


def box_iou(first, second, eps=1e-7):
    first = np.asarray(first, np.float32); second = np.asarray(second, np.float32)
    intersection_size = np.maximum(0, np.minimum(first[:, None, 2:], second[None, :, 2:]) - np.maximum(first[:, None, :2], second[None, :, :2]))
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_first = np.maximum(0, first[:, 2] - first[:, 0]) * np.maximum(0, first[:, 3] - first[:, 1])
    area_second = np.maximum(0, second[:, 2] - second[:, 0]) * np.maximum(0, second[:, 3] - second[:, 1])
    return intersection / (area_first[:, None] + area_second[None, :] - intersection + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--reference", required=True, help="原始预测目录")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overlay, arguments.set)
    device = torch.device(arguments.device)
    size = tuple(int(value) for value in config["train"]["phases"][-1]["size"])
    dataset = MultiModalDataset(config["data"]["test_root"], arguments.frames, size, False, config["data"], {})
    model = TriModalRTDETR(config, device)
    state = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema") or state["model"])
    model.eval()

    postprocess = {**config["validation"], "confidence": 0.001, "class_confidence": {}}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    reference_dir = Path(arguments.reference)
    with torch.no_grad():
        for batch in loader:
            for key in ("img", "ir", "depth", "depth_valid"):
                batch[key] = batch[key].to(device, non_blocking=True)
            detections = predict_batch_tta(model, batch, batch["meta"], postprocess)
            for image_id, result in zip(batch["image_ids"], detections):
                predicted = np.column_stack((
                    result["labels"].cpu().numpy(), result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()))
                order = np.argsort(-predicted[:, 5])
                predicted = predicted[order]
                reference = np.loadtxt(reference_dir / f"{image_id}.txt", ndmin=2)
                if reference.size:
                    reference = reference[np.argsort(-reference[:, 5])]
                print(f"{image_id}: 我的框 {len(predicted)}, ref 框 {len(reference)}")
                delta_scores, matched = [], 0
                def to_xyxy(rows):
                    xyxy = np.asarray(rows, np.float32).copy()
                    xyxy[:, 0] -= xyxy[:, 2] / 2
                    xyxy[:, 1] -= xyxy[:, 3] / 2
                    xyxy[:, 2] += xyxy[:, 0]
                    xyxy[:, 3] += xyxy[:, 1]
                    return xyxy
                for index, row in enumerate(reference):
                    cls_mask = predicted[:, 0].astype(int) == int(row[0])
                    if not cls_mask.any():
                        if index < 10:
                            print(f"  ref[{index}] cls={int(row[0])} conf={row[5]:.3f} 无同类候选")
                        continue
                    pred_box = to_xyxy(predicted[cls_mask, 1:5])
                    ref_box = to_xyxy(row[1:5][None, :])
                    iou = box_iou(pred_box, ref_box)[0]
                    best = int(np.argmax(iou))
                    if iou[best] >= 0.5:
                        matched += 1
                        delta_scores.append(abs(float(predicted[cls_mask][best, 5]) - float(row[5])))
                    elif index < 10:
                        print(f"  ref[{index}] cls={int(row[0])} conf={row[5]:.3f} box={row[1:5].round(3)} "
                              f"最近IoU={iou[best]:.2f}")
                print(f"  匹配 {matched}/{len(reference)}, "
                      f"分数差 max={max(delta_scores) if delta_scores else float('nan'):.2e} "
                      f"mean={np.mean(delta_scores) if delta_scores else float('nan'):.2e}")


if __name__ == "__main__":
    main()
