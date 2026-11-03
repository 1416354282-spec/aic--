#!/usr/bin/env python3
"""训练集上按类别独立扫置信度阈值, 选使各类 AP@50-95 最优的提交阈值。

与提交管线一致: TTA + batched NMS, confidence=0.001 全保留, 然后逐类在
阈值网格上计算 AP (每类 AP 只依赖该类预测, 可独立寻优后组合)。

用法:
  python scripts/diag_class_ap.py <checkpoint.pt> [--config configs/base.yaml]
      [--overlay configs/remote_westd.yaml] [--overlay configs/experiments/rtdetr_x.yaml]
      [--output outputs/class_thresholds.json] [--tta]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.dataset import MultiModalDataset, collate_fn
from aic_v6.engine.evaluator import predict_batch, predict_batch_tta
from aic_v6.metrics.detection_map import detection_map
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR

CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]
GRID = np.round(np.arange(0.15, 0.96, 0.05), 2).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--output", default=str(ROOT / "outputs/class_thresholds.json"))
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device")
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overlay)
    device = torch.device(arguments.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
    dataset = MultiModalDataset(config["data"]["train_root"], None, size, False, config["data"], {})
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"训练集 {len(dataset)} 张, 分辨率 {size}, TTA={arguments.tta}")

    model = TriModalRTDETR(config, device)
    state = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema") or state["model"])
    model.eval()

    postprocess = {"confidence": 0.001, "nms_iou": 0.7, "max_detections": 100}
    predictions, ground_truth = [], []
    with torch.no_grad():
        predict_fn = predict_batch_tta if arguments.tta else predict_batch
        for batch in loader:
            for key in ("img", "ir", "depth", "depth_valid"):
                batch[key] = batch[key].to(device, non_blocking=True)
            detections = predict_fn(model, batch, batch["meta"], postprocess)
            for i, image_id in enumerate(batch["image_ids"]):
                for j in range(int(batch["num_boxes"][i])):
                    ground_truth.append({
                        "image_id": image_id,
                        "class_id": int(batch["labels"][i, j]),
                        "bbox": batch["boxes"][i, j].tolist(),
                    })
                result = detections[i]
                for box, score, cls in zip(result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()):
                    predictions.append({
                        "image_id": image_id,
                        "class_id": int(cls),
                        "confidence": float(score),
                        "bbox": box.tolist(),
                    })
    print(f"预测 {len(predictions)} 框, GT {len(ground_truth)}")

    # 逐类独立扫阈值
    best = {}
    for cls in range(12):
        class_preds = [p for p in predictions if p["class_id"] == cls]
        class_truth = [t for t in ground_truth if t["class_id"] == cls]
        aps = []
        for threshold in GRID:
            filtered = [p for p in class_preds if p["confidence"] >= threshold]
            metrics = detection_map(filtered, class_truth, num_classes=12)
            aps.append(float(np.mean(metrics["per_class"][cls])))
        aps = np.asarray(aps)
        # 3 点滑动平均平滑, 抗稀有类噪声
        smooth = np.convolve(aps, np.ones(3) / 3, mode="same")
        best_idx = int(np.argmax(smooth))
        best[str(cls)] = {"threshold": GRID[best_idx], "ap50_95": float(aps[best_idx]),
                          "ap_smoothed": float(smooth[best_idx]), "curve": dict(zip(GRID, aps.tolist()))}
        curve = " ".join(f"{t}:{a:.3f}" for t, a in zip(GRID, aps))
        print(f"{cls:2d} {CLASS_NAMES[cls]:12s} 最优 τ={GRID[best_idx]:.2f} "
              f"(AP={aps[best_idx]:.3f}, 平滑={smooth[best_idx]:.3f})  [GT={len(class_truth)}]")

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "checkpoint": arguments.checkpoint, "tta": arguments.tta, "size": list(size),
        "per_class": best,
        "recommended": {cls: best[str(cls)]["threshold"] for cls in range(12)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n推荐阈值: {json.dumps({int(k): v['threshold'] for k, v in best.items()})}")
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()
