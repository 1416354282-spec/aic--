"""测试「TTA 一致性」作为背景错框判据。

思路: production 用水平翻转 TTA 合并, 但只在单侧出现的框 (另一分支同类别位置没有对应框)
往往是把纹理/背景当成目标的幻觉。对每个最终框取
    agreement = min( 与 pass0 同类框的最大 IoU, 与 pass1 同类框的最大 IoU )
两分支都提出的框 agreement≈1, 只有单侧提出的框 agreement 很小。

若该信号能复现 oracle「删 IoU==0 框」的收益, 就可以全自动删错框, 无需人工逐框审。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aic_v8.data.boxes import box_iou, cxcywh_to_xyxy
from aic_v8.data.manifest import read_labels
from aic_v8.engine.evaluator import _prediction_to_original
from aic_v8.engine.trainer import read_ids
from tune_thresholds import Meta, average_precision, build_gt_index, replay


def pass_boxes_original(image, pass_index, global_threshold, max_detections, pre_cap):
    """单个 TTA 分支解码到原图归一化坐标 (翻转分支先把 cx 镜像回来)。"""
    source = image["passes"][pass_index]
    scores = torch.tensor(source["scores"], dtype=torch.float32)
    labels = torch.tensor(source["labels"], dtype=torch.float32)
    boxes = torch.tensor(source["boxes"], dtype=torch.float32)
    valid = scores > global_threshold
    scores, labels, boxes = scores[valid], labels[valid], boxes[valid]
    if len(scores):
        order = scores.argsort(descending=True)[:pre_cap]
        scores, labels, boxes = scores[order], labels[order], boxes[order]
    else:
        boxes, scores, labels = boxes[:0], scores[:0], labels[:0]
    if pass_index == 1 and len(boxes):
        boxes = boxes.clone()
        boxes[:, 0] = 1 - boxes[:, 0]
    meta = image["meta"]
    handle = Meta()
    handle.original_hw = tuple(meta["original_hw"])
    handle.output_hw = tuple(meta["output_hw"])
    handle.ratio = float(meta["ratio"])
    handle.pad_xy = tuple(meta["pad_xy"])
    if len(boxes):
        boxes = torch.stack([_prediction_to_original(box, handle) for box in boxes])
    return boxes.numpy(), scores.numpy(), labels.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--global", dest="global_threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--pre-cap", type=int, default=300)
    arguments = parser.parse_args()

    images = []
    with open(arguments.raw, encoding="utf-8") as handle:
        for line in handle:
            images.append(json.loads(line))
    val_ids = read_ids(Path(arguments.split_dir) / f"fold_{arguments.fold}_val.txt")
    ground_truth = []
    for image_id in val_ids:
        labels, _ = read_labels(Path(arguments.train_root) / "labels" / f"{image_id}.txt")
        for row in labels:
            ground_truth.append({"image_id": image_id, "class_id": int(row[0]), "bbox": [float(v) for v in row[1:5]]})
    gt_index = build_gt_index(ground_truth)
    per_image_gt = defaultdict(list)
    for item in ground_truth:
        per_image_gt[item["image_id"]].append(item["bbox"])

    predictions, agreements, overlaps = [], [], []
    for image in images:
        boxes, scores, labels = replay(image, {}, arguments.global_threshold, arguments.nms_iou,
                                       arguments.max_detections)
        side = []
        for index in (0, 1):
            side.append(pass_boxes_original(image, index, arguments.global_threshold,
                                           arguments.max_detections, arguments.pre_cap))
        xyxy = cxcywh_to_xyxy(boxes.numpy())
        for position, (box, score, label) in enumerate(zip(boxes.tolist(), scores.tolist(), labels.tolist())):
            cls = int(label)
            best = []
            for side_boxes, side_scores, side_labels in side:
                same = side_labels == cls
                if not same.any():
                    best.append(0.0)
                    continue
                target = cxcywh_to_xyxy(side_boxes[same])
                best.append(float(box_iou(xyxy[position:position + 1], target)[0].max()))
            agreements.append(min(best))
            ground = per_image_gt.get(image["image_id"], [])
            overlaps.append(0.0 if not ground else float(
                box_iou(xyxy[position:position + 1], cxcywh_to_xyxy(np.asarray(ground, np.float32)))[0].max()))
            predictions.append({"image_id": image["image_id"], "class_id": cls,
                                "confidence": float(score), "bbox": box})

    agreements = np.asarray(agreements)
    overlaps = np.asarray(overlaps)
    confidences = np.asarray([p["confidence"] for p in predictions])
    baseline = average_precision(predictions, gt_index, 12)["map50_95"]
    print(f"验证图 {len(images)} 张, 框 {len(predictions)} 个, 基线 map50_95 = {baseline:.4f}\n")

    print("== 一致性信号对「完全无重叠 (IoU==0)」错框的判别力")
    print(f"{'agreement阈值':>14} {'删除数':>7} {'删对':>6} {'误删TP':>7} {'精确率':>7} {'召回率':>7}")
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        delete = agreements < threshold
        junk_true = overlaps < 1e-9
        correct = int((delete & junk_true).sum())
        wrong = int((delete & ~junk_true).sum())
        total_junk = int(junk_true.sum())
        print(f"{threshold:14.2f} {int(delete.sum()):7d} {correct:6d} {wrong:7d} "
              f"{100.0 * correct / max(int(delete.sum()), 1):6.1f}% "
              f"{100.0 * correct / max(total_junk, 1):6.1f}%")

    print("\n== 按一致性删除后的收益 (对比: 删 IoU==0 框的 oracle 上限)")
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        keep = agreements >= threshold
        kept = [p for p, k in zip(predictions, keep) if k]
        result = average_precision(kept, gt_index, 12)["map50_95"]
        print(f"  删 agreement<{threshold:.1f}: 删除 {len(predictions) - len(kept):5d} 框, "
              f"map50_95={result:.4f}, Δ={result - baseline:+.4f}")
    keep = overlaps >= 1e-9
    kept = [p for p, k in zip(predictions, keep) if k]
    result = average_precision(kept, gt_index, 12)["map50_95"]
    print(f"  oracle 删 IoU==0  : 删除 {len(predictions) - len(kept):5d} 框, "
          f"map50_95={result:.4f}, Δ={result - baseline:+.4f}")


if __name__ == "__main__":
    main()
