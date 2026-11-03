"""评估不同的「每图 100 个名额如何挑」策略 (在验证集上, 预算与提交一致)。

背景: 赛题规则下 AP 对提交框数单调不减, 且每图上限 100 框。所以唯一还有余量的
自由度是「哪 100 个框」。production 是每个 TTA 分支先各自截断到 100 再合并 NMS,
若最终 100 框里存在同类别近似重复框, 名额就被浪费了 —— 把它们换成第 101+ 个
不同目标的框可以提升召回。本脚本量化这件事。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aic_v8.data.manifest import read_labels
from aic_v8.data.boxes import cxcywh_to_xyxy, box_iou
from aic_v8.engine.trainer import read_ids
from tune_thresholds import make_scorer, replay


def duplicate_slots(image, class_thresholds, global_threshold, nms_iou, max_detections, pre_cap, dedup_iou):
    """统计最终 100 框里有多少是「同类别、与更高分框 IoU>dedup_iou」的重复名额。"""
    boxes, scores, labels = replay(image, class_thresholds, global_threshold, nms_iou, max_detections, pre_cap)
    if not len(boxes):
        return 0, 0
    xyxy = cxcywh_to_xyxy(boxes.numpy())
    redundant = 0
    for label in labels.unique().tolist():
        index = (labels == label).nonzero().flatten()
        if len(index) < 2:
            continue
        overlaps = box_iou(xyxy[index], xyxy[index])
        overlaps[np.triu_indices(len(index))] = 0  # 只保留 i>j (即更低分与更高分)
        redundant += int((overlaps > dedup_iou).any(0).sum())
    return len(boxes), redundant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", help="export_passes.py 产出的验证集 JSONL")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--global", dest="global_threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=100)
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
    if arguments.limit:
        images = images[: arguments.limit]
        used = {image["image_id"] for image in images}
        ground_truth = [g for g in ground_truth if g["image_id"] in used]
    print(f"验证图 {len(images)} 张")

    # 变体: (名称, 每分支候选上限, 合并时 NMS 的 IoU)
    variants = [
        ("A production (pre=100, nms=0.70)", 100, 0.70),
        ("B pre=300, nms=0.70", 300, 0.70),
        ("C pre=300, nms=0.60", 300, 0.60),
        ("D pre=300, nms=0.50", 300, 0.50),
        ("E pre=100, nms=0.50", 100, 0.50),
    ]
    results = {}
    for name, pre_cap, nms_iou in variants:
        scorer = make_scorer(images, ground_truth, 12, pre_cap=pre_cap)
        result = scorer({}, arguments.global_threshold, nms_iou, arguments.max_detections)
        results[name] = result
        print(f"{name:34s} map50_95={result['map50_95']:.4f}  map50={result['map50']:.4f}  "
              f"map75={result['map75']:.4f}")

    baseline = results["A production (pre=100, nms=0.70)"]["map50_95"]
    print("\n== 相对 production 的增益")
    for name, result in results.items():
        print(f"  {name:34s} Δ={result['map50_95'] - baseline:+.4f}   预测框总数影响见下")
    print("\n== 最终 100 框中的近似重复名额 (同类别 IoU>0.5 且分更低)")
    for name, pre_cap, nms_iou in [("production", 100, 0.70), ("pre=300,nms=0.70", 300, 0.70)]:
        total = redundant = 0
        for image in images:
            count, bad = duplicate_slots(image, {}, arguments.global_threshold, nms_iou,
                                         arguments.max_detections, pre_cap, 0.5)
            total += count
            redundant += bad
        print(f"  {name:18s} 共 {total} 框, 其中重复浪费 {redundant} 个名额 "
              f"({100.0 * redundant / max(total, 1):.2f}%)")


if __name__ == "__main__":
    main()
