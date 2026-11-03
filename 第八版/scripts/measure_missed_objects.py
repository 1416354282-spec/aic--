"""测量「补充框」的收益空间: 模型漏掉了多少真值物体, 以及手工补框的上限。

两个问题分开回答:
  1. 召回: 已提交的框覆盖了多少真值物体 (IoU>=0.5)? 按类别拆开看漏检率。
  2. 上限: 假设把漏掉的真值框「完美补上」(位置精确、高置信度), 并把每图最低分的
     若干框挤掉以遵守 100 框上限, mAP 能涨多少 —— 这是人工补框的天花板。
     现实中手画框达不到真值精度, 所以实际收益会明显低于此值。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aic_v8.constants import CLASS_NAMES
from aic_v8.data.boxes import box_iou, cxcywh_to_xyxy
from aic_v8.data.manifest import read_labels
from aic_v8.engine.trainer import read_ids
from tune_thresholds import average_precision, build_gt_index, replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--global", dest="global_threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
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
    gt_index = build_gt_index(ground_truth)

    # 每图提交框 (保持分数降序)
    submitted = {}
    predictions = []
    for image in images:
        boxes, scores, labels = replay(image, {}, arguments.global_threshold, arguments.nms_iou,
                                       arguments.max_detections)
        rows = sorted(zip(scores.tolist(), labels.tolist(), boxes.tolist()), key=lambda r: -r[0])
        submitted[image["image_id"]] = rows
        for score, label, box in rows:
            predictions.append({"image_id": image["image_id"], "class_id": int(label),
                                "confidence": float(score), "bbox": box})
    baseline = average_precision(predictions, gt_index, 12)
    print(f"验证图 {len(images)} 张, 提交框 {len(predictions)} 个, 真值 {len(ground_truth)} 个")
    print(f"基线 map50_95 = {baseline['map50_95']:.4f}\n")

    # 贪心匹配: 每类按置信度降序, IoU>=0.5 视为已覆盖
    covered = defaultdict(set)
    for cls in range(12):
        cls_preds = sorted([p for p in predictions if p["class_id"] == cls],
                           key=lambda p: -p["confidence"])
        per_image_gt = {}
        for index, item in enumerate(ground_truth):
            if item["class_id"] == cls:
                per_image_gt.setdefault(item["image_id"], []).append(index)
        for p in cls_preds:
            candidates = [i for i in per_image_gt.get(p["image_id"], []) if i not in covered[cls]]
            if not candidates:
                continue
            target = cxcywh_to_xyxy(np.asarray([ground_truth[i]["bbox"] for i in candidates], np.float32))
            predicted = cxcywh_to_xyxy(np.asarray([p["bbox"]], np.float32))
            overlaps = box_iou(predicted, target)[0]
            best = int(overlaps.argmax())
            if overlaps[best] >= 0.5:
                covered[cls].add(candidates[best])

    print("== 召回: 真值中被提交框覆盖 (IoU>=0.5) 的比例")
    print(f"{'类别':13s} {'真值数':>7} {'已覆盖':>7} {'漏检':>7} {'漏检率':>8}")
    missed = []
    for cls in range(12):
        total = sum(1 for g in ground_truth if g["class_id"] == cls)
        hit = len(covered[cls])
        for index, item in enumerate(ground_truth):
            if item["class_id"] == cls and index not in covered[cls]:
                missed.append(item)
        print(f"{CLASS_NAMES[cls]:13s} {total:7d} {hit:7d} {total - hit:7d} {100.0 * (total - hit) / max(total, 1):7.1f}%")
    total_gt = len(ground_truth)
    print(f"{'合计':13s} {total_gt:7d} {total_gt - len(missed):7d} {len(missed):7d} "
          f"{100.0 * len(missed) / total_gt:7.1f}%")

    # 上限: 把漏检真值框以高置信度补上, 挤掉每图最低分的等量框
    added = defaultdict(list)
    for item in missed:
        added[item["image_id"]].append(item)
    augmented = []
    for image_id, rows in submitted.items():
        extra = added.get(image_id, [])
        kept = rows[: max(len(rows) - len(extra), 0)]  # 挤掉最低分的等量框
        for score, label, box in kept:
            augmented.append({"image_id": image_id, "class_id": int(label),
                              "confidence": float(score), "bbox": box})
        for item in extra:
            augmented.append({"image_id": image_id, "class_id": item["class_id"],
                              "confidence": 0.999, "bbox": item["bbox"]})
    ceiling = average_precision(augmented, gt_index, 12)
    print(f"\n== 完美补框上限 (位置=真值, 置信度=0.999, 挤掉最低分框)")
    print(f"补框 {len(missed)} 个 -> map50_95 = {ceiling['map50_95']:.4f}  "
          f"Δ = {ceiling['map50_95'] - baseline['map50_95']:+.4f}")
    print("\n== 逐类 AP 变化")
    for cls in range(12):
        a = np.mean(baseline["per_class"][cls])
        b = np.mean(ceiling["per_class"][cls])
        print(f"  {cls:2d} {CLASS_NAMES[cls]:12s} {a:.4f} -> {b:.4f}  ({b - a:+.4f})")


if __name__ == "__main__":
    main()
