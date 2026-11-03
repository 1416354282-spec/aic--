"""测量「补框精度 → 收益」的关系: 手画的框要画多准才有意义。

方法: 把漏检的真值框按中心等比缩放, 使其与原真值的 IoU 落在目标值附近
(IoU = 1/s^2, s 为缩放倍数; s>1 表示画大了, 相当于手工圈选时框得松一些),
再补进提交并挤掉最低分框, 测 mAP。这样能得到「平均 IoU=X 时能涨多少」的曲线。
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

from aic_v6.constants import CLASS_NAMES
from aic_v6.data.boxes import box_iou, cxcywh_to_xyxy
from aic_v6.data.manifest import read_labels
from aic_v6.engine.trainer import read_ids
from tune_thresholds import average_precision, build_gt_index, replay

TARGETS = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
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
    gt_index = build_gt_index(ground_truth)

    submitted, predictions = {}, []
    for image in images:
        boxes, scores, labels = replay(image, {}, arguments.global_threshold, arguments.nms_iou,
                                       arguments.max_detections)
        rows = sorted(zip(scores.tolist(), labels.tolist(), boxes.tolist()), key=lambda r: -r[0])
        submitted[image["image_id"]] = rows
    baseline = average_precision([{"image_id": i, "class_id": int(l), "confidence": float(s), "bbox": b}
                                  for i, rows in submitted.items() for s, l, b in rows], gt_index, 12)
    print(f"基线 map50_95 = {baseline['map50_95']:.4f}")

    # 找出漏检真值 (与提交框 IoU<0.5)
    covered = defaultdict(set)
    for cls in range(12):
        per_image_gt = defaultdict(list)
        for index, item in enumerate(ground_truth):
            if item["class_id"] == cls:
                per_image_gt[item["image_id"]].append(index)
        for image_id, rows in submitted.items():
            cls_rows = [(s, b) for s, l, b in rows if int(l) == cls]
            if not cls_rows:
                continue
            candidates = list(per_image_gt.get(image_id, []))
            if not candidates:
                continue
            target = cxcywh_to_xyxy(np.asarray([ground_truth[i]["bbox"] for i in candidates], np.float32))
            predicted = cxcywh_to_xyxy(np.asarray([b for _, b in cls_rows], np.float32))
            overlaps = box_iou(predicted, target)
            for position, index in enumerate(candidates):
                if overlaps[:, position].max() >= 0.5:
                    covered[cls].add(index)
    missed = [item for index, item in enumerate(ground_truth)
              if item["class_id"] not in range(12) or index not in covered[item["class_id"]]]
    print(f"漏检真值 {len(missed)} 个 / 共 {len(ground_truth)} 个\n")

    print(f"{'补框平均IoU':>12} {'缩放倍数':>9} {'map50_95':>10} {'Δ':>9}")
    for target in TARGETS:
        scale = 1.0 / np.sqrt(target)
        added = defaultdict(list)
        for item in missed:
            cx, cy, w, h = item["bbox"]
            added[item["image_id"]].append((item["class_id"], [cx, cy, w * scale, h * scale]))
        augmented = []
        for image_id, rows in submitted.items():
            extra = added.get(image_id, [])
            for score, label, box in rows[: max(len(rows) - len(extra), 0)]:
                augmented.append({"image_id": image_id, "class_id": int(label),
                                  "confidence": float(score), "bbox": box})
            for cls, box in extra:
                augmented.append({"image_id": image_id, "class_id": cls, "confidence": 0.999, "bbox": box})
        # 实际达成的平均 IoU (验证缩放模型)
        achieved = []
        for item in missed:
            cx, cy, w, h = item["bbox"]
            a = cxcywh_to_xyxy(np.asarray([[cx, cy, w, h]], np.float32))
            b = cxcywh_to_xyxy(np.asarray([[cx, cy, w * scale, h * scale]], np.float32))
            achieved.append(float(box_iou(a, b)[0][0]))
        result = average_precision(augmented, gt_index, 12)
        print(f"{np.mean(achieved):>12.3f} {scale:>9.3f} {result['map50_95']:>10.4f} "
              f"{result['map50_95'] - baseline['map50_95']:>+9.4f}")


if __name__ == "__main__":
    main()
