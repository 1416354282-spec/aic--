"""定位「可删错框」的分布: 收益是否集中在人眼可审的低置信度区域。

若 {与任意 GT IoU<0.1} 的框几乎全落在低置信度段, 那么逐图人工审查只需看这些框,
工作量可控; 若它们散布在全置信度范围, 逐框人工纠正就不可行。
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

BANDS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.33), (0.33, 0.5), (0.5, 1.01)]


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
    gt_by_class_image = build_gt_index(ground_truth)
    per_image_gt = defaultdict(list)
    for item in ground_truth:
        per_image_gt[item["image_id"]].append(item["bbox"])

    predictions = []
    for image in images:
        boxes, scores, labels = replay(image, {}, arguments.global_threshold, arguments.nms_iou,
                                       arguments.max_detections)
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            predictions.append({"image_id": image["image_id"], "class_id": int(label),
                                "confidence": float(score), "bbox": box})
    print(f"验证图 {len(images)} 张, 预测 {len(predictions)} 框")

    # 每个框与「任意类别」GT 的最大 IoU
    junk = np.zeros(len(predictions))
    for index, p in enumerate(predictions):
        boxes = per_image_gt.get(p["image_id"], [])
        if not boxes:
            junk[index] = 0.0
            continue
        target = cxcywh_to_xyxy(np.asarray(boxes, np.float32))
        predicted = cxcywh_to_xyxy(np.asarray([p["bbox"]], np.float32))
        junk[index] = float(box_iou(predicted, target)[0].max())
    confidences = np.asarray([p["confidence"] for p in predictions])

    baseline = average_precision(predictions, gt_by_class_image, 12)
    print(f"基线 map50_95 = {baseline['map50_95']:.4f}\n")

    print("== 各置信度段的框数 与 其中的背景错框 (任意类 IoU<0.1)")
    print(f"{'置信度段':>16} {'框数':>7} {'占比':>7} {'IoU<0.1':>8} {'占该段':>8} {'占全部错框':>10}")
    total_junk = int((junk < 0.1).sum())
    for low, high in BANDS:
        mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if not count:
            continue
        band_junk = int(((junk < 0.1) & mask).sum())
        print(f"{low:6.2f}~{high:6.2f} {count:7d} {100.0 * count / len(predictions):6.1f}% "
              f"{band_junk:8d} {100.0 * band_junk / count:7.1f}% {100.0 * band_junk / max(total_junk, 1):9.1f}%")

    print("\n== 「只在低置信度段删错框」能拿到多少收益")
    print(f"{'审查范围':>16} {'删除框数':>9} {'保留率':>8} {'map50_95':>10} {'Δ':>8}")
    for ceiling in [0.02, 0.05, 0.1, 0.2, 0.33, 1.01]:
        keep = ~((junk < 0.1) & (confidences < ceiling))
        kept = [p for p, k in zip(predictions, keep) if k]
        result = average_precision(kept, gt_by_class_image, 12)
        print(f"{'conf < ' + format(ceiling, '.2f'):>16} {len(predictions) - len(kept):9d} "
              f"{100.0 * len(kept) / len(predictions):7.1f}% {result['map50_95']:10.4f} "
              f"{result['map50_95'] - baseline['map50_95']:+8.4f}")

    print("\n== 背景错框按类别 (任意类 IoU<0.1 的框数 / 该类总框数)")
    for cls in range(12):
        index = [i for i, p in enumerate(predictions) if p["class_id"] == cls]
        if not index:
            continue
        bad = sum(1 for i in index if junk[i] < 0.1)
        print(f"  {cls:2d} {CLASS_NAMES[cls]:12s} {bad:6d} / {len(index):6d}  ({100.0 * bad / len(index):5.1f}%)")


if __name__ == "__main__":
    main()
