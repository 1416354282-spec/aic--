"""量化「逐图手工纠正检测框」的收益上限。

原理: 删掉一个 FP 会让它之后所有框的精度升高 (召回不变), AP 上的「最大精度包络」
因此可能上升 —— 所以精准删错框理论上弱增; 删掉 TP 则必然掉分。全局阈值之所以掉分,
是因为它同时删掉了 TP。

本脚本用「上帝视角」在验证集上测三种纠正强度的上限:
  1. 完美纠正: 只保留与同类 GT 匹配 (IoU>=0.5) 的框 —— 收益天花板
  2. 明显错框全删: 删掉与任意 GT 的 IoU 都 < 0.3 的框 (人工肉眼能判定的那种)
  3. 宽松错框全删: 删掉与任意 GT 的 IoU 都 < 0.5 的框

真实手工纠正的效果一定低于 (1), 大致落在 (2) 附近。若 (2) 的增益都接近 0,
说明逐图纠正不值得投入。
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

from aic_v6.data.boxes import box_iou, cxcywh_to_xyxy
from aic_v6.data.manifest import read_labels
from aic_v6.engine.trainer import read_ids
from tune_thresholds import average_precision, build_gt_index, make_scorer, replay


def max_iou_to_any_gt(predictions, ground_truth_by_image):
    """每个预测框与「任意类别」GT 的最大 IoU (纠正时人眼并不知道类别是否对)。"""
    result = []
    for p in predictions:
        boxes = ground_truth_by_image.get(p["image_id"], [])
        if not boxes:
            result.append(0.0)
            continue
        target = cxcywh_to_xyxy(np.asarray(boxes, np.float32))
        predicted = cxcywh_to_xyxy(np.asarray([p["bbox"]], np.float32))
        result.append(float(box_iou(predicted, target)[0].max()))
    return result


def max_iou_same_class(predictions, gt_by_class_image):
    result = []
    for p in predictions:
        boxes = gt_by_class_image.get((p["class_id"], p["image_id"]), [])
        if not boxes:
            result.append(0.0)
            continue
        target = cxcywh_to_xyxy(np.asarray(boxes, np.float32))
        predicted = cxcywh_to_xyxy(np.asarray([p["bbox"]], np.float32))
        result.append(float(box_iou(predicted, target)[0].max()))
    return result


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
    print(f"验证图 {len(images)} 张, GT {len(ground_truth)} 框")

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
    print(f"基线预测 {len(predictions)} 框")

    baseline = average_precision(predictions, gt_by_class_image, 12)
    print(f"\n基线 map50_95 = {baseline['map50_95']:.4f}  (预测 {len(predictions)} 框)\n")

    same_class = max_iou_same_class(predictions, gt_by_class_image)
    any_class = max_iou_to_any_gt(predictions, per_image_gt)

    for name, keep_mask in [
        ("1. 完美纠正 (同类 IoU>=0.5 才留)", np.asarray(same_class) >= 0.5),
        ("2. 删明显错框 (任意类 IoU<0.3 全删)", np.asarray(any_class) >= 0.3),
        ("3. 删较明显错框 (任意类 IoU<0.5 全删)", np.asarray(any_class) >= 0.5),
        ("4. 删最明显错框 (任意类 IoU<0.1 全删)", np.asarray(any_class) >= 0.1),
    ]:
        kept = [p for p, keep in zip(predictions, keep_mask) if keep]
        result = average_precision(kept, gt_by_class_image, 12)
        print(f"{name:38s} 保留 {len(kept):6d} 框 ({100.0 * len(kept) / len(predictions):5.1f}%)  "
              f"map50_95={result['map50_95']:.4f}  Δ={result['map50_95'] - baseline['map50_95']:+.4f}")


if __name__ == "__main__":
    main()
