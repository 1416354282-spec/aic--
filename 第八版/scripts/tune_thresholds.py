"""在验证集上搜索各类别最优置信度阈值 (基于 export_passes.py 的全量候选)。

方法论:
  production 后处理 = 「按类别阈值筛 -> 分数降序 -> 取 top-100」, 其中 TTA 的两个分支
  各自截断到 100 后再 NMS 合并 -> 再取 top-100。本脚本用 torch 逐行镜像
  engine/evaluator.predict_batch_tta 的这段逻辑 (阈值可注入), 因此任何阈值组合都能
  被精确重放, 无需重新跑 GPU 前向。

  mAP 计算与 aic_v8.metrics.detection_map 完全等价 (同一贪心匹配 + 101 点插值),
  但把 10 个 IoU 阈值并行处理, 使得单次评估从 ~秒级降到可接受范围。
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

from aic_v8.data.boxes import box_iou, cxcywh_to_xyxy
from aic_v8.data.manifest import read_labels
from aic_v8.engine.evaluator import _prediction_to_original
from aic_v8.engine.trainer import read_ids

IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_POINTS = np.linspace(0, 1, 101)
GRID = [0.001, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Meta:
    pass


def threshold_vector(labels, class_thresholds, global_threshold):
    return torch.tensor([class_thresholds.get(int(c), global_threshold) for c in labels])


def replay(image, class_thresholds, global_threshold, nms_iou, max_detections, pre_cap=None):
    """精确镜像 predict_batch_tta 的后处理, 返回原图归一化 cxcywh 的检测。

    pre_cap: 每个 TTA 分支进入 NMS 合并前的候选上限 (production 用 max_detections=100,
    即各分支先截断再合并)。把它放大可让最终 100 个名额从更大的候选池里挑, 去掉
    近似重复框腾出的名额就能分给第 101+ 个不同目标。
    """
    pre_cap = max_detections if pre_cap is None else pre_cap
    kept = []
    for pass_index, source in enumerate(image["passes"]):
        boxes = torch.tensor(source["boxes"], dtype=torch.float32)
        scores = torch.tensor(source["scores"], dtype=torch.float32)
        labels = torch.tensor(source["labels"], dtype=torch.float32)
        if not len(scores):
            kept.append((boxes, scores, labels))
            continue
        valid = scores > threshold_vector(labels, class_thresholds, global_threshold)
        score_i, label_i, box_i = scores[valid], labels[valid], boxes[valid]
        if len(score_i):
            order = score_i.argsort(descending=True)[:pre_cap]
            kept.append((box_i[order], score_i[order], label_i[order]))
        else:
            kept.append((box_i[:0], score_i[:0], label_i[:0]))
    original, flipped = kept
    if len(flipped[0]):
        # 与 predict_batch_tta 一致: 翻转分支的 cx 镜像回原空间后再合并
        mirrored = flipped[0].clone()
        mirrored[:, 0] = 1 - mirrored[:, 0]
        flipped = (mirrored, flipped[1], flipped[2])
    if len(original[0]) and len(flipped[0]):
        boxes = torch.cat([original[0], flipped[0]])
        scores = torch.cat([original[1], flipped[1]])
        labels = torch.cat([original[2], flipped[2]])
        order = scores.argsort(descending=True)[:30000]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        from torchvision.ops import batched_nms

        xyxy = torch.stack((boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                            boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2), dim=1)
        keep = batched_nms(xyxy, scores, labels, nms_iou)[:max_detections]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    else:
        boxes, scores, labels = original if len(original[0]) else flipped
    meta = image["meta"]
    handle = Meta()
    handle.original_hw = tuple(meta["original_hw"])
    handle.output_hw = tuple(meta["output_hw"])
    handle.ratio = float(meta["ratio"])
    handle.pad_xy = tuple(meta["pad_xy"])
    if len(boxes):
        boxes = torch.stack([_prediction_to_original(box, handle) for box in boxes])
    return boxes, scores, labels


def build_gt_index(ground_truth):
    index = {}
    for item in ground_truth:
        index.setdefault((item["class_id"], item["image_id"]), []).append(item["bbox"])
    return index


def average_precision(predictions, gt_by_class_image, num_classes, cache=None):
    """与 aic_v8.metrics.detection_map 等价的 AP (10 个 IoU 阈值并行处理)。"""
    cache = {} if cache is None else cache
    per_class = {}
    for cls in range(num_classes):
        items = [p for p in predictions if p["class_id"] == cls]
        items.sort(key=lambda p: p["confidence"], reverse=True)
        count = len(items)
        true_positive = np.zeros((10, count))
        false_positive = np.zeros((10, count))
        if count:
            lookup = []
            for p in items:
                key = (cls, p["image_id"])
                boxes = gt_by_class_image.get(key, [])
                if boxes:
                    if key not in cache:
                        cache[key] = cxcywh_to_xyxy(np.asarray(boxes, np.float32))
                    lookup.append((p["image_id"], len(boxes), cache[key]))
                else:
                    lookup.append((p["image_id"], 0, None))
            # 逐图顺序处理, 10 个 IoU 阈值并行
            rows = defaultdict(list)
            for position, (image_id, _, _) in enumerate(lookup):
                rows[image_id].append(position)
            for image_id, positions in rows.items():
                target = lookup[positions[0]][2]
                if target is None:
                    false_positive[:, positions] = 1
                    continue
                local = np.zeros((10, len(target)), bool)
                predicted = cxcywh_to_xyxy(np.asarray([items[p]["bbox"] for p in positions], np.float32))
                overlaps = box_iou(predicted, target)
                for row, position in enumerate(positions):
                    available = np.where(~local, overlaps[row][None, :], -1.0)
                    best = available.argmax(1)
                    best_value = available[np.arange(10), best]
                    hit = best_value >= IOU_THRESHOLDS
                    true_positive[hit, position] = 1
                    false_positive[~hit, position] = 1
                    local[np.arange(10)[hit], best[hit]] = True
        truth_count = sum(len(v) for (c, _), v in gt_by_class_image.items() if c == cls)
        cumulative_tp = true_positive.cumsum(1)
        cumulative_all = (true_positive + false_positive).cumsum(1)
        recall = cumulative_tp / max(truth_count, 1)
        precision = cumulative_tp / np.maximum(cumulative_all, 1e-12)
        average_precisions = []
        for row in range(10):
            suffix = np.maximum.accumulate(precision[row][::-1])[::-1]
            values = []
            for point in RECALL_POINTS:
                index = np.searchsorted(recall[row], point, side="left")
                values.append(suffix[index] if index < len(suffix) else 0.0)
            average_precisions.append(float(np.mean(values)))
        per_class[cls] = average_precisions
    values = list(per_class.values())
    return {
        "map50": float(np.mean([v[0] for v in values])),
        "map75": float(np.mean([v[5] for v in values])),
        "map50_95": float(np.mean([np.mean(v) for v in values])),
        "per_class": per_class,
    }


def make_scorer(images, ground_truth, num_classes, pre_cap=None):
    """预计算 GT 索引; 返回 score(class_thresholds, global_threshold, nms_iou, max_detections) -> dict。"""
    gt_by_class_image = build_gt_index(ground_truth)
    cache = {}

    def score(class_thresholds, global_threshold, nms_iou, max_detections):
        predictions = []
        for image in images:
            boxes, scores, labels = replay(image, class_thresholds, global_threshold, nms_iou, max_detections, pre_cap)
            for box, value, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                predictions.append({"image_id": image["image_id"], "class_id": int(label),
                                    "confidence": float(value), "bbox": box})
        return average_precision(predictions, gt_by_class_image, num_classes, cache)

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", help="export_passes.py 产出的 JSONL")
    parser.add_argument("--train-root", required=True, help="含 labels/ 的训练集根目录 (读 GT)")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--global", dest="global_threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 张搜索")
    parser.add_argument("--min-gt", type=int, default=30, help="GT 少于该值的类别不优化")
    parser.add_argument("--pin", default="", help="固定阈值 '11=0.8'")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--global-sweep", action="store_true", help="先扫全局阈值")
    parser.add_argument("--output", help="阈值 JSON 输出")
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
    gt_counts = defaultdict(int)
    for item in ground_truth:
        gt_counts[item["class_id"]] += 1
    print(f"验证图 {len(images)} 张, GT {len(ground_truth)} 框, GT 分布 {dict(sorted(gt_counts.items()))}")

    score = make_scorer(images, ground_truth, arguments.num_classes)
    baseline = score({}, arguments.global_threshold, arguments.nms_iou, arguments.max_detections)
    print(f"\n基线 (全部 {arguments.global_threshold}): map50_95={baseline['map50_95']:.4f} "
          f"map50={baseline['map50']:.4f}")

    if arguments.global_sweep:
        print("\n== 全局阈值扫描")
        for candidate in GRID:
            result = score({}, candidate, arguments.nms_iou, arguments.max_detections)
            print(f"  全局 {candidate:5.3f} -> map50_95={result['map50_95']:.4f}  map50={result['map50']:.4f}")

    pinned = {}
    for pair in (arguments.pin or "").split(","):
        if pair.strip():
            cls, value = pair.split("=", 1)
            pinned[int(cls)] = float(value)
    thresholds = {cls: arguments.global_threshold for cls in range(arguments.num_classes)}
    thresholds.update(pinned)
    current = baseline["map50_95"]
    optimizable = [c for c in sorted(range(arguments.num_classes), key=lambda c: -gt_counts[c])
                   if gt_counts[c] >= arguments.min_gt and c not in pinned]
    print(f"\n可优化类别: {optimizable}")

    for pass_index in range(arguments.passes):
        for cls in optimizable:
            best_threshold, best_map = thresholds[cls], current
            for candidate in GRID:
                thresholds[cls] = candidate
                result = score(thresholds, arguments.global_threshold, arguments.nms_iou, arguments.max_detections)
                if result["map50_95"] > best_map:
                    best_map, best_threshold = result["map50_95"], candidate
            thresholds[cls] = best_threshold
            current = best_map
            print(f"  pass{pass_index + 1} cls={cls:2d} thr={best_threshold:5.3f} map={best_map:.4f}", flush=True)

    final = score(thresholds, arguments.global_threshold, arguments.nms_iou, arguments.max_detections)
    print(f"\n== 基线 map50_95={baseline['map50_95']:.4f} -> 优化后 {final['map50_95']:.4f} "
          f"(Δ={final['map50_95'] - baseline['map50_95']:+.4f})")
    print(f"   map50: {baseline['map50']:.4f} -> {final['map50']:.4f}")
    print("\n== 各类别 (GT | 基线AP | 优化AP | 阈值)")
    for cls in range(arguments.num_classes):
        print(f"  {cls:2d} GT={gt_counts[cls]:4d} | {np.mean(baseline['per_class'][cls]):.4f} -> "
              f"{np.mean(final['per_class'][cls]):.4f} | thr={thresholds[cls]:.3f}")
    print("\n== class_confidence")
    print(json.dumps({str(c): t for c, t in sorted(thresholds.items()) if t > arguments.global_threshold}))
    if arguments.output:
        Path(arguments.output).write_text(json.dumps({
            "global": arguments.global_threshold,
            "class_confidence": {str(c): t for c, t in sorted(thresholds.items()) if t > arguments.global_threshold},
            "map50_95_baseline": baseline["map50_95"],
            "map50_95_optimized": final["map50_95"],
            "val_images": len(images),
        }, indent=2), encoding="utf-8")
        print(f"已保存 {arguments.output}")


if __name__ == "__main__":
    main()
