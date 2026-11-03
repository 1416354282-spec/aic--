#!/usr/bin/env python3
"""在验证集原始预测上自动搜索各类别最优置信度阈值 (数据驱动, 替代手工试)。

闭环流程:
  1. 远程: python scripts/predict_raw.py <checkpoint> --target val --tta \
           --config configs/base.yaml --overlay configs/experiments/rtdetr_x.yaml \
           --output outputs/sweep/val_raw.jsonl
     (checkpoint 必须与生成 x_last_1920_tta 原始测试预测的是同一个, 否则阈值不可迁移)
  2. 本机/远程: python scripts/sweep_thresholds.py outputs/sweep/val_raw.jsonl \
           --config configs/base.yaml --overlay configs/experiments/rtdetr_x.yaml \
           --pin 11=0.8 --output outputs/sweep/thresholds.json
  3. 把搜出的阈值套到原始测试预测上:
     python scripts/apply_class_threshold.py outputs/submissions/x_last_1920_tta \
           --output outputs/submissions/x_last_1920_tta_thr \
           --global 0.001 --threshold 0=0.30 ...  (脚本末尾会直接打印这条命令)

重要说明:
  - 评分用与赛题一致的 mAP@50-95 (IoU 0.50:0.05:0.95 + 101 点插值, 复用 aic_v6.metrics.detection_map)。
  - 阈值语义与管线一致: conf > threshold 才保留。
  - 验证集 GT 太少的类别 (默认 <30 框) 无法可靠优化, 自动跳过并保持 --global;
     tricycle 在 fold_0 验证集只有 1 个 GT, 务必用 --pin 11=<现有值> 固定。
  - 该脚本只动 numpy/排序, CPU 即可跑; predict_raw 需要 GPU, 放远程。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.manifest import read_labels
from aic_v6.engine.trainer import read_ids
from aic_v6.metrics.detection_map import detection_map

COARSE = [0.001, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
          0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def parse_dict(text):
    """'7=0.5,11=0.8' -> {7: 0.5, 11: 0.8} (空字符串 -> {})"""
    result = {}
    for pair in (text or "").split(","):
        if not pair.strip():
            continue
        cls, thr = pair.split("=", 1)
        result[int(cls)] = float(thr)
    return result


def load_raw_jsonl(path):
    predictions = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            image_id = item["image_id"]
            for box, score, label in zip(item["boxes"], item["scores"], item["labels"]):
                predictions.append({
                    "image_id": image_id,
                    "class_id": int(label),
                    "confidence": float(score),
                    "bbox": [float(v) for v in box],
                })
    return predictions


def load_raw_txtdir(path):
    """与 apply_class_threshold.py 同格式的 txt 目录 (每图一个文件)。"""
    predictions = []
    for file in sorted(Path(path).glob("*.txt")):
        for line in file.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 6:
                continue
            cls, cx, cy, w, h, conf = [float(v) for v in fields]
            predictions.append({
                "image_id": file.stem,
                "class_id": int(cls),
                "confidence": conf,
                "bbox": [cx, cy, w, h],
            })
    return predictions


def load_gt(train_root, val_ids):
    ground_truth = []
    for image_id in val_ids:
        labels, _ = read_labels(Path(train_root) / "labels" / f"{image_id}.txt")
        for row in labels:
            ground_truth.append({
                "image_id": image_id,
                "class_id": int(row[0]),
                "bbox": [float(v) for v in row[1:5]],
            })
    return ground_truth


def score(predictions, ground_truth, thresholds, num_classes, global_threshold):
    kept = [p for p in predictions if p["confidence"] > thresholds.get(p["class_id"], global_threshold)]
    return detection_map(kept, ground_truth, num_classes)


def optimize_thresholds(predictions, ground_truth, num_classes, global_threshold,
                        min_gt, pinned, passes=3, verbose=True):
    thresholds = {cls: global_threshold for cls in range(num_classes)}
    gt_counts = Counter(g["class_id"] for g in ground_truth)
    order = sorted(range(num_classes), key=lambda c: -gt_counts[c])
    skipped = [c for c in order if gt_counts[c] < min_gt and c not in pinned]
    order = [c for c in order if c not in skipped and c not in pinned]
    if verbose:
        print(f"GT 统计: {dict(sorted(gt_counts.items()))}")
        print(f"可优化: {order}")
        print(f"跳过(GT<{min_gt}): {skipped}  固定(pin): {sorted(pinned)}")
    baseline = score(predictions, ground_truth, thresholds, num_classes, global_threshold)
    current_map = baseline["map50_95"]

    for pass_index in range(passes):
        for cls in order:
            if pass_index == 0:
                candidates = [t for t in COARSE if t >= global_threshold]
            else:
                center = thresholds[cls]
                candidates = sorted({round(center + offset * 0.02, 3) for offset in range(-6, 7)}
                                    | {center}) 
                candidates = [t for t in candidates if global_threshold <= t <= 0.9]
            best_thr, best_map = thresholds[cls], current_map
            for candidate in candidates:
                thresholds[cls] = candidate
                result = score(predictions, ground_truth, thresholds, num_classes, global_threshold)
                if result["map50_95"] > best_map:
                    best_map, best_thr = result["map50_95"], candidate
            thresholds[cls] = best_thr
            current_map = best_map
            if verbose:
                print(f"  pass{pass_index + 1} cls={cls:2d} thr={best_thr:.3f} map={best_map:.4f}")
    return thresholds, baseline, skipped, gt_counts


def per_class_curve(predictions, ground_truth, cls, global_threshold, num_classes):
    cls_preds = [p for p in predictions if p["class_id"] == cls]
    cls_gt = [g for g in ground_truth if g["class_id"] == cls]
    print(f"== 类别 {cls} AP-阈值曲线 (GT={len(cls_gt)}, 预测={len(cls_preds)})")
    for thr in COARSE:
        kept = [p for p in cls_preds if p["confidence"] > thr]
        result = detection_map(kept, cls_gt, num_classes)
        aps = result["per_class"][cls]
        print(f"  thr={thr:5.3f}  AP={float(np_mean(aps)):.4f}  AP50={aps[0]:.4f}  AP75={aps[5]:.4f}  kept={len(kept)}")


def np_mean(values):
    import numpy as np
    return np.mean(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", help="原始预测: JSONL 文件 或 txt 目录")
    parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--global", dest="global_threshold", type=float, default=0.001)
    parser.add_argument("--min-gt", type=int, default=30,
                        help="验证集 GT 数低于该值的类别不做阈值优化 (避免过拟合)")
    parser.add_argument("--pin", default="", help="固定阈值 '11=0.8,7=0.78' (不参与搜索, 直接采用)")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--class", dest="cls_curve", type=int, default=None, help="只画单类 AP-阈值曲线")
    parser.add_argument("--evaluate-dict", default="", help="不搜索, 只评估给定字典 '0=0.3,2=0.4'")
    parser.add_argument("--output", help="阈值字典保存为 JSON")
    parser.add_argument("--apply-to", help="原始测试预测目录, 用于打印 apply_class_threshold 命令")
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overlay, arguments.set)
    num_classes = int(config["model"]["num_classes"])
    fold = int(config["data"]["fold"])
    val_ids = read_ids(Path(config["data"]["split_dir"]) / f"fold_{fold}_val.txt")

    raw_path = Path(arguments.raw)
    predictions = load_raw_jsonl(raw_path) if raw_path.is_file() else load_raw_txtdir(raw_path)
    ground_truth = load_gt(config["data"]["train_root"], val_ids)
    print(f"原始预测 {len(predictions)} 框, 验证集 {len(val_ids)} 图, GT {len(ground_truth)} 框")

    if arguments.cls_curve is not None:
        per_class_curve(predictions, ground_truth, arguments.cls_curve,
                        arguments.global_threshold, num_classes)
        return

    if arguments.evaluate_dict:
        thresholds = parse_dict(arguments.evaluate_dict)
        result = score(predictions, ground_truth, thresholds, num_classes, arguments.global_threshold)
        print(f"== 给定阈值 {thresholds} 的评估")
        print(f"map50={result['map50']:.4f}  map75={result['map75']:.4f}  map50_95={result['map50_95']:.4f}")
        for cls in range(num_classes):
            aps = result["per_class"][cls]
            import numpy as np
            print(f"  {cls:2d} AP={np.mean(aps):.4f}")
        return

    pinned = parse_dict(arguments.pin)
    thresholds, baseline, skipped, gt_counts = optimize_thresholds(
        predictions, ground_truth, num_classes, arguments.global_threshold,
        arguments.min_gt, pinned, arguments.passes,
    )
    final = score(predictions, ground_truth, thresholds, num_classes, arguments.global_threshold)
    print(f"\n== 结果: 基线(全 {arguments.global_threshold}) map50_95={baseline['map50_95']:.4f} "
          f"-> 优化后 map50_95={final['map50_95']:.4f}  (Δ={final['map50_95'] - baseline['map50_95']:+.4f})")
    print(f"map50: {baseline['map50']:.4f} -> {final['map50']:.4f}   "
          f"map75: {baseline['map75']:.4f} -> {final['map75']:.4f}")
    print("\n== 各类别 (GT数 | 基线AP | 优化后AP | 阈值)")
    import numpy as np
    for cls in range(num_classes):
        thr = thresholds[cls]
        marker = " (pin)" if cls in pinned else (" (skip)" if cls in skipped else "")
        print(f"  {cls:2d} GT={gt_counts[cls]:4d} | {np.mean(baseline['per_class'][cls]):.4f} -> "
              f"{np.mean(final['per_class'][cls]):.4f} | thr={thr:.3f}{marker}")

    non_default = {str(cls): thr for cls, thr in thresholds.items()
                   if thr > arguments.global_threshold or cls in pinned}
    print("\n== class_confidence (可直接 --set validation.class_confidence=...)")
    print(json.dumps(non_default))

    if arguments.output:
        Path(arguments.output).write_text(json.dumps({
            "global": arguments.global_threshold,
            "class_confidence": non_default,
            "map50_95_baseline": baseline["map50_95"],
            "map50_95_optimized": final["map50_95"],
        }, indent=2), encoding="utf-8")
        print(f"已保存: {arguments.output}")

    if arguments.apply_to:
        args = [f"--global {arguments.global_threshold}"]
        for cls in sorted(non_default, key=int):
            args.append(f"--threshold {cls}={non_default[cls]:.3f}")
        print("\n== 套用到原始测试预测的命令 ==")
        print(f"python scripts/apply_class_threshold.py {arguments.apply_to} "
              f"--output {arguments.apply_to}_thr {' '.join(args)} --zip")


if __name__ == "__main__":
    main()
