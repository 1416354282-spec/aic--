"""逐图审计全部预测: 找出异常图 (退化输出/类别异常/空场景/重复框过多), 输出 CSV。

用法:
  python scripts/audit_predictions.py <预测txt目录> --csv <输出csv>
"""
import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.constants import CLASS_NAMES
from aic_v6.data.boxes import box_iou, cxcywh_to_xyxy


def load(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        rows.append(tuple(float(v) for v in fields))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--top", type=int, default=12, help="打印前 N 张最可疑的图")
    arguments = parser.parse_args()

    directory = Path(arguments.predictions)
    files = sorted(directory.glob("*.txt"))
    records = []
    class_totals = Counter()
    for path in files:
        rows = load(path)
        labels = [int(r[0]) for r in rows]
        scores = [r[5] for r in rows]
        counts = Counter(labels)
        distinct_classes = len(counts)
        top_score = max(scores) if scores else 0.0
        above_33 = sum(1 for s in scores if s > 0.33)
        above_05 = sum(1 for s in scores if s > 0.5)
        # 同类别近似重复 (IoU>0.5 且分数更低) 浪费的名额
        redundant = 0
        if rows:
            boxes = np.asarray([[r[1], r[2], r[3], r[4]] for r in rows], np.float32)
            xyxy = cxcywh_to_xyxy(boxes)
            label_array = np.asarray(labels)
            for label in set(labels):
                index = np.nonzero(label_array == label)[0]
                if len(index) < 2:
                    continue
                overlaps = box_iou(xyxy[index], xyxy[index])
                overlaps[np.triu_indices(len(index))] = 0
                redundant += int((overlaps > 0.5).any(0).sum())
        class_totals.update(counts)
        records.append({
            "image_id": path.stem, "boxes": len(rows), "distinct_classes": distinct_classes,
            "top_score": round(top_score, 5), "above_0.33": above_33, "above_0.50": above_05,
            "redundant": redundant, "redundant_ratio": round(redundant / max(len(rows), 1), 3),
            "dominant_class": CLASS_NAMES[counts.most_common(1)[0][0]] if counts else "",
            "dominant_share": round(counts.most_common(1)[0][1] / max(len(rows), 1), 3) if counts else 0.0,
        })

    with open(arguments.csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    boxes_stat = [r["boxes"] for r in records]
    print(f"审计 {len(records)} 张图 -> {arguments.csv}")
    print(f"每图框数: min={min(boxes_stat)} 中位={statistics.median(boxes_stat)} max={max(boxes_stat)}")
    print(f"满分(100框)的图: {sum(1 for b in boxes_stat if b == 100)} 张")
    print(f"零框的图: {sum(1 for b in boxes_stat if b == 0)} 张")
    print(f"最高置信度<0.33 的图 (模型几乎没找到东西): {sum(1 for r in records if r['top_score'] < 0.33)} 张")
    print(f"最高置信度<0.1 的图: {sum(1 for r in records if r['top_score'] < 0.1)} 张")
    print(f"可区分类别数<3 的图: {sum(1 for r in records if r['distinct_classes'] < 3)} 张")
    print(f"近似重复名额占比: 中位 {statistics.median(r['redundant_ratio'] for r in records):.3f} "
          f"最大 {max(r['redundant_ratio'] for r in records):.3f}")
    print("\n预测框类别总数:")
    for cls in range(12):
        print(f"  {cls:2d} {CLASS_NAMES[cls]:12s} {class_totals[cls]:6d}")
    print(f"\n== 最可疑的 {arguments.top} 张 (按 最高置信度 升序, 即模型最没把握的)")
    for record in sorted(records, key=lambda r: r["top_score"])[: arguments.top]:
        print(f"  {record['image_id']:28s} 框={record['boxes']:3d} 最高分={record['top_score']:.4f} "
              f"类别数={record['distinct_classes']:2d} 主类={record['dominant_class']}({record['dominant_share']})")
    print(f"\n== 重复名额最多的 {arguments.top} 张")
    for record in sorted(records, key=lambda r: -r["redundant_ratio"])[: arguments.top]:
        print(f"  {record['image_id']:28s} 框={record['boxes']:3d} 重复={record['redundant']:3d} "
              f"({record['redundant_ratio']}) 最高分={record['top_score']:.4f}")


if __name__ == "__main__":
    main()
