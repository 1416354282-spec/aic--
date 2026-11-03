#!/usr/bin/env python3
"""按全局置信度精确截取前 N 个框, 再写成提交目录 (每图一个同名 txt)。

与 filter_by_budget.py 的区别: 那里用 "分数 >= 第 N 高分" 保留, 并列分数会让总数
略超 N; 这里按 (score, image_id, 行号) 稳定排序后取前 N 行, 总数恰好等于预算。

可选 --nms-iou: 先对每张图的同类框做分数降序贪心去重 (IoU > 阈值则抑制), 再做
全局 Top-N。TTA 的翻转分支在 NMS 0.7 下会留下 IoU 0.5~0.7 的镜像重复框, 这一层
去重可以去掉它们, 代价是同图同类真实重叠目标 (IoU>0.5) 会被抑制。

输入是 TTA 稠密产物 (conf>=0.001, max_detections=100), 只做行级筛选, 不重新推理。

用法:
  python scripts/top_budget.py <src_dir> <dst_dir> <budget> \
      [--nms-iou 0.5] [--zip] [--report <json>]
"""
import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.constants import CLASS_NAMES


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def dedup_same_class(items, nms_iou):
    """items: [(score, cls, box(cxcywh), line, index)] → 同类分数降序贪心去重。"""
    grouped = defaultdict(list)
    for item in items:
        grouped[item[1]].append(item)
    kept = []
    for group in grouped.values():
        group.sort(key=lambda row: (-row[0], row[4]))
        kept_group = []          # (box_xyxy,)
        for score, cls, (cx, cy, w, h), line, index in group:
            xyxy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if any(iou_xyxy(xyxy, other) > nms_iou for other in kept_group):
                continue
            kept_group.append(xyxy)
            kept.append((score, cls, (cx, cy, w, h), line, index))
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("budget", type=int)
    parser.add_argument("--nms-iou", type=float, default=None,
                        help="可选: 先做同类贪心去重的 IoU 阈值 (如 0.5)")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--report", default=None, help="统计报告 JSON 路径 (默认放在 dst 同级)")
    arguments = parser.parse_args()

    src = Path(arguments.src)
    dst = Path(arguments.dst)
    if arguments.budget <= 0:
        raise SystemExit("budget 必须为正数")

    files = sorted(src.glob("*.txt"))
    if not files:
        raise SystemExit(f"{src} 下没有 txt 文件")

    raw_total = 0
    survivors = []           # (score, cls, box, line, index, image_id)
    for path in files:
        items = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{path.name} 第 {index + 1} 行列数错误: {line!r}")
            cls = int(float(fields[0]))
            box = tuple(float(v) for v in fields[1:5])
            items.append((float(fields[5]), cls, box, line, index))
        raw_total += len(items)
        if arguments.nms_iou is not None:
            items = dedup_same_class(items, arguments.nms_iou)
        survivors.extend((score, cls, box, line, index, path.stem)
                         for score, cls, box, line, index in items)

    after_dedup = len(survivors)
    budget = min(arguments.budget, after_dedup)
    survivors.sort(key=lambda row: (-row[0], row[5], row[4]))
    kept = survivors[:budget]
    threshold = kept[-1][0] if kept else 0.0

    grouped = defaultdict(list)
    for score, cls, box, line, index, image_id in kept:
        grouped[image_id].append((score, line))

    dst.mkdir(parents=True, exist_ok=True)
    counts, classes = [], Counter()
    for path in files:
        items = sorted(grouped.get(path.stem, []), key=lambda item: -item[0])
        counts.append(len(items))
        for _, line in items:
            classes[int(float(line.split()[0]))] += 1
        (dst / path.name).write_text("\n".join(line for _, line in items), encoding="utf-8")

    kept_total = sum(counts)
    report = {
        "source": str(src),
        "destination": str(dst),
        "raw_total": raw_total,
        "nms_iou": arguments.nms_iou,
        "after_dedup": after_dedup,
        "dedup_removed": raw_total - after_dedup,
        "budget": arguments.budget,
        "kept_total": kept_total,
        "score_threshold": threshold,
        "per_image": {
            "min": min(counts),
            "max": max(counts),
            "mean": kept_total / len(files),
            "empty": sum(1 for count in counts if count == 0),
        },
        "class_counts": {name: classes.get(index, 0) for index, name in enumerate(CLASS_NAMES)},
    }

    print(f"原始总框数 {raw_total}")
    if arguments.nms_iou is not None:
        print(f"同类去重 IoU>{arguments.nms_iou}: 删除 {report['dedup_removed']} 框 -> 剩余 {after_dedup}")
    print(f"保留 {kept_total} (预算 {arguments.budget})")
    print(f"保留最低分 {threshold:.8f}")
    print(f"每图框数   : {min(counts)} ~ {max(counts)}, 平均 {kept_total / len(files):.2f}, "
          f"空图 {report['per_image']['empty']}")
    print("类别分布:")
    for index, name in enumerate(CLASS_NAMES):
        count = classes.get(index, 0)
        print(f"  {index:2d} {name:11s} {count:6d}  {100.0 * count / max(kept_total, 1):5.1f}%")

    report_path = Path(arguments.report) if arguments.report else dst.parent / f"{dst.name}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: {report_path}")

    if arguments.zip:
        zip_path = dst.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dst.glob("*.txt")):
                archive.write(path, path.name)
        print(f"打包: {zip_path}")


if __name__ == "__main__":
    main()
