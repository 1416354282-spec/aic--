#!/usr/bin/env python3
"""对已有提交做「同类去重 + 按类阈值」后处理 (无需重新推理)。

依据 (2026-10-03 诊断):
  1. TTA 两分支在 NMS 0.7 下会留下 IoU 0.5~0.7 的镜像重复框 (seat 15 对 / light 12 对 /
     garbage can 15 对, 用户目视看到的"重叠很高"即此类)。而 new_labels_2000 里同类 GT 框
     IoU>0.5 的对子极少 (person 23 / animal 24 / seat 4 / bicycle 5 / light 1, 其余 0),
     两个 IoU>0.5 的预测几乎不可能同时是 TP —— 对全部类别做同类 IoU 0.5 贪心去重,
     最坏只损失 ~23 个 TP, 换掉 ~250 个重复 FP。
  2. garbage can 在 conf 0.2~0.5 段目视误检率 ~65% (29 FP vs 16 TP), 阈值提到 0.5
     可去掉这批最扎眼的误检; 该段处于排序尾部, 对 mAP 影响很小。

用法:
  python scripts/dedup_and_threshold.py <提交目录> --output <新目录> \
      [--nms-iou 0.5] [--threshold 9=0.5] [--global 0.0] [--zip]
"""
import argparse
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def greedy_nms(rows, nms_iou):
    """rows: [(conf, line, box)], box=cxcywh 归一化。按分数降序同类贪心抑制。"""
    rows = sorted(rows, key=lambda r: -r[0])
    kept = []
    for conf, line, box in rows:
        cx, cy, w, h = box
        xyxy = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        if any(iou_xyxy(xyxy, k[3]) > nms_iou for k in kept):
            continue
        kept.append((conf, line, box, xyxy))
    return [(conf, line) for conf, line, _, _ in kept]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nms-iou", type=float, default=0.5, help="同类去重 IoU 阈值")
    parser.add_argument("--threshold", action="append", default=[], metavar="CLS=THR")
    parser.add_argument("--global", dest="global_threshold", type=float, default=0.0,
                        help="未单独指定的类别的置信度下限 (0.0=全保留)")
    parser.add_argument("--zip", action="store_true")
    arguments = parser.parse_args()

    thresholds = {}
    for pair in arguments.threshold:
        cls, thr = pair.split("=", 1)
        thresholds[int(cls)] = float(thr)

    input_dir = Path(arguments.input)
    output_dir = Path(arguments.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before = Counter()
    after = Counter()
    nms_removed = Counter()
    thr_removed = Counter()
    for path in sorted(input_dir.glob("*.txt")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) != 6:
                continue
            cls, conf = int(float(f[0])), float(f[5])
            before[cls] += 1
            if conf <= thresholds.get(cls, arguments.global_threshold):
                thr_removed[cls] += 1
                continue
            box = [float(v) for v in f[1:5]]
            rows.append((conf, line, (cls, box)))
        by_class = defaultdict(list)
        for conf, line, (cls, box) in rows:
            by_class[cls].append((conf, line, box))
        lines = []
        for cls, items in by_class.items():
            kept = greedy_nms(items, arguments.nms_iou)
            nms_removed[cls] += len(items) - len(kept)
            after[cls] += len(kept)
            lines += [line for _, line in kept]
        (output_dir / path.name).write_text("\n".join(lines), encoding="utf-8")

    names = ["person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
             "light", "garbage can", "uav", "tricycle"]
    print(f"输入 {input_dir.name}: {sum(before.values())} 框 -> 输出 {sum(after.values())} 框")
    print(f"{'class':>13} {'原':>6} {'阈值删':>7} {'去重删':>7} {'保留':>6}")
    for cls in range(12):
        if before[cls] or after[cls]:
            print(f"{names[cls]:>13} {before[cls]:6d} {thr_removed[cls]:7d} "
                  f"{nms_removed[cls]:7d} {after[cls]:6d}")
    print(f"去重 IoU={arguments.nms_iou}, 类阈值={thresholds or '无'}")
    print(f"输出: {output_dir}")

    if arguments.zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.glob("*.txt")):
                archive.write(path, path.name)
        print(f"打包: {zip_path}")


if __name__ == "__main__":
    main()
