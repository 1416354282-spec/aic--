#!/usr/bin/env python3
"""对已有提交结果按类别做额外置信度阈值过滤 (后处理, 无需重新推理)。

用法:
  python scripts/apply_class_threshold.py <提交目录> --output <新目录> \
      --threshold 7=0.5 --threshold 11=0.5 --global 0.001

说明:
  - 指定了 --threshold 的类别完全用该阈值 (可低于全局), 未指定的类别用全局阈值,
    与管线级 evaluator._class_thresholds 语义一致。
  - 不带 --threshold 时只打印各类别框数/置信度分布, 便于定阈值。
  - 也可配合 --zip 输出压缩包。
"""
import argparse
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]


def parse_thresholds(pairs):
    result = {}
    for pair in pairs or []:
        cls, thr = pair.split("=", 1)
        result[int(cls)] = float(thr)
    return result


def read_submission(directory):
    per_class = defaultdict(list)
    for path in sorted(Path(directory).glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 6:
                continue
            cls, conf = int(float(fields[0])), float(fields[5])
            per_class[cls].append((path.stem, line))
    return per_class


def print_stats(per_class, title, total_images):
    print(f"== {title} (共 {sum(len(v) for v in per_class.values())} 框 / {total_images} 图)")
    for cls in range(len(CLASS_NAMES)):
        entries = per_class.get(cls, [])
        if not entries:
            print(f"  {cls:2d} {CLASS_NAMES[cls]:12s}: 0 框")
            continue
        confs = sorted(float(line.split()[5]) for _, line in entries)
        quantile = lambda p: confs[min(len(confs) - 1, int(p * len(confs)))]
        print(f"  {cls:2d} {CLASS_NAMES[cls]:12s}: {len(entries):4d} 框  "
              f"min={confs[0]:.2f} p25={quantile(.25):.2f} p50={quantile(.50):.2f} "
              f"p75={quantile(.75):.2f} max={confs[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="提交目录 (每图一个 txt)")
    parser.add_argument("--output", help="过滤后输出目录 (缺省只打印统计)")
    parser.add_argument("--threshold", action="append", default=[], metavar="CLS=THR",
                        help="类别额外阈值, 可多次, 如 7=0.5")
    parser.add_argument("--global", dest="global_threshold", type=float, default=None,
                        help="全局阈值 (只对未指定额外阈值的类别生效)")
    parser.add_argument("--zip", action="store_true", help="同时打包输出目录为 zip")
    arguments = parser.parse_args()

    input_dir = Path(arguments.input)
    thresholds = parse_thresholds(arguments.threshold)
    total_images = len(list(input_dir.glob("*.txt")))

    per_class = read_submission(input_dir)
    print_stats(per_class, f"输入 {input_dir.name}", total_images)

    if arguments.output is None:
        return
    output_dir = Path(arguments.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = defaultdict(int)
    removed = defaultdict(int)
    for path in sorted(input_dir.glob("*.txt")):
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 6:
                continue
            cls, conf = int(float(fields[0])), float(fields[5])
            threshold = thresholds.get(cls, arguments.global_threshold or 0.0)
            if conf > threshold:
                lines.append(line)
                kept[cls] += 1
            else:
                removed[cls] += 1
        (output_dir / path.name).write_text("\n".join(lines), encoding="utf-8")

    print(f"== 过滤后 (阈值: 全局={arguments.global_threshold or 0.001} "
          f"额外={thresholds or '无'})")
    for cls in range(len(CLASS_NAMES)):
        if kept[cls] or removed[cls]:
            print(f"  {cls:2d} {CLASS_NAMES[cls]:12s}: {kept[cls]:4d} 保留, {removed[cls]:4d} 剔除")
    print(f"输出: {output_dir}")

    if arguments.zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.glob("*.txt")):
                archive.write(path, path.name)
        print(f"打包: {zip_path}")


if __name__ == "__main__":
    main()
