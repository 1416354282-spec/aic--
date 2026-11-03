"""按「逐类目标框数」出提交产物 (不重新推理)。

与 filter_by_budget.py 的区别: 那个控制总量, 这个直接指定每一类各留多少框。
实现等价于「给每一类单独定一个置信度阈值」—— 保留该类别分数最高的前 N 个,
所以与 evaluator._class_thresholds 的逐类阈值语义一致, 只是用计数而非分数来参数化。

用法:
  python scripts/set_class_quota.py <源产物目录> <输出目录> \
      --quota 0=1986 --quota 1=133 --quota 2=1961 ...
未在 --quota 中出现的类别: 沿用 --baseline 目录里该类别的框数。
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aic_v8.constants import CLASS_NAMES


def load(directory):
    """返回 per_image[image_id] = [(cls, score, line), ...]"""
    per_image = {}
    for path in sorted(Path(directory).glob("*.txt")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                f = line.split()
                rows.append((int(float(f[0])), float(f[5]), line))
        per_image[path.stem] = rows
    return per_image


def counts_by_class(per_image):
    counter = Counter()
    for rows in per_image.values():
        for cls, _, _ in rows:
            counter[cls] += 1
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--quota", action="append", default=[], metavar="CLS=N")
    parser.add_argument("--baseline", default=None, help="未指定配额的类别沿用此目录的框数")
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()

    quota = {}
    for pair in arguments.quota:
        cls, value = pair.split("=", 1)
        quota[int(cls)] = int(value)

    per_image = load(arguments.source)
    available = counts_by_class(per_image)

    if arguments.baseline:
        base_counts = counts_by_class(load(arguments.baseline))
        for cls in range(len(CLASS_NAMES)):
            quota.setdefault(cls, base_counts.get(cls, 0))

    print(f"源 {arguments.source}: {sum(available.values())} 框")
    print(f"{'class':>14} {'目标':>7} {'可用候选':>9} {'实际保留':>9}")
    keep_per_class = {}
    for cls in range(len(CLASS_NAMES)):
        target = min(quota.get(cls, available.get(cls, 0)), available.get(cls, 0))
        keep_per_class[cls] = target
        flag = "" if target == quota.get(cls, 0) else "  <- 候选不足"
        print(f"{CLASS_NAMES[cls]:>14} {quota.get(cls, 0):>7} {available.get(cls, 0):>9} {target:>9}{flag}")

    # 每类取分数最高的前 N 个 (按类别独立排序); 分数并列时按行序稳定
    selected = defaultdict(set)   # cls -> {line}
    for cls, target in keep_per_class.items():
        rows = [(score, line) for img in per_image.values()
                for c, score, line in img if c == cls]
        rows.sort(key=lambda r: -r[0])
        for _, line in rows[:target]:
            selected[cls].add(line)

    if arguments.report_only:
        return

    output = Path(arguments.output)
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    written = Counter()
    for image_id, rows in per_image.items():
        lines = [line for cls, _, line in rows if line in selected[cls]]
        written.update(int(float(line.split()[0])) for line in lines)
        (output / f"{image_id}.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n写出 -> {output}")
    print(f"总计 {sum(written.values())} 框")
    for cls in range(len(CLASS_NAMES)):
        if written[cls] != keep_per_class[cls]:
            print(f"  !! {CLASS_NAMES[cls]}: 期望 {keep_per_class[cls]}, 实得 {written[cls]}")


if __name__ == "__main__":
    main()
