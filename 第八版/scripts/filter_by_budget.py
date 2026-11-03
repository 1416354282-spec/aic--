"""按「总框数预算」过滤 TTA 产物 (不重新推理)。

取全局第 N 高的分数作为阈值, 使保留框数约为 N。同分并列时可能略有出入, 会打印实际值。
原始产物是 conf>=0.001 + max_detections=100 的稠密输出, 本脚本只做行级过滤。

用法:
  python scripts/filter_by_budget.py <src_dir> <dst_dir> <目标总框数>
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.constants import CLASS_NAMES

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
budget = int(sys.argv[3])

# 读入全部框 (文件顺序固定, 便于复现)
files = sorted(src.glob("*.txt"))
records = []          # (score, image_id)
lines_by_image = {}   # image_id -> [line, ...]
for path in files:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines_by_image[path.stem] = lines
    for line in lines:
        records.append((float(line.split()[5]), path.stem))

total = len(records)
print(f"原始总框数 {total}")
if budget >= total:
    raise SystemExit(f"目标 {budget} >= 原始 {total}, 无需过滤")

scores = np.array([r[0] for r in records])
order = np.argsort(-scores, kind="stable")
# 第 budget 个 (1-based) 的分数作阈值; 用 >= 保留, 故取该分数本身
threshold = float(scores[order[budget - 1]])
kept = int((scores >= threshold).sum())
print(f"全局第 {budget} 高分 = {threshold:.8f}  →  阈值 >= 该值保留 {kept} 框")

dst.mkdir(parents=True, exist_ok=True)
counts, classes, empties = [], Counter(), 0
kept_total = 0
for path in files:
    out = []
    for line in lines_by_image[path.stem]:
        if float(line.split()[5]) >= threshold:
            out.append(line)
            classes[int(float(line.split()[0]))] += 1
    kept_total += len(out)
    counts.append(len(out))
    if not out:
        empties += 1
    (dst / path.name).write_text("\n".join(out), encoding="utf-8")

print(f"\n写出 {len(files)} 个文件 -> {dst}")
print(f"保留总框数   : {kept_total}  (目标 {budget})")
print(f"平均每张     : {kept_total / len(files):.2f}")
print(f"框数范围     : {min(counts)} ~ {max(counts)}")
print(f"空图         : {empties}")
print("\n类别分布:")
for index, name in enumerate(CLASS_NAMES):
    n = classes.get(index, 0)
    print(f"  {index:2d} {name:9s} {n:6d}  {100.0 * n / max(kept_total, 1):5.1f}%")
