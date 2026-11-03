"""独立校验 TTA 预测产物 (不依赖 infer.py 内置校验)。

检查: 文件数/命名与测试集对齐、列数与范围、框数分布、类别分布、坐标退化情况。
用法: python scripts/verify_submission_dir.py <pred_dir> <test_root>
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.constants import CLASS_NAMES

pred_dir = Path(sys.argv[1])
test_root = Path(sys.argv[2])

expected = sorted(p.stem for p in (test_root / "visible").glob("*") if p.is_file())
actual = sorted(p.stem for p in pred_dir.glob("*.txt"))

print(f"测试集图片     : {len(expected)}")
print(f"预测文件       : {len(actual)}")
missing = set(expected) - set(actual)
extra = set(actual) - set(expected)
print(f"缺失           : {len(missing)}  {sorted(missing)[:5]}")
print(f"多出           : {len(extra)}  {sorted(extra)[:5]}")

counts, classes, empties = [], Counter(), 0
bad_range = bad_cols = bad_cls = 0
min_w = min_h = 1e9
for image_id in expected:
    path = pred_dir / f"{image_id}.txt"
    if not path.exists():
        continue
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts.append(len(lines))
    if not lines:
        empties += 1
    for line in lines:
        fields = line.split()
        if len(fields) != 6:
            bad_cols += 1
            continue
        cls = float(fields[0])
        values = [float(v) for v in fields[1:]]
        if cls != int(cls) or not 0 <= cls < 12:
            bad_cls += 1
        if not all(0 <= v <= 1 for v in values):
            bad_range += 1
        classes[int(cls)] += 1
        min_w = min(min_w, values[2])
        min_h = min(min_h, values[3])

total = sum(counts)
print(f"\n总框数         : {total}")
print(f"平均每张       : {total / max(len(counts), 1):.1f}")
print(f"框数范围       : {min(counts)} ~ {max(counts)}  (上限必须 <= 100)")
print(f"空预测图片     : {empties}")
print(f"列数错误       : {bad_cols}")
print(f"类别越界       : {bad_cls}")
print(f"坐标/置信越界  : {bad_range}")
print(f"最窄框 w       : {min_w:.2e}   最矮框 h: {min_h:.2e}")

print("\n类别分布:")
for index, name in enumerate(CLASS_NAMES):
    n = classes.get(index, 0)
    pct = 100.0 * n / max(total, 1)
    print(f"  {index:2d} {name:9s} {n:7d}  {pct:5.1f}%")

ok = (not missing and not extra and not bad_cols and not bad_cls
      and not bad_range and max(counts) <= 100)
print("\n校验结论:", "通过" if ok else "存在问题")
