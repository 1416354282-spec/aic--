"""为「稀有类多保留」标定逐类阈值: 固定稀有类阈值, 反解普通类阈值使总量命中预算。

语义与 scripts/apply_class_threshold.py 及 evaluator._class_thresholds 一致: conf > thr。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aic_v8.constants import CLASS_NAMES

SRC = ROOT / "outputs/submissions/tta_5090run_ep103"
RARE = [1, 7, 9, 10, 11]          # boat, ball, garbage can, uav, tricycle
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 7800

per_class = {c: [] for c in range(12)}
for path in sorted(SRC.glob("*.txt")):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            f = line.split()
            per_class[int(float(f[0]))].append(float(f[5]))
per_class = {c: np.array(v) for c, v in per_class.items()}
common = [c for c in range(12) if c not in RARE]

common_scores = np.concatenate([per_class[c] for c in common])
print(f"普通类候选 {len(common_scores)}  |  稀有类候选 "
      f"{sum(len(per_class[c]) for c in RARE)}\n")
print(f"{'稀有类阈值':>10} | {'稀有类框数':>10} | {'留给普通类':>10} | {'普通类阈值(精确)':>16} | "
      f"{'整数替代':>9} {'总量':>7}")
for t_rare in (0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
    rare_total = sum(int((per_class[c] > t_rare).sum()) for c in RARE)
    remain = BUDGET - rare_total
    if remain <= 0:
        print(f"{t_rare:>10.2f} | {rare_total:>10d} | 预算不足")
        continue
    if remain > len(common_scores):
        remain = len(common_scores)
    exact = np.sort(common_scores)[::-1][remain - 1]
    n_exact = int((common_scores > exact).sum())
    # 找一个给量接近的整数阈值
    best = None
    for cand in [0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        n = int((common_scores > cand).sum())
        if best is None or abs(n - remain) < abs(best[1] - remain):
            best = (cand, n)
    print(f"{t_rare:>10.2f} | {rare_total:>10d} | {remain:>10d} | {exact:>16.6f} | "
          f"{best[0]:>9.2f} {rare_total + best[1]:>7d}")

print("\n== 各稀有类在不同阈值下的框数")
print(f"{'threshold':>10} " + " ".join(f"{CLASS_NAMES[c][:9]:>9}" for c in RARE))
for t in (0.001, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.48):
    row = " ".join(f"{int((per_class[c] > t).sum()):>9d}" for c in RARE)
    print(f"{t:>10.3f} {row}")
