#!/usr/bin/env python3
"""build_priors: Relation-Aware Refiner 的离线先验统计 (设计文档 §3.7.1).

仅使用官方训练集标注, 输出:
  - cooc: 共现矩阵 (C×C), P_cooc[c_i][c_j] = P(c_j 与 c_i 同图出现), 加性平滑防 0
  - spatial_mu / spatial_cov: 逐类框中心 (cx, cy) 的 2D 高斯 (归一化坐标)
  - 熵加权: 高熵类 (分布近均匀) 先验自动趋 0, 折叠进协方差 (Σ_eff = Σ / w)

合规性: 先验全部来自官方训练集 GT 统计, 无外部数据、离线运行, 且以冻结常数
注入 (非参数, 只提供归纳偏置)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aic_v7.constants import NUM_CLASSES
from aic_v7.data.manifest import read_labels


def collect_priors(labels_dir: Path, smoothing: float = 1e-2, ridge: float = 1e-4):
    """返回 (cooc, spatial_mu, spatial_cov, counts, stats)。"""
    cooc_count = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    centers: list[list[np.ndarray]] = [[] for _ in range(NUM_CLASSES)]
    counts = np.zeros((NUM_CLASSES,), dtype=np.float64)
    images = 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        labels, _ = read_labels(label_path)
        if len(labels) == 0:
            continue
        images += 1
        classes = labels[:, 0].astype(int)
        present = set(classes.tolist())
        for cls in present:
            counts[cls] += 1
        for cls in present:
            for other in present:
                cooc_count[cls, other] += 1
        for row in labels:
            cls = int(row[0])
            centers[cls].append(np.asarray(row[1:3], np.float32))

    # ── 共现矩阵: P(c_j | c_i), 加性平滑 ──
    denom = counts[:, None] + smoothing * NUM_CLASSES
    cooc = (cooc_count + smoothing) / denom

    # ── 空间先验: 逐类中心 2D 高斯 ──
    spatial_mu = np.zeros((NUM_CLASSES, 2), np.float32)
    spatial_cov = np.zeros((NUM_CLASSES, 2, 2), np.float32)
    entropies = np.zeros((NUM_CLASSES,), np.float32)
    for cls in range(NUM_CLASSES):
        if len(centers[cls]) == 0:
            spatial_cov[cls] = np.eye(2, dtype=np.float32)  # 无样本: 高熵 → 先验趋 0
            entropies[cls] = 1.0
            continue
        points = np.stack(centers[cls])
        mu = points.mean(0)
        if len(points) < 2:
            # 单样本类: 无协方差可言, 用 ridge 对角 (高熵 → 先验趋 0)
            cov = np.eye(2, dtype=np.float32) * ridge
            det = float(ridge)
        else:
            cov = np.cov(points, rowvar=False) + np.eye(2) * ridge
            sign, logdet = np.linalg.slogdet(cov)
            det = float(np.exp(logdet)) if sign > 0 else float(ridge)
        # 高斯熵 H = 0.5·log((2πe)²·|Σ|); 归一化到 [0, 1] 供熵加权
        entropies[cls] = 0.5 * (np.log(2 * np.pi * np.e) * 2 + np.log(max(det, 1e-12)))
        spatial_mu[cls] = mu
        spatial_cov[cls] = cov

    # ── 熵加权折叠进协方差: 高熵类协方差放大 → log 密度平坦 → 先验自动失效 ──
    entropy_min, entropy_max = entropies.min(), entropies.max()
    span = max(entropy_max - entropy_min, 1e-6)
    weights = np.exp(-(entropies - entropy_min) / span)  # (0, 1], 高熵 → 小权重
    spatial_cov = spatial_cov / np.maximum(weights[:, None, None], 1e-3)

    stats = {
        "images": images,
        "class_counts": counts.tolist(),
        "entropies": entropies.tolist(),
        "entropy_weights": weights.tolist(),
    }
    return cooc.astype(np.float32), spatial_mu, spatial_cov, counts.astype(np.float32), stats


def main():
    parser = argparse.ArgumentParser(description="Build relation priors from training labels (设计文档 §3.7.1)")
    parser.add_argument("--labels-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--smoothing", type=float, default=1e-2)
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()
    cooc, spatial_mu, spatial_cov, counts, stats = collect_priors(
        Path(args.labels_dir), args.smoothing, args.ridge
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cooc=cooc,
        spatial_mu=spatial_mu,
        spatial_cov=spatial_cov,
        counts=counts,
        meta=json.dumps({**stats, "num_classes": NUM_CLASSES, "smoothing": args.smoothing, "ridge": args.ridge}),
    )
    print(f"saved {output} ({stats['images']} images)")
    for cls in range(NUM_CLASSES):
        print(f"  class {cls:2d}: count={int(counts[cls]):5d} entropy={stats['entropies'][cls]:.3f} "
              f"weight={stats['entropy_weights'][cls]:.3f}")


if __name__ == "__main__":
    main()
