#!/usr/bin/env python3
"""make_split: 为当前数据集布局生成 5 折训练/验证划分 (确定性, 可复现)。

当前本地数据集为扁平命名 (visible/00000004.jpg), 第二版 splits 对应旧布局
(000013_034_00000001), 二者不匹配; 本脚本按 id 哈希确定性划分 80/20。
用法: python scripts/make_split.py --train-root <训练集> --output outputs/splits
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def split_ids(ids, num_folds=5, val_ratio=0.2, seed="aic_v7"):
    """按 id 哈希确定性划分: fold k 的验证集 = hash 尾数落在 [k, k+ratio) 的样本。"""
    folds = [[] for _ in range(num_folds)]
    for image_id in ids:
        digest = hashlib.md5(f"{seed}:{image_id}".encode()).hexdigest()
        bucket = int(digest, 16) % 100000 / 100000  # [0, 1)
        fold = min(int(bucket * num_folds / (1.0 + 1e-12)), num_folds - 1)
        within = (bucket * num_folds) % 1.0
        if within < val_ratio:
            folds[fold].append(image_id)
    return folds


def main():
    parser = argparse.ArgumentParser(description="生成 v7 五折划分 (确定性)")
    parser.add_argument("--train-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/splits")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    visible_dir = Path(args.train_root) / "visible"
    ids = sorted(path.stem for path in visible_dir.glob("*") if path.is_file())
    if not ids:
        raise FileNotFoundError(f"未找到训练图片: {visible_dir}")
    val_sets = split_ids(ids, args.num_folds, args.val_ratio)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for fold, val_ids in enumerate(val_sets):
        train_ids = [image_id for image_id in ids if image_id not in set(val_ids)]
        (output / f"fold_{fold}_train.txt").write_text("\n".join(train_ids), encoding="utf-8")
        (output / f"fold_{fold}_val.txt").write_text("\n".join(val_ids), encoding="utf-8")
        print(f"fold {fold}: train={len(train_ids)} val={len(val_ids)}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
