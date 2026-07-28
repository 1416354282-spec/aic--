from __future__ import annotations

import re
from collections import Counter, defaultdict


def infer_group_id(image_id: str) -> str:
    """保守移除末尾帧号，避免明显相邻帧跨折。"""
    value = re.sub(r"([_-])\d{6,}$", "", image_id)
    return value or image_id


def multilabel_group_folds(records: list[dict], n_folds: int = 3, seed: int = 42) -> list[list[dict]]:
    """贪心 group-aware 多标签近似分层。每个 group 只进入一个 fold。
    
    策略：按 group 大小降序处理（LPT），优先分配给当前样本量最小的 fold，
    同时用类别分布偏离度打破平局。"""
    import random
    groups = defaultdict(list)
    for row in records:
        groups[row.get("group_id") or infer_group_id(row["image_id"])].append(row)
    total_per_class = Counter(c for row in records for c in row.get("classes", []))
    total_records = len(records)
    target_fraction = 1.0 / n_folds

    fold_rows = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    fold_counts = [Counter() for _ in range(n_folds)]

    # 按 group 大小降序（大 group 先分配，有利于平衡）
    items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    rng = random.Random(seed)
    rng.shuffle([item for item in items if len(item[1]) == len(items[0][1])])  # 只 shuffle 同等大小的 group

    for _, rows in items:
        group_counts = Counter(c for row in rows for c in row.get("classes", []))
        group_size = len(rows)

        def _class_imbalance(index: int) -> float:
            """衡量加入该 group 后 fold 的类别偏离度。"""
            imbalance = 0.0
            for c, n in group_counts.items():
                new_frac = (fold_counts[index][c] + n) / max(total_per_class[c], 1)
                imbalance += (new_frac - target_fraction) ** 2
            return imbalance

        # 第一优先级：样本量最小；第二优先级：类别最均衡
        min_size = min(fold_sizes)
        candidates = [i for i, s in enumerate(fold_sizes) if s == min_size]
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = min(candidates, key=_class_imbalance)

        fold_rows[chosen].extend(rows)
        fold_sizes[chosen] += group_size
        fold_counts[chosen].update(group_counts)

    return fold_rows


def assert_no_group_leakage(train: list[dict], val: list[dict]) -> None:
    train_groups = {r.get("group_id") or infer_group_id(r["image_id"]) for r in train}
    val_groups = {r.get("group_id") or infer_group_id(r["image_id"]) for r in val}
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"发现 {len(overlap)} 个跨集合 group")
