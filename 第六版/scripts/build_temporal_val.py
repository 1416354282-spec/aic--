#!/usr/bin/env python3
"""构建时序验证集: fold_0_val 的 399 帧 + 训练集内同序列的密集邻帧 (±window)。

邻帧只作跟踪上下文 (预测 + GT 都在), mAP 只算在 val 帧上。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_id(image_id: str) -> tuple[str, int] | None:
    """'000013_034_00000001' -> ('000013_034', 1); 'shuming_909_00000270' -> ('shuming_909', 270);
    纯数字 '00000004' -> ('', 4)。"""
    if re.fullmatch(r"\d+", image_id):
        return "", int(image_id)
    parts = image_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ids", required=True, help="fold_0_val.txt")
    parser.add_argument("--train-visible", required=True, help="训练集 visible 目录")
    parser.add_argument("--window", type=int, default=2, help="每侧邻帧数 (默认 2)")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    val_ids = [line.strip() for line in Path(arguments.val_ids).read_text(encoding="utf-8").splitlines() if line.strip()]
    train_ids = sorted(path.stem for path in Path(arguments.train_visible).glob("*") if path.is_file())
    # 每个序列: [(帧号, 实际id字符串)], 用真实 id 而非按位数重建 (000002.png != 00000002)
    by_sequence: dict[str, list[tuple[int, str]]] = {}
    missing = []
    for image_id in train_ids:
        parsed = parse_id(image_id)
        if parsed is None:
            missing.append(image_id)
            continue
        by_sequence.setdefault(parsed[0], []).append((parsed[1], image_id))

    selected: set[str] = set(val_ids)
    skipped = 0
    for image_id in val_ids:
        parsed = parse_id(image_id)
        if parsed is None:
            skipped += 1
            continue
        sequence, frame = parsed
        items = sorted(by_sequence.get(sequence, []))
        positions = [i for i, (number, _) in enumerate(items) if number == frame]
        if not positions:
            skipped += 1
            continue
        index = positions[0]
        for _, neighbor_id in items[max(0, index - arguments.window): index + arguments.window + 1]:
            selected.add(neighbor_id)

    Path(arguments.output).write_text("\n".join(sorted(selected)), encoding="utf-8")
    print(f"val 帧: {len(val_ids)}  未解析/缺失: {skipped}")
    print(f"总帧 (val + 邻帧): {len(selected)}  其中邻帧: {len(selected) - len(set(val_ids))}")
    print(f"输出: {arguments.output}")


if __name__ == "__main__":
    main()
