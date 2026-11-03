#!/usr/bin/env python3
"""predict_raw.py 输出的 JSONL → 逐图 txt 目录 (供 temporal_post.py 等工具使用)。"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(arguments.jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows = []
            for box, score, label in zip(record["boxes"], record["scores"], record["labels"]):
                rows.append(" ".join(f"{value:.6g}" for value in [label, *box, score]))
            (output / f"{record['image_id']}.txt").write_text("\n".join(rows), encoding="utf-8")
    print(f"完成: {arguments.jsonl} -> {output}")


if __name__ == "__main__":
    main()
