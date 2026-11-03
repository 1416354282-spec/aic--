#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aic_v7.config import load_config
from aic_v7.engine.trainer import run_training


def parse_overrides(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"覆盖参数必须是 key=value: {item}")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    return overrides


def main():
    parser = argparse.ArgumentParser(description="AIC2026 第七版训练入口 (D-FINE + Relation-Aware Refiner)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.overlay, parse_overrides(args.set)).to_dict()
    run_dir = run_training(config, resume=args.resume, device=args.device)
    print(run_dir)


if __name__ == "__main__":
    main()
