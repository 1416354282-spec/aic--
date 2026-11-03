#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aic_v7.config import load_config
from aic_v7.data.dataset import MultiModalDataset
from aic_v7.engine.evaluator import evaluate
from aic_v7.engine.trainer import read_ids
from aic_v7.models.multimodal_dfine import TriModalDFINE


def main():
    parser = argparse.ArgumentParser(description="AIC2026 第七版验证入口")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    config = load_config(args.config, args.overlay, overrides).to_dict()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    fold = int(config["data"]["fold"])
    ids = read_ids(Path(config["data"]["split_dir"]) / f"fold_{fold}_val.txt")
    size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
    dataset = MultiModalDataset(config["data"]["train_root"], ids, size, False, config["data"], {})
    model = TriModalDFINE(config, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema") or state["model"])
    print(evaluate(model, dataset, config, device))


if __name__ == "__main__":
    main()
