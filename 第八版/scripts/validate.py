#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.inference.runtime import load_inference_checkpoint, build_inference_model
from aic_v8.data.dataset import MultiModalDataset
from aic_v8.engine.evaluator import evaluate
from aic_v8.engine.trainer import read_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=None)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device")
    arguments = parser.parse_args()

    config, state = load_inference_checkpoint(arguments.checkpoint, arguments.config, arguments.overlay, arguments.set)
    device = torch.device(arguments.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    fold = int(config["data"]["fold"])
    ids = read_ids(Path(config["data"]["split_dir"]) / f"fold_{fold}_val.txt")
    size = tuple(int(v) for v in config.get("validation", {}).get("size", config["train"]["phases"][-1]["size"]))
    dataset = MultiModalDataset(config["data"]["train_root"], ids, size, False, config["data"], {})
    model = build_inference_model(config, state, device)
    print(evaluate(model, dataset, config, device))


if __name__ == "__main__":
    main()
