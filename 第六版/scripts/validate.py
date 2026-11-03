#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.dataset import MultiModalDataset
from aic_v6.engine.evaluator import evaluate
from aic_v6.engine.trainer import read_ids
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
parser.add_argument("--overlay", action="append", default=[])
parser.add_argument("--set", action="append", default=[])
parser.add_argument("--device")
arguments = parser.parse_args()

config = load_config(arguments.config, arguments.overlay, arguments.set)
device = torch.device(arguments.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
fold = int(config["data"]["fold"])
ids = read_ids(Path(config["data"]["split_dir"]) / f"fold_{fold}_val.txt")
size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
dataset = MultiModalDataset(config["data"]["train_root"], ids, size, False, config["data"], {})
model = TriModalRTDETR(config, device)
state = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(state.get("ema") or state["model"])
print(evaluate(model, dataset, config, device))
