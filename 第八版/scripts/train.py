#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.config import load_config
from aic_v8.engine.trainer import run_training


parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
parser.add_argument("--overlay", action="append", default=[])
parser.add_argument("--set", action="append", default=[])
parser.add_argument("--resume")
parser.add_argument("--device")
arguments = parser.parse_args()

configuration = load_config(arguments.config, arguments.overlay, arguments.set)
print(run_training(configuration, arguments.resume, arguments.device))
