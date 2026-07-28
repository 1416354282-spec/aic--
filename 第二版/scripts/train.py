#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from aic_v2.config import load_config
from aic_v2.engine.trainer import run_training

p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).parents[1]/"configs/base.yaml"));p.add_argument("--overlay",action="append",default=[]);p.add_argument("--set",action="append",default=[]);p.add_argument("--resume");p.add_argument("--device");a=p.parse_args()
cfg=load_config(a.config,a.overlay,a.set);print(run_training(cfg,a.resume,a.device))
