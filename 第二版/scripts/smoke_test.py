#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from aic_v2.config import load_config,config_fingerprint
p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).parents[1]/"configs/base.yaml"));p.add_argument("--overlay",action="append",default=[]);p.add_argument("--set",action="append",default=[]);a=p.parse_args();cfg=load_config(a.config,a.overlay,a.set)
print("配置可解析",config_fingerprint(cfg));print("train_root",cfg["data"]["train_root"]);print("weights",cfg["model"]["weights"])
