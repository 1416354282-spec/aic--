#!/usr/bin/env python3
import argparse,json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from aic_v2.config import load_config
from aic_v2.data.dataset import MultiModalDataset
from aic_v2.engine.checkpoint import load_checkpoint
from aic_v2.engine.evaluator import evaluate
from aic_v2.models.early_fusion import EarlyFusionYOLO

p=argparse.ArgumentParser();p.add_argument("checkpoint");p.add_argument("--config",default=str(Path(__file__).parents[1]/"configs/base.yaml"));p.add_argument("--overlay",action="append",default=[]);p.add_argument("--set",action="append",default=[]);p.add_argument("--device");a=p.parse_args();cfg=load_config(a.config,a.overlay,a.set);device=torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
ids=[x.strip() for x in (Path(cfg["data"]["split_dir"])/f"fold_{cfg['data']['fold']}_val.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
ds=MultiModalDataset(cfg["data"]["train_root"],ids,cfg["data"]["image_size"],False,cfg["data"],{});channels=ds[0]["img"].shape[0];model=EarlyFusionYOLO(cfg["model"]["weights"],channels,cfg["model"]["num_classes"],str(device),False).model
state=torch.load(a.checkpoint,map_location=device);model.load_state_dict(state.get("ema") or state["model"]);print(json.dumps(evaluate(model,ds,cfg,device),ensure_ascii=False,indent=2))
