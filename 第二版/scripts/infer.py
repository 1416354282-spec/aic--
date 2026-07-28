#!/usr/bin/env python3
import argparse,sys,torch
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from aic_v2.config import load_config
from aic_v2.data.dataset import MultiModalDataset,collate_fn
from aic_v2.engine.evaluator import predict_batch
from aic_v2.inference.submission import write_prediction
from aic_v2.models.early_fusion import EarlyFusionYOLO

p=argparse.ArgumentParser();p.add_argument("checkpoint");p.add_argument("--config",default=str(Path(__file__).parents[1]/"configs/base.yaml"));p.add_argument("--overlay",action="append",default=[]);p.add_argument("--set",action="append",default=[]);p.add_argument("--output",default=str(Path(__file__).parents[1]/"outputs/submissions/predictions"));p.add_argument("--device");a=p.parse_args();cfg=load_config(a.config,a.overlay,a.set);device=torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
ds=MultiModalDataset(cfg["data"]["test_root"],None,cfg["data"]["image_size"],False,cfg["data"],{});channels=ds[0]["img"].shape[0];model=EarlyFusionYOLO(cfg["model"]["weights"],channels,cfg["model"]["num_classes"],str(device),False).model;state=torch.load(a.checkpoint,map_location=device,weights_only=False);model.load_state_dict(state.get("ema") or state["model"]);model.eval();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
loader=DataLoader(ds,batch_size=cfg["train"]["batch_size"],shuffle=False,num_workers=cfg["train"]["workers"],collate_fn=collate_fn)
with torch.no_grad():
 for batch in loader:
  det=predict_batch(model,batch["img"].to(device),batch["meta"],cfg["data"]["image_size"],cfg["validation"])
  for image_id,result in zip(batch["image_ids"],det):write_prediction(out/f"{image_id}.txt",result["boxes"].cpu(),result["scores"].cpu(),result["labels"].cpu())
print(f"完成: {len(ds)} 张 -> {out}")
