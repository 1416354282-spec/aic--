from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aic_v2.config import config_fingerprint, save_config
from aic_v2.data.dataset import MultiModalDataset, collate_fn
from aic_v2.engine.checkpoint import load_checkpoint, save_checkpoint
from aic_v2.engine.evaluator import evaluate
from aic_v2.models.early_fusion import EarlyFusionYOLO
from aic_v2.models.ema import ModelEMA


def set_seed(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_fingerprint(train_ids, val_ids):
    return hashlib.sha256("\n".join(sorted(train_ids)+["--VAL--"]+sorted(val_ids)).encode()).hexdigest()[:16]


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    if cfg["optimizer"].lower() == "sgd": return torch.optim.SGD(groups, lr=cfg["learning_rate"], momentum=cfg.get("momentum", .937), nesterov=True)
    return torch.optim.AdamW(groups, lr=cfg["learning_rate"])


def run_training(config, resume=None, device=None):
    seed = config["project"]["seed"]; set_seed(seed, config["project"].get("deterministic", True))
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    fold = config["data"]["fold"]; split_dir = Path(config["data"]["split_dir"])
    train_ids = read_ids(split_dir/f"fold_{fold}_train.txt"); val_ids = read_ids(split_dir/f"fold_{fold}_val.txt")
    split_hash = split_fingerprint(train_ids, val_ids)
    train_ds = MultiModalDataset(config["data"]["train_root"], train_ids, config["data"]["image_size"], True, config["data"], config["augmentation"])
    val_ds = MultiModalDataset(config["data"]["train_root"], val_ids, config["data"]["image_size"], False, config["data"], {})
    train_cfg = config["train"]
    loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["workers"], pin_memory=device.type=="cuda", drop_last=True, collate_fn=collate_fn)
    channels = train_ds[0]["img"].shape[0]
    detector = EarlyFusionYOLO(config["model"]["weights"], channels, config["model"]["num_classes"], str(device), config["model"].get("allow_download", False))
    model = detector.model
    optimizer = build_optimizer(model, train_cfg)
    accumulation = max(1, train_cfg.get("accumulation_steps", 1)); updates_per_epoch = max(1, len(loader)//accumulation)
    total_updates = train_cfg["epochs"] * updates_per_epoch; warmup = train_cfg["warmup_epochs"] * updates_per_epoch; final_ratio = train_cfg["final_lr_ratio"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(step,1)/max(warmup,1) if step<warmup else final_ratio+(1-final_ratio)*.5*(1+np.cos(np.pi*(step-warmup)/max(total_updates-warmup,1))))
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("amp",True) and device.type=="cuda")
    ema = ModelEMA(model, train_cfg["ema"]["decay"]) if train_cfg["ema"].get("enabled",True) else None
    start_epoch=global_step=0; best=-1.0
    if resume:
        state=load_checkpoint(resume,model,optimizer,scheduler,scaler,ema,split_hash,device);start_epoch=state["epoch"];global_step=state["global_step"];best=state["best_metric"]
    run_dir=Path(config["project"]["output_root"])/(time.strftime("%Y%m%d_%H%M%S")+"_"+config_fingerprint(config));(run_dir/"weights").mkdir(parents=True);save_config(config,run_dir/"resolved.yaml")
    (run_dir/"environment.json").write_text(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"device":str(device),"split_hash":split_hash},indent=2),encoding="utf-8")
    metrics_log=open(run_dir/"metrics.jsonl","w",encoding="utf-8")
    def _log(entry:dict):
        metrics_log.write(json.dumps(entry,ensure_ascii=False)+"\n");metrics_log.flush()
    _log({"type":"init","fingerprint":config_fingerprint(config),"split_hash":split_hash,"epochs":train_cfg["epochs"],"lr":train_cfg["learning_rate"],"batch_size":train_cfg["batch_size"],"accumulation":accumulation,"image_size":config["data"]["image_size"]})
    for epoch in range(start_epoch,train_cfg["epochs"]):
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.0
        for batch_index,batch in enumerate(loader):
            train_batch={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items() if k in {"img","batch_idx","cls","bboxes"}}
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
                output=model(train_batch);loss=(output[0].sum() if isinstance(output,(tuple,list)) else output)/accumulation
            if not torch.isfinite(loss): raise FloatingPointError(f"epoch={epoch} batch={batch_index} loss 非有限")
            scaler.scale(loss).backward();loss_sum+=float(loss)*accumulation
            if (batch_index+1)%accumulation==0 or batch_index+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),train_cfg["grad_clip"]);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);scheduler.step();global_step+=1
                if ema: ema.update(model)
        train_loss=loss_sum/max(len(loader),1);lr=float(scheduler.get_last_lr()[0])
        print(f"epoch {epoch+1}/{train_cfg['epochs']} loss={train_loss:.4f} lr={lr:.3e}")
        _log({"type":"train","epoch":epoch+1,"global_step":global_step,"loss":round(train_loss,6),"lr":lr})
        save_checkpoint(run_dir/"weights/last.pt",model,optimizer,scheduler,scaler,ema,epoch+1,global_step,best,config,split_hash)
        if (epoch+1)%train_cfg["validate_every"]==0 or epoch+1==train_cfg["epochs"]:
            eval_model=ema.ema if ema else model;metrics=evaluate(eval_model,val_ds,config,device)
            print(json.dumps({k:v for k,v in metrics.items() if k!="per_class"},indent=2))
            per_class_ap50=[float(np.mean(metrics["per_class"][c][:1])) if metrics["per_class"].get(c) else 0.0 for c in range(config["model"]["num_classes"])]
            _log({"type":"val","epoch":epoch+1,"global_step":global_step,"map50":round(metrics["map50"],6),"map75":round(metrics["map75"],6),"map50_95":round(metrics["map50_95"],6),"per_class_ap50":per_class_ap50})
            if metrics["map50_95"]>best:
                best=metrics["map50_95"];save_checkpoint(run_dir/"weights/best.pt",model,optimizer,scheduler,scaler,ema,epoch+1,global_step,best,config,split_hash)
        if (epoch+1)%train_cfg["save_every"]==0: save_checkpoint(run_dir/f"weights/epoch_{epoch+1}.pt",model,optimizer,scheduler,scaler,ema,epoch+1,global_step,best,config,split_hash)
    metrics_log.close();return run_dir
