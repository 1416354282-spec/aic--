from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from aic_v8.io_utils import atomic_output


def restore_rng_state(rng):
    if not rng:
        return
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    if torch.cuda.is_available() and rng.get('cuda') is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])


def load_initial_model_weights(model, weights):
    """Allow old teachers to initialize a NEW run with zero-residual depth adapters only."""
    current = model.state_dict()
    adapter_keys = {name for name in current if name.startswith("depth_encoder.input_adapters.")}
    missing = set(current) - set(weights)
    unexpected = set(weights) - set(current)
    if unexpected or (missing and missing != adapter_keys):
        raise RuntimeError(f"Initialization state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    current.update(weights)
    model.load_state_dict(current, strict=True)
    return sorted(missing)


def save_checkpoint(path, model, optimizer, scheduler, scaler, ema, epoch, global_step, best_metric, config, split_hash, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.ema.state_dict() if ema else None,
        "ema_updates": ema.updates if ema else 0,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
        "split_hash": split_hash,
        "training_state": training_state or {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    with atomic_output(path) as temporary:
        torch.save(payload, temporary)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None, expected_split_hash=None, map_location="cpu", restore_rng=True):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if expected_split_hash and state.get("split_hash") != expected_split_hash:
        raise ValueError("checkpoint 与当前 split 不一致")
    model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer"):
        # 跨阶段 resume: 冻结状态改变导致参数组数变化, 旧优化器状态无法恢复,
        # 从新阶段起点重新 warmup (同阶段 resume 正常恢复)
        if len(optimizer.param_groups) == len(state["optimizer"]["param_groups"]):
            optimizer.load_state_dict(state["optimizer"])
            if scheduler and state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
        else:
            print("[checkpoint] 参数组数不匹配, 跳过优化器/调度器状态 (跨阶段 resume)")
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if ema and state.get("ema"):
        ema.ema.load_state_dict(state["ema"])
        ema.updates = state.get("ema_updates", 0)
    if restore_rng:
        restore_rng_state(state.get('rng'))
    return state
