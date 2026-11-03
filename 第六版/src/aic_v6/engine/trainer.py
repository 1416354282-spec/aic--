"""v6 trainer: phase-driven training (three phases per design doc).

Phase structure (from config train.phases):
  each phase: {epochs, lr, size: [w, h], batch_size, freeze_backbone, freeze_decoder}
  - optimizer/scheduler are rebuilt per phase (independent cosine schedules)
  - dataset size switches per phase (multi-resolution, 32-divisible sizes)
  - EMA / AMP / grad accumulation / checkpointing follow v3.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from aic_v6.config import config_fingerprint, save_config
from aic_v6.data.dataset import MultiModalDataset, collate_fn
from aic_v6.engine.checkpoint import load_checkpoint, save_checkpoint
from aic_v6.engine.evaluator import evaluate
from aic_v6.models.ema import ModelEMA
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR

# 连续多少个 batch 的 loss 非有限就认定权重已 NaN、中止训练。
# 依据: 跳过时不做 optimizer.step(), 权重无法自行恢复, 继续跑只是空转。
CONSECUTIVE_SKIP_LIMIT = 25


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_fingerprint(train_ids, val_ids):
    payload = "\n".join(sorted(train_ids) + ["--VAL--"] + sorted(val_ids))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_optimizer(model: TriModalRTDETR, phase: dict):
    base_lr = float(phase["lr"])
    ratios = {
        "rgb": float(phase.get("rgb_lr_ratio", 0.1)),
        "detector": float(phase.get("detector_lr_ratio", 0.5)),
        "classifier": float(phase.get("classifier_lr_ratio", 1.0)),
        "new": 1.0,
    }
    grouped = {(role, decay): [] for role in ratios for decay in (False, True)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = model.parameter_role(name)
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        grouped[(role, use_decay)].append(parameter)
    parameter_groups = []
    for (role, use_decay), parameters in grouped.items():
        if not parameters:
            continue
        parameter_groups.append({
            "params": parameters,
            "lr": base_lr * ratios[role],
            "weight_decay": float(phase.get("weight_decay", 5e-4)) if use_decay else 0.0,
            "role": role,
        })
    if phase.get("optimizer", "adamw").lower() == "sgd":
        return torch.optim.SGD(parameter_groups, lr=base_lr, momentum=phase.get("momentum", 0.937), nesterov=True)
    return torch.optim.AdamW(parameter_groups, lr=base_lr)


def _json_metrics(metrics):
    output = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            flattened = value.detach().float().cpu().flatten()
            output[key] = float(flattened[0]) if len(flattened) == 1 else flattened.tolist()
        else:
            output[key] = value
    return output


def _phase_of_epoch(phases, epoch):
    """Return (phase_index, phase_start_epoch) for a 1-based epoch across phases."""
    start = 0
    for index, phase in enumerate(phases):
        if epoch < start + int(phase["epochs"]):
            return index, start
        start += int(phase["epochs"])
    return len(phases) - 1, start - int(phases[-1]["epochs"])


def run_training(config, resume=None, device=None):
    seed = int(config["project"]["seed"])
    set_seed(seed, config["project"].get("deterministic", True))
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    fold = int(config["data"]["fold"])
    split_dir = Path(config["data"]["split_dir"])
    train_ids = read_ids(split_dir / f"fold_{fold}_train.txt")
    val_ids = read_ids(split_dir / f"fold_{fold}_val.txt")
    split_hash = split_fingerprint(train_ids, val_ids)
    phases = config["train"]["phases"]
    total_epochs = sum(int(phase["epochs"]) for phase in phases)
    initial_size = tuple(int(v) for v in phases[0]["size"])
    train_dataset = MultiModalDataset(
        config["data"]["train_root"], train_ids, initial_size, True,
        config["data"], config["augmentation"],
    )
    val_dataset = MultiModalDataset(
        config["data"]["train_root"], val_ids, initial_size, False,
        config["data"], {},
    )
    model = TriModalRTDETR(config, device)

    ema_decay = config["train"].get("ema", {}).get("decay", 0.9999)
    use_ema = config["train"].get("ema", {}).get("enabled", True)
    ema = ModelEMA(model, ema_decay) if use_ema else None

    start_epoch = 0
    best = -1.0
    global_step = 0
    if resume:
        state = load_checkpoint(resume, model, None, None, None, ema, split_hash, device)
        start_epoch = int(state["epoch"])
        best = float(state["best_metric"])
        global_step = int(state["global_step"])

    run_dir = Path(config["project"]["output_root"]) / (time.strftime("%Y%m%d_%H%M%S") + "_" + config_fingerprint(config))
    (run_dir / "weights").mkdir(parents=True)
    save_config(config, run_dir / "resolved.yaml")
    (run_dir / "environment.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "split_hash": split_hash,
        "rtdetr_mode": config["model"].get("mode"),
        "fusion": config["model"].get("fusion"),
    }, indent=2), encoding="utf-8")

    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_log:
        for epoch in range(start_epoch, total_epochs):
            phase_index, phase_start = _phase_of_epoch(phases, epoch)
            phase = phases[phase_index]
            if epoch == phase_start or (resume and start_epoch == epoch):
                # ── phase boundary (或 resume 起点): rebuild loader / optimizer / scheduler ──
                phase_epochs = int(phase["epochs"])
                train_dataset.set_size(tuple(int(v) for v in phase["size"]))
                val_dataset.set_size(tuple(int(v) for v in phase["size"]))
                loader = DataLoader(
                    train_dataset,
                    batch_size=int(phase["batch_size"]),
                    shuffle=True,
                    num_workers=int(config["train"].get("workers", 0)),
                    pin_memory=device.type == "cuda",
                    drop_last=True,
                    collate_fn=collate_fn,
                )
                model.set_phase(phase_index + 1, config)
                optimizer = build_optimizer(model, phase)
                accumulation = max(1, int(phase.get("accumulation_steps", 1)))
                updates_per_epoch = max(1, int(np.ceil(len(loader) / accumulation)))
                warmup_updates = int(phase.get("warmup_epochs", 3)) * updates_per_epoch
                total_updates = phase_epochs * updates_per_epoch
                final_ratio = float(phase.get("final_lr_ratio", 0.01))

                def schedule(step, warmup=warmup_updates, total=total_updates, final=final_ratio):
                    if step < warmup:
                        return max(step, 1) / max(warmup, 1)
                    progress = (step - warmup) / max(total - warmup, 1)
                    return final + (1 - final) * 0.5 * (1 + np.cos(np.pi * progress))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
                scaler = torch.amp.GradScaler("cuda", enabled=phase.get("amp", True) and device.type == "cuda")
                if resume and state is not None and start_epoch == epoch:
                    # resume within this phase: restore optimizer state
                    load_checkpoint(resume, model, optimizer, scheduler, scaler, None, split_hash, device)

            phase_epoch = epoch - phase_start
            # close-mosaic schedule: the dataset needs the absolute epoch to
            # decide whether mosaic is still active (last N epochs are clean).
            train_dataset.set_epoch(epoch, total_epochs, phase_index + 1)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            skipped_streak = 0
            last_finite_metrics = None
            recent_losses = []
            if device.type == "cuda":
                # 每 epoch 重置峰值, 用于观测显存是否随 epoch 增长 (OOM 排查)
                torch.cuda.reset_peak_memory_stats()
                epoch_mem_start = torch.cuda.memory_allocated() / 1e9
            for batch_index, batch in enumerate(loader):
                train_batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                    if key in {"img", "ir", "depth", "depth_valid", "depth_format",
                               "ir_quality", "depth_quality", "boxes", "labels", "num_boxes"}
                }
                if device.type == "cuda" and batch_index % 50 == 0:
                    print(f"[mem] epoch={epoch+1} batch={batch_index}/{len(loader)} "
                          f"alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
                          f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G "
                          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss, loss_metrics = model.loss(train_batch)
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    # AMP 数值不稳定兜底: 跳过该 batch, 清空梯度避免污染后续累积
                    mem = f" alloc={torch.cuda.memory_allocated()/1e9:.2f}G" \
                        f" peak={torch.cuda.max_memory_allocated()/1e9:.2f}G" if device.type == "cuda" else ""
                    skipped_streak += 1
                    # 打印跳过前后的 loss 轨迹与图片 id: 用来区分"渐变爆炸"还是"突现 NaN",
                    # 并直接指出是哪几张图触发的。(2026-09-27 排查 Phase 1 发散时加)
                    trail = " ".join(f"{i}:{v:.2f}" for i, v in recent_losses[-6:])
                    ids = batch.get("image_ids", [])
                    print(f"[WARNING] epoch={epoch + 1} batch={batch_index} loss 非有限, 跳过该 batch"
                          f" (连续第 {skipped_streak} 次){mem}\n"
                          f"          最近 loss 轨迹 (batch:loss): {trail}\n"
                          f"          本批图片: {ids}", flush=True)
                    # ★ 必须显式释放这批的计算图: continue 不会调用 backward, 图会一直被
                    #   loss / scaled_loss 引用; 下一个 batch 的前向要再开一份图, 两份图叠加
                    #   直接把显存打满 → CUDA OOM (2026-09-27 在 24GB 卡上实测: 平时峰值
                    #   14.34G, 跳过一次后下一次前向 OOM 于 allocated 23.0G)。
                    #   注意不要 del loss_metrics: epoch 末尾的 train_entry 还要用它,
                    #   而且它只装标量 (全是 .detach()), 并不持有计算图。
                    del loss, scaled_loss
                    optimizer.zero_grad(set_to_none=True)
                    # ★ 连续跳过不做 optimizer.step() → 权重一旦变 NaN 就永远回不来,
                    #   整个 run 会以"空转"形式跑完却什么也学不到。所以连续太多次就直接失败,
                    #   免得白等十几个小时。(2026-09-27: Phase 1 曾在 epoch 2 连续 98 次)
                    if skipped_streak >= CONSECUTIVE_SKIP_LIMIT:
                        # 定位: 直接报出哪些参数已经非有限, 精确指向出问题的模块
                        broken = []
                        for name, parameter in model.named_parameters():
                            if not torch.isfinite(parameter).all():
                                count = int((~torch.isfinite(parameter)).sum())
                                broken.append((name, count, parameter.numel()))
                        report = "\n".join(
                            f"    {n}  ({c}/{t} 个元素非有限)" for n, c, t in broken[:20]
                        ) or "    (参数全部有限 —— 说明 NaN 来自激活/缓冲而非权重)"
                        raise RuntimeError(
                            f"连续 {skipped_streak} 个 batch 的 loss 非有限 —— 权重极可能已经 NaN "
                            f"且无法自行恢复 (跳过时不执行 optimizer.step())，"
                            f"已中止以免空转。最后一次: epoch={epoch + 1} batch={batch_index}\n"
                            f"  非有限的参数共 {len(broken)} 个:\n{report}"
                        )
                    continue
                skipped_streak = 0
                last_finite_metrics = loss_metrics
                scaler.scale(scaled_loss).backward()
                loss_value = float(loss.detach())
                loss_sum += loss_value
                recent_losses.append((batch_index, loss_value))
                if len(recent_losses) > 20:
                    del recent_losses[0]
                should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(phase.get("grad_clip", 10.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    if ema:
                        ema.update(model)
            train_entry = {
                "type": "train", "epoch": epoch + 1, "phase": phase_index + 1, "global_step": global_step,
                "loss": loss_sum / max(len(loader), 1), "lr": [group["lr"] for group in optimizer.param_groups],
                "size": list(train_dataset.size), "last_loss": _json_metrics(last_finite_metrics or {}),
                "skipped": skipped_streak,
            }
            if device.type == "cuda":
                train_entry["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
                train_entry["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                train_entry["gpu_epoch_start_gb"] = round(epoch_mem_start, 2)
            metrics_log.write(json.dumps(train_entry, ensure_ascii=False) + "\n")
            metrics_log.flush()
            print(json.dumps(train_entry, ensure_ascii=False))
            save_checkpoint(
                run_dir / "weights/last.pt", model, optimizer, scheduler, scaler, ema,
                epoch + 1, global_step, best, config, split_hash,
            )
            validate_every = int(config["train"].get("validate_every", 5))
            should_validate = (phase_epoch + 1) % validate_every == 0 or epoch + 1 == total_epochs
            if should_validate:
                evaluation_model = ema.ema if ema else model
                metrics = evaluate(evaluation_model, val_dataset, config, device)
                entry = {"type": "val", "epoch": epoch + 1, "phase": phase_index + 1,
                         **{key: value for key, value in metrics.items() if key != "per_class"}}
                metrics_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                metrics_log.flush()
                print(json.dumps(entry, ensure_ascii=False, indent=2))
                if metrics["map50_95"] > best:
                    best = metrics["map50_95"]
                    save_checkpoint(
                        run_dir / "weights/best.pt", model, optimizer, scheduler, scaler, ema,
                        epoch + 1, global_step, best, config, split_hash,
                    )
            save_every = int(config["train"].get("save_every", 10))
            if (phase_epoch + 1) % save_every == 0:
                save_checkpoint(
                    run_dir / f"weights/epoch_{epoch+1}.pt", model, optimizer, scheduler, scaler, ema,
                    epoch + 1, global_step, best, config, split_hash,
                )
    return run_dir
