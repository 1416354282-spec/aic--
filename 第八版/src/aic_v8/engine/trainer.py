"""v6 trainer: phase-driven training (three phases per design doc).

Phase structure (from config train.phases):
  each phase: {epochs, lr, size: [w, h], batch_size, freeze_backbone, freeze_decoder}
  - optimizer/scheduler are rebuilt per phase (independent cosine schedules)
  - dataset size switches per phase (multi-resolution, 32-divisible sizes)
  - EMA / AMP / grad accumulation / checkpointing follow v3.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from aic_v8.config import config_fingerprint, save_config
from aic_v8.data.dataset import MultiModalDataset, collate_fn
from aic_v8.engine.checkpoint import load_checkpoint, save_checkpoint, load_initial_model_weights, restore_rng_state
from aic_v8.data.provenance import dataset_fingerprint, preprocessing_config
from aic_v8.io_utils import atomic_output
from aic_v8.engine.early_stopping import EarlyStopping, Phase2EarlyStopping
from aic_v8.engine.evaluator import evaluate
from aic_v8.models.ema import ModelEMA
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR

# 连续多少个 batch 的 loss 非有限就认定权重已 NaN、中止训练。
# 依据: 跳过时不做 optimizer.step(), 权重无法自行恢复, 继续跑只是空转。
CONSECUTIVE_SKIP_LIMIT = 25
CONSECUTIVE_BAD_UPDATE_LIMIT = 3


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
    learning_rates = {role: base_lr * ratio for role, ratio in ratios.items()}
    if "depth_adapter_lr" in phase:
        learning_rates["depth_adapter"] = float(phase["depth_adapter_lr"])
    grouped = {(role, decay): [] for role in learning_rates for decay in (False, True)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = model.parameter_role(name)
        if "depth_adapter_lr" in phase and name.startswith("depth_encoder.input_adapters."):
            role = "depth_adapter"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        grouped[(role, use_decay)].append(parameter)
    parameter_groups = []
    for (role, use_decay), parameters in grouped.items():
        if not parameters:
            continue
        parameter_groups.append({
            "params": parameters,
            "lr": learning_rates[role],
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


def _nonfinite_parameters(model):
    broken = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not torch.isfinite(parameter).all():
            count = int((~torch.isfinite(parameter)).sum())
            broken.append((name, count, parameter.numel()))
    return broken


def _nonfinite_gradients(model):
    broken = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            count = int((~torch.isfinite(gradient)).sum())
            broken.append((name, count, gradient.numel()))
    return broken


def _format_nonfinite_parameters(broken, limit=20):
    return "\n".join(
        f"    {name}  ({count}/{total} 个元素非有限)"
        for name, count, total in broken[:limit]
    ) or "    (参数全部有限)"


def _phase_of_epoch(phases, epoch):
    """Return (phase_index, phase_start_epoch) for a 1-based epoch across phases."""
    start = 0
    for index, phase in enumerate(phases):
        if epoch < start + int(phase["epochs"]):
            return index, start
        start += int(phase["epochs"])
    return len(phases) - 1, start - int(phases[-1]["epochs"])


def normalize_accumulated_gradients(model, accumulated, planned):
    """Correct a short group's loss/planned to its actual mean before clipping."""
    if accumulated < 1:
        raise ValueError('Cannot normalize an empty accumulation group')
    if accumulated != planned:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(planned / accumulated)


def run_training(config, resume=None, device=None):
    config = copy.deepcopy(config)
    seed = int(config["project"]["seed"])
    set_seed(seed, config["project"].get("deterministic", True))
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    fold = int(config["data"]["fold"])
    split_dir = Path(config["data"]["split_dir"])
    train_ids = read_ids(split_dir / f"fold_{fold}_train.txt")
    val_ids = read_ids(split_dir / f"fold_{fold}_val.txt")
    split_hash = split_fingerprint(train_ids, val_ids)
    data_hash = dataset_fingerprint(config["data"], train_ids, val_ids)
    phases = config["train"]["phases"]
    total_epochs = sum(int(phase["epochs"]) for phase in phases)
    initial_size = tuple(int(v) for v in phases[0]["size"])
    validation_size = tuple(config.get('validation', {}).get('size', phases[-1]['size']))
    evaluation_protocol = config_fingerprint({'version': 2, 'size': validation_size,
                                              'validation': config.get('validation', {})})
    train_dataset = MultiModalDataset(
        config["data"]["train_root"], train_ids, initial_size, True,
        config["data"], config["augmentation"],
    )
    val_dataset = MultiModalDataset(
        config["data"]["train_root"], val_ids, validation_size, False,
        config["data"], {},
    )
    model = TriModalRTDETR(config, device)

    # Transfer only learned weights to a new dataset/run; resume retains its own split checks.
    init_weights = config["train"].get("init_weights")
    teacher_preprocessing = None
    if init_weights and not resume:
        initial = torch.load(init_weights, map_location="cpu", weights_only=False)
        weights = initial.get("ema") if config["train"].get("init_use_ema", True) else None
        source = "ema" if weights is not None else "model"
        added = load_initial_model_weights(model, weights if weights is not None else initial["model"])
        if added:
            print(f"[init] Kept {len(added)} new depth-adapter tensors at identity initialization", flush=True)
        print(f"[init] Loaded {source} weights from {init_weights}; optimizer, epoch and best metric start fresh", flush=True)
        if initial.get("config", {}).get("data"):
            teacher_preprocessing = preprocessing_config(initial["config"]["data"])
        del initial, weights

    ema_decay = config["train"].get("ema", {}).get("decay", 0.9999)
    use_ema = config["train"].get("ema", {}).get("enabled", True)
    ema = ModelEMA(model, ema_decay) if use_ema else None

    start_epoch = 0
    best = -1.0
    global_step = 0
    state = None
    if resume:
        state = load_checkpoint(resume, model, None, None, None, ema, split_hash, device, restore_rng=False)
        start_epoch = int(state["epoch"])
        best = float(state["best_metric"])
        global_step = int(state["global_step"])

    progress = (state or {}).get("training_state", {})
    if state:
        if progress.get('inference_only'):
            raise ValueError('Baseline best is an inference candidate; use init_weights, or resume from last.pt')
        saved_hash = progress.get('dataset_hash')
        if saved_hash is not None and saved_hash != data_hash:
            raise ValueError('Dataset labels/root/preprocessing changed; use init_weights for a new run')
        if saved_hash is None and 'data' in state.get('config', {}):
            if preprocessing_config(state['config']['data']) != preprocessing_config(config['data']):
                raise ValueError('Legacy checkpoint preprocessing differs; use init_weights for a new run')
    rebaseline = bool(state) and progress.get('evaluation_protocol') != evaluation_protocol
    if rebaseline:
        best = -1.0
        print('[resume] Evaluation protocol changed; reset best/patience and remeasure baseline', flush=True)
    stopped_phase_epochs = dict(progress.get("stopped_phase_epochs", {}))
    for index, epochs in stopped_phase_epochs.items():
        phases[int(index)]["epochs"] = int(epochs)
    total_epochs = sum(int(phase["epochs"]) for phase in phases)
    early_stopping = Phase2EarlyStopping(
        config["train"].get("phase2_early_stopping", {}), None if rebaseline else progress.get("early_stopping")
    )
    final_stopping = EarlyStopping(
        config["train"].get("early_stopping", {}),
        None if rebaseline else progress.get("final_early_stopping"),
    )
    if early_stopping.enabled and len(phases) < 3:
        raise ValueError("Phase 2 early stopping requires a third phase")

    run_dir = Path(config["project"]["output_root"]) / (time.strftime("%Y%m%d_%H%M%S") + "_" + config_fingerprint(config))
    (run_dir / "weights").mkdir(parents=True)
    save_config(config, run_dir / "resolved.yaml")
    (run_dir / "environment.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "split_hash": split_hash,
        "rtdetr_mode": config["model"].get("mode"),
        "fusion": config["model"].get("fusion"),
    }, indent=2), encoding="utf-8")

    # Preserve the previous winning artifact when continuing a run into a new directory.
    if resume and not rebaseline:
        previous_best = Path(resume).parent / 'best.pt'
        if previous_best.exists():
            candidate = torch.load(previous_best, map_location='cpu', weights_only=False, mmap=True)
            matching = (candidate.get('split_hash') == split_hash
                        and candidate.get('best_metric') == best
                        and candidate.get('training_state', {}).get('evaluation_protocol') == evaluation_protocol)
            if matching:
                with atomic_output(run_dir / 'weights/best.pt') as temporary:
                    shutil.copyfile(previous_best, temporary)
            else:
                rebaseline = True
            del candidate
        else:
            rebaseline = True
        if rebaseline:
            best = -1.0
            early_stopping = Phase2EarlyStopping(config['train'].get('phase2_early_stopping', {}))
            final_stopping = EarlyStopping(config['train'].get('early_stopping', {}))

    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_log:
        if rebaseline or (init_weights and not resume and config['train'].get('evaluate_initial', True)):
            best_config = config
            original_baseline = None
            if teacher_preprocessing and teacher_preprocessing != preprocessing_config(config['data']):
                original_config = copy.deepcopy(config)
                original_config['data'].update(teacher_preprocessing)
                original_dataset = MultiModalDataset(config['data']['train_root'], val_ids,
                    validation_size, False, original_config['data'], {})
                original_baseline = evaluate(ema.ema if ema else model, original_dataset, original_config, device)
                original_entry = {'type': 'teacher_original_val', 'epoch': 0,
                                  'preprocessing': teacher_preprocessing, **original_baseline}
                metrics_log.write(json.dumps(original_entry, ensure_ascii=False) + '\n')
                print(json.dumps(original_entry, ensure_ascii=False), flush=True)
                del original_dataset
            baseline = evaluate(ema.ema if ema else model, val_dataset, config, device)
            best = float(baseline['map50_95'])
            entry = {'type': 'initial_val', 'epoch': start_epoch, 'source': str(resume or init_weights),
                     'size': list(validation_size), **baseline}
            metrics_log.write(json.dumps(entry, ensure_ascii=False) + '\n')
            metrics_log.flush()
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            if original_baseline and original_baseline['map50_95'] > best:
                best = float(original_baseline['map50_95'])
                best_config = original_config
            if len(phases) == 1 and final_stopping.enabled:
                final_stopping.update(best)
            baseline_state = {'inference_only': True,
                              'optimizer_phase': _phase_of_epoch(phases, start_epoch)[0],
                              'dataset_hash': data_hash, 'evaluation_protocol': evaluation_protocol,
                              'stopped_phase_epochs': stopped_phase_epochs,
                              'early_stopping': early_stopping.state_dict(),
                              'final_early_stopping': final_stopping.state_dict()}
            save_checkpoint(run_dir / 'weights/best.pt', model, None, None, None, ema,
                            start_epoch, global_step, best, best_config, split_hash, baseline_state)
        epoch = start_epoch
        while epoch < total_epochs:
            phase_index, phase_start = _phase_of_epoch(phases, epoch)
            phase = phases[phase_index]
            if epoch == phase_start or (resume and start_epoch == epoch):
                # ── phase boundary (或 resume 起点): rebuild loader / optimizer / scheduler ──
                phase_epochs = int(phase["epochs"])
                train_dataset.set_size(tuple(int(v) for v in phase["size"]))
                val_dataset.set_size(validation_size)
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
                if phase.get('amp', False):
                    raise ValueError('FP32 forward requires amp=false with feature-gradient limiting')
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                if resume and state is not None and start_epoch == epoch:
                    # A checkpoint saved at a phase boundary still owns the OLD optimizer.
                    saved_phase = progress.get("optimizer_phase")
                    if saved_phase is None:
                        saved_phase = _phase_of_epoch(state.get("config", config)["train"]["phases"], max(0, start_epoch - 1))[0]
                    if int(saved_phase) == phase_index:
                        load_checkpoint(resume, model, optimizer, scheduler, scaler, None, split_hash, device, restore_rng=False)
                    # Restore after initialization/baseline and before the next loader iterator.
                    restore_rng_state(state.get('rng'))

            phase_epoch = epoch - phase_start
            model.train()
            if hasattr(model, "reset_modality_gradient_stats"):
                model.reset_modality_gradient_stats()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            skipped_streak = 0
            skipped_batches = 0
            finite_batches = 0
            accumulated = 0
            bad_update_streak = 0
            skipped_updates = 0
            last_finite_metrics = None
            recent_losses = []
            if device.type == "cuda":
                # 每 epoch 重置峰值, 用于观测显存是否随 epoch 增长 (OOM 排查)
                torch.cuda.reset_peak_memory_stats()
                epoch_mem_start = torch.cuda.memory_allocated() / 1e9
            for batch_index, batch in enumerate(loader):
                train_batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items() if key in {"img", "ir", "depth", "depth_valid", "depth_type", "boxes", "labels", "num_boxes"}
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
                    skipped_batches += 1
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
                    accumulated = 0
                    # ★ 连续跳过不做 optimizer.step() → 权重一旦变 NaN 就永远回不来,
                    #   整个 run 会以"空转"形式跑完却什么也学不到。所以连续太多次就直接失败,
                    #   免得白等十几个小时。(2026-09-27: Phase 1 曾在 epoch 2 连续 98 次)
                    if skipped_streak >= CONSECUTIVE_SKIP_LIMIT:
                        # 定位: 直接报出哪些参数已经非有限, 精确指向出问题的模块
                        broken = _nonfinite_parameters(model)
                        report = _format_nonfinite_parameters(broken)
                        if not broken:
                            report = "    (参数全部有限 —— 说明 NaN 来自激活/缓冲而非权重)"
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
                accumulated += 1
                finite_batches += 1
                loss_value = float(loss.detach())
                loss_sum += loss_value
                recent_losses.append((batch_index, loss_value))
                if len(recent_losses) > 20:
                    del recent_losses[0]
                should_step = accumulated == accumulation or batch_index + 1 == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    normalize_accumulated_gradients(model, accumulated, accumulation)
                    accumulated = 0
                    broken_gradients = _nonfinite_gradients(model)
                    if broken_gradients:
                        bad_update_streak += 1
                        skipped_updates += 1
                        print(
                            f"[WARNING] epoch={epoch + 1} batch={batch_index} "
                            f"检测到非有限梯度, 跳过 optimizer.step "
                            f"(连续第 {bad_update_streak} 次)\n"
                            f"  非有限梯度共 {len(broken_gradients)} 个:\n"
                            f"{_format_nonfinite_parameters(broken_gradients)}",
                            flush=True,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        if bad_update_streak >= CONSECUTIVE_BAD_UPDATE_LIMIT:
                            raise RuntimeError(
                                f"连续 {bad_update_streak} 次优化更新检测到非有限梯度，"
                                f"已在参数损坏前中止。最后一次: epoch={epoch + 1} batch={batch_index}"
                            )
                        continue
                    grad_value_clip = float(phase.get("grad_value_clip", 0.0))
                    if grad_value_clip > 0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), grad_value_clip)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(phase.get("grad_clip", 10.0))
                    )
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError(
                            f"有限梯度的全局范数仍溢出 ({float(grad_norm)})；"
                            "请降低 train.phases[].grad_value_clip"
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    broken = _nonfinite_parameters(model)
                    if broken:
                        raise RuntimeError(
                            "optimizer.step 后参数变为非有限，已立即中止。\n"
                            f"{_format_nonfinite_parameters(broken)}"
                        )
                    bad_update_streak = 0
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    if ema:
                        ema.update(model)
            train_entry = {
                "type": "train", "epoch": epoch + 1, "phase": phase_index + 1, "global_step": global_step,
                "loss": loss_sum / max(finite_batches, 1), "lr": [group["lr"] for group in optimizer.param_groups],
                "lr_roles": [group.get("role", "unspecified") for group in optimizer.param_groups],
                "size": list(train_dataset.size), "last_loss": _json_metrics(last_finite_metrics or {}),
                "skipped": skipped_batches, "skipped_updates": skipped_updates,
            }
            if device.type == "cuda":
                train_entry["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
                train_entry["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                train_entry["gpu_epoch_start_gb"] = round(epoch_mem_start, 2)
            if hasattr(model, "pop_modality_gradient_stats"):
                train_entry["modality_gradient_stats"] = model.pop_modality_gradient_stats()
                train_entry["gradient_accumulation_steps"] = accumulation
                train_entry["gradient_scaler_scale"] = float(scaler.get_scale())
            metrics_log.write(json.dumps(train_entry, ensure_ascii=False) + "\n")
            metrics_log.flush()
            print(json.dumps(train_entry, ensure_ascii=False))
            validate_every = int(config["train"].get("validate_every", 5))
            should_validate = (phase_epoch + 1) % validate_every == 0 or phase_epoch + 1 == int(phase["epochs"])
            improved = False
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
                    improved = True
                if phase_index == 1 and early_stopping.enabled:
                    stop = early_stopping.update(metrics["map50_95"])
                    if stop and phase_epoch + 1 < int(phase["epochs"]):
                        planned_epochs = int(phase["epochs"])
                        phase["epochs"] = phase_epoch + 1
                        stopped_phase_epochs[str(phase_index)] = phase_epoch + 1
                        total_epochs = sum(int(item["epochs"]) for item in phases)
                        save_config(config, run_dir / "resolved.yaml")
                        event = {
                            "type": "phase_early_stop", "epoch": epoch + 1, "phase": 2,
                            "next_phase": 3, "planned_phase_epochs": planned_epochs,
                            "actual_phase_epochs": phase_epoch + 1,
                            "patience": early_stopping.patience, "min_delta": early_stopping.min_delta,
                            **early_stopping.state_dict(),
                        }
                        metrics_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                        metrics_log.flush()
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                if phase_index == len(phases) - 1 and final_stopping.enabled:
                    stop = final_stopping.update(metrics["map50_95"])
                    if stop and phase_epoch + 1 < int(phase["epochs"]):
                        planned_epochs = int(phase["epochs"])
                        phase["epochs"] = phase_epoch + 1
                        stopped_phase_epochs[str(phase_index)] = phase_epoch + 1
                        total_epochs = sum(int(item["epochs"]) for item in phases)
                        save_config(config, run_dir / "resolved.yaml")
                        event = {
                            "type": "training_early_stop", "epoch": epoch + 1,
                            "phase": phase_index + 1, "planned_phase_epochs": planned_epochs,
                            "actual_phase_epochs": phase_epoch + 1,
                            "patience": final_stopping.patience, "min_delta": final_stopping.min_delta,
                            **final_stopping.state_dict(),
                        }
                        metrics_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                        metrics_log.flush()
                        print(json.dumps(event, ensure_ascii=False), flush=True)
            training_state = {
                "dataset_hash": data_hash,
                "evaluation_protocol": evaluation_protocol,
                "optimizer_phase": phase_index,
                "stopped_phase_epochs": stopped_phase_epochs,
                "early_stopping": early_stopping.state_dict(),
                "final_early_stopping": final_stopping.state_dict(),
            }
            # Save AFTER validation: resume must retain patience and the shortened schedule.
            save_checkpoint(
                run_dir / "weights/last.pt", model, optimizer, scheduler, scaler, ema,
                epoch + 1, global_step, best, config, split_hash, training_state=training_state,
            )
            if improved:
                save_checkpoint(
                    run_dir / "weights/best.pt", model, optimizer, scheduler, scaler, ema,
                    epoch + 1, global_step, best, config, split_hash, training_state=training_state,
                )
            save_every = int(config["train"].get("save_every", 10))
            if (phase_epoch + 1) % save_every == 0:
                save_checkpoint(
                    run_dir / f"weights/epoch_{epoch+1}.pt", model, optimizer, scheduler, scaler, ema,
                    epoch + 1, global_step, best, config, split_hash, training_state=training_state,
                )
            epoch += 1
    return run_dir
