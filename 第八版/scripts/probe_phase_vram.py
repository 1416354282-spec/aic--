"""训练态显存探针: 在给定 phase 尺寸下试探 batch_size 上限。

与 smoke_gpu_phase2.py(只测推理)不同, 这里跑的是**真实训练步**:
  model.train() + predict_raw(含 denoising 构造) + detector.loss + mask 辅助损失 + backward
所以它量到的是训练峰值显存 —— 也就是选 batch_size 时真正会 OOM 的那个数。

用法 (GPU 机器上, 需要 train_root/labels 存在):
  python scripts/probe_phase_vram.py --size 1536 864 --batch-sizes 3 4 5
  python scripts/probe_phase_vram.py --size 1920 1088 --batch-sizes 2 3 \
      --overlay configs/experiments/runs_x.yaml --overlay configs/remote_westd.yaml

输出每个 batch_size 的峰值显存与耗时; OOM 会打标记并继续下一个, 不中断整轮试探。
默认按"最后一个 phase = 全解冻"来测 —— 那是最坏情况 (显存峰值最高)。

⚠ 测 Phase 1/2 必须加 --phase 1 / --phase 2, 否则会用全解冻的权重状态去测小分辨率,
  得出远高于实际的峰值 (实测 24GB 4090 上全解冻的 640x640 batch16 会 OOM, 但真实 Phase 1
  冻结了 backbone+decoder, 峰值低得多)。--phase 只影响 requires_grad, 梯度累积仍需另算。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.config import load_config
from aic_v8.data.dataset import MultiModalDataset, collate_fn
from aic_v8.engine.trainer import read_ids
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR

TRAIN_KEYS = ("img", "ir", "depth", "depth_valid", "boxes", "labels", "num_boxes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs=2, type=int, required=True, metavar=("W", "H"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--accumulation", type=int, default=1, help="仅用于报告等效 batch")
    parser.add_argument("--phase", type=int, default=None,
                        help="要测的阶段序号 (1-based)。默认最后一个阶段 (全解冻 = 最坏情况)。"
                             "测 Phase 1/2 必须显式指定, 否则会按全解冻测出远高于实际的显存。")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备 (本地 CPU 上跑这个探针没有意义)")

    size = tuple(arguments.size)
    if any(value % 32 for value in size):
        raise SystemExit(f"{size} 必须是 32 的倍数 (RT-DETR 约束)")
    config = load_config(arguments.config, arguments.overlay, arguments.set)
    device = torch.device(arguments.device)
    num_phases = len(config["train"]["phases"])
    phase_index = int(arguments.phase) if arguments.phase else num_phases
    if not 1 <= phase_index <= num_phases:
        raise SystemExit(f"--phase 必须在 1..{num_phases} 之间, 收到 {phase_index}")
    config["train"]["phases"][phase_index - 1]["size"] = list(size)   # 让 evaluate/letterbox 用同一个尺寸

    fold = int(config["data"]["fold"])
    split_dir = Path(config["data"]["split_dir"])
    train_ids = read_ids(split_dir / f"fold_{fold}_train.txt")
    dataset = MultiModalDataset(
        config["data"]["train_root"], train_ids, size, True, config["data"], config["augmentation"]
    )

    model = TriModalRTDETR(config, device)
    model.set_phase(phase_index, config)
    model.train()
    phase_spec = config["train"]["phases"][phase_index - 1]
    freeze = "全解冻" if not (phase_spec.get("freeze_backbone") or phase_spec.get("freeze_decoder")) \
        else f"freeze_backbone={phase_spec.get('freeze_backbone', True)} freeze_decoder={phase_spec.get('freeze_decoder', True)}"

    print(f"device={torch.cuda.get_device_name(0)}  总显存={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"size={size[0]}x{size[1]}  phase={phase_index}/{num_phases} ({freeze})  weights={config['model']['weights']}")
    print(f"baseline(模型+EMA 之前) 已占用 {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    for batch_size in arguments.batch_sizes:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True, collate_fn=collate_fn)
        batch = next(iter(loader))
        train_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in TRAIN_KEYS}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            loss, _ = model.loss(train_batch)
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  batch={batch_size:>2d}  峰值 {peak:6.2f} GB  单步 {time.time() - started:5.2f}s  "
                  f"等效 batch={batch_size * arguments.accumulation}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch_size:>2d}  OOM (跳过)", flush=True)
        finally:
            model.zero_grad(set_to_none=True)
            del batch, train_batch
            torch.cuda.empty_cache()
    print("提示: 训练时还有 EMA(一份完整模型副本, 约 +0.4GB) 与梯度累积, 请给峰值留 2~3 GB 余量。")


if __name__ == "__main__":
    main()
