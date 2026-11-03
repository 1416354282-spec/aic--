"""TTA 推理速度基准: 决定本地走 GPU 还是 CPU。"""
import sys
import time
from pathlib import Path

ROOT = Path(r"D:/AIC面对城市场景的视觉多模态目标检测/第八版")
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from aic_v8.config import load_config
from aic_v8.data.dataset import MultiModalDataset, collate_fn
from aic_v8.engine.evaluator import predict_batch_tta
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR

CKPT = ROOT / "outputs/runs_1600_400_revised_phase3_batch3/weights/best.pt"

device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
images = int(sys.argv[3]) if len(sys.argv) > 3 else 5

config = load_config(
    ROOT / "configs/base.yaml",
    [ROOT / "configs/experiments/v8_4090_fold.yaml",
     ROOT / "configs/experiments/local_5090run.yaml"],
)
size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
print(f"device={device}  batch={batch_size}  size={size}")
print(f"test_root={config['data']['test_root']}")
print(f"weights={config['model']['weights']}")

dataset = MultiModalDataset(config["data"]["test_root"], None, size, False, config["data"], {})
print(f"测试图数量 = {len(dataset.ids)}  首尾: {dataset.ids[0]} .. {dataset.ids[-1]}")

t0 = time.time()
model = TriModalRTDETR(config, device)
print(f"模型构建 + 载入权重: {time.time() - t0:.1f}s")
state = torch.load(CKPT, map_location=device, weights_only=False)
missing, unexpected = model.load_state_dict(state.get("ema") or state["model"], strict=False)
print(f"checkpoint epoch={state.get('epoch')} best_metric={state.get('best_metric'):.4f}")
print(f"missing={len(missing)}  unexpected={len(unexpected)}")
if missing:
    print("  MISSING:", missing[:8])
if unexpected:
    print("  UNEXPECTED:", unexpected[:8])
model.eval()

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

# 先热身一次 (排除首次算子初始化的开销)
with torch.no_grad():
    for batch in loader:
        for key in ("img", "ir", "depth", "depth_valid"):
            batch[key] = batch[key].to(device)
        predict_batch_tta(model, batch, batch["meta"], config["validation"])
        break

torch.cuda.synchronize() if device.type == "cuda" else None
t0 = time.time()
done = 0
with torch.no_grad():
    for batch in loader:
        for key in ("img", "ir", "depth", "depth_valid"):
            batch[key] = batch[key].to(device)
        detections = predict_batch_tta(model, batch, batch["meta"], config["validation"])
        done += len(detections)
        if done >= images:
            break
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - t0
per = elapsed / max(done, 1)
print(f"\nTTA 推理 {done} 张耗时 {elapsed:.1f}s → {per:.2f} s/张")
print(f"1000 张预计 {per * 1000 / 60:.1f} 分钟")
if device.type == "cuda":
    print(f"峰值显存 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
