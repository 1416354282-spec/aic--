"""GPU 冒烟: 校验 checkpoint 与本地配置的架构一致性, 并跑通单图前向。"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.dataset import MultiModalDataset, collate_fn
from aic_v6.engine.evaluator import predict_batch
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR

config = load_config(str(ROOT / "configs/experiments/runs_x_phase2.yaml"))
device = torch.device("cuda:0")
size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
print(f"device={torch.cuda.get_device_name(0)}  size={size}  weights={config['model']['weights']}")

model = TriModalRTDETR(config, device)
state = torch.load(str(ROOT / "outputs/runs_x/last_newlabels.pt"), map_location=device, weights_only=False)
state_dict = state.get("ema") or state["model"]
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"epoch={state.get('epoch')}  best_metric={state.get('best_metric')}")
print(f"missing={len(missing)}  unexpected={len(unexpected)}")
if missing:
    print("  MISSING:", missing[:10])
if unexpected:
    print("  UNEXPECTED:", unexpected[:10])
model.eval()

dataset = MultiModalDataset(config["data"]["test_root"], None, size, False, config["data"], {})
print(f"dataset ids={len(dataset)}  depth_encoding={config['data']['depth_encoding']}")
items = [dataset[i] for i in range(2)]
batch = collate_fn(items)
for key in ("img", "ir", "depth", "depth_valid"):
    batch[key] = batch[key].to(device)

torch.cuda.reset_peak_memory_stats()
start = time.time()
with torch.no_grad():
    detections = predict_batch(model, batch, batch["meta"], config["validation"])
torch.cuda.synchronize()
print(f"forward {time.time() - start:.2f}s  peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
for image_id, result, item in zip(batch["image_ids"], detections, batch["meta"]):
    print(f"  {image_id}: {len(result['boxes'])} boxes  orig={tuple(item.original_hw)}  "
          f"top={[round(float(s), 3) for s in result['scores'][:5]]}")
