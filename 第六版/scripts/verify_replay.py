"""校验 tune_thresholds 的重放逻辑与 production 的 predict_batch_tta 完全一致。

做法: 对同几张验证图, 一路走真实 GPU 前向 + predict_batch_tta (conf=0.001, top-100),
另一路读取 export_passes.py 导出的全量候选做重放, 逐框比对。
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.dataset import MultiModalDataset, collate_fn
from aic_v6.engine.evaluator import predict_batch_tta
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR
from aic_v6.engine.trainer import read_ids

sys.path.insert(0, str(ROOT / "scripts"))
from tune_thresholds import replay  # noqa: E402

CONFIG = str(ROOT / "configs/experiments/runs_x_phase2.yaml")
EXPORT = str(ROOT / "outputs/sweep/val_passes_last_newlabels.jsonl")
BATCH = 3  # 必须与 export_passes.py 的 --batch 一致, 否则 cuDNN 选核不同会带来 ~1e-5 扰动
LIMIT = 6

config = load_config(CONFIG, [], [
    "data.train_root=D:/AIC面对城市场景的视觉多模态目标检测/第六版/outputs/tmp_val_root",
    "data.split_dir=D:/AIC面对城市场景的视觉多模态目标检测/第二版/outputs/splits",
])
device = torch.device("cuda:0")
size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
ids = read_ids(Path(config["data"]["split_dir"]) / f"fold_{int(config['data']['fold'])}_val.txt")[:LIMIT]
dataset = MultiModalDataset(config["data"]["train_root"], ids, size, False, config["data"], {})

model = TriModalRTDETR(config, device)
state = torch.load(str(ROOT / "outputs/runs_x/last_newlabels.pt"), map_location=device, weights_only=False)
model.load_state_dict(state.get("ema") or state["model"], strict=False)
model.eval()

from torch.utils.data import DataLoader  # noqa: E402

loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
reference = {}
with torch.no_grad():
    for batch in loader:
        for key in ("img", "ir", "depth", "depth_valid"):
            batch[key] = batch[key].to(device)
        for image_id, result in zip(batch["image_ids"], predict_batch_tta(model, batch, batch["meta"], config["validation"])):
            reference[image_id] = result

exported = {}
with open(EXPORT, encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        exported[record["image_id"]] = record

worst = 0.0
for image_id, expected in reference.items():
    boxes, scores, labels = replay(exported[image_id], {}, 0.001, config["validation"]["nms_iou"],
                                   config["validation"]["max_detections"])
    expected = {key: value.cpu() for key, value in expected.items()}
    same_count = len(boxes) == len(expected["boxes"])
    box_diff = (boxes - expected["boxes"]).abs().max().item() if len(boxes) and same_count else float("nan")
    score_diff = (scores - expected["scores"]).abs().max().item() if len(scores) and same_count else float("nan")
    label_diff = (labels - expected["labels"]).abs().max().item() if len(labels) and same_count else float("nan")
    worst = max([worst, box_diff if box_diff == box_diff else 9e9])
    print(f"{image_id}: 真实 n={len(expected['boxes'])}  重放 n={len(boxes)}  "
          f"box|Δ|={box_diff:.2e} score|Δ|={score_diff:.2e} label|Δ|={label_diff:.2e}")

print("\n判定:", "一致" if worst < 1e-5 else f"不一致 (最大 box 偏差 {worst:.3e})")
