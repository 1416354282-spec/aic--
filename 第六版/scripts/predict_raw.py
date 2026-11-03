#!/usr/bin/env python3
"""导出原始(未阈值化)预测, 供各类别阈值扫描使用。

为什么要这个脚本:
  阈值调整只有在"验证集 + 同一 checkpoint + 同一后处理"上才能闭环。
  而已提交的 x_last_1920_38_v2 已经把阈值烙进了文件 (各类 min conf 就是当时的阈值),
  任何低于该底线的阈值都无法从 v2 里恢复被删掉的框 —— 必须重新导出 conf=0.001 的原始预测。

用法 (远程 5090D 上执行):
  # 1) 验证集原始预测 (用与提交完全相同的 checkpoint 与 --tta 设置)
  python scripts/predict_raw.py <checkpoint> --target val --tta \
      --config configs/base.yaml --overlay configs/experiments/rtdetr_x.yaml \
      --output outputs/sweep/val_raw.jsonl

  # 2) 测试集原始预测 (同样设置)
  python scripts/predict_raw.py <checkpoint> --target test --tta \
      --config configs/base.yaml --overlay configs/experiments/rtdetr_x.yaml \
      --output outputs/sweep/test_raw.jsonl

  # 然后在本机/远程跑 scripts/sweep_thresholds.py 得到各类阈值,
  # 再用 scripts/apply_class_threshold.py 套用到测试集原始预测上。

说明:
  - 输出为 JSONL, 每图一行: {"image_id", "boxes": [[cx,cy,w,h]...], "scores": [...], "labels": [...]}
    坐标已是原图归一化 cxcywh, 与提交格式一致。
  - 导出时强制 confidence=0.001 且忽略 config 里的 class_confidence (保证真正原始)。
  - 请务必用生成 x_last_1920_38_v2 的那个 checkpoint (如远程 last_ep38) 与相同 --tta 设置,
    否则验证集上搜出的阈值与测试集行为不一致。
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.dataset import MultiModalDataset, collate_fn
from aic_v6.engine.evaluator import predict_batch, predict_batch_tta
from aic_v6.engine.trainer import read_ids
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--target", choices=["val", "test"], default="val")
    parser.add_argument("--output", required=True, help="JSONL 输出路径")
    parser.add_argument("--tta", action="store_true", help="与提交一致的 TTA (水平翻转)")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 张 (冒烟用)")
    parser.add_argument("--ids-file", default=None, help="自定义 id 列表文件 (每行一个 id, 覆盖 val/test 选择)")
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overlay, arguments.set)
    device = torch.device(arguments.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    data_config = config["data"]
    fold = int(data_config["fold"])
    size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])

    if arguments.target == "val":
        ids = read_ids(Path(data_config["split_dir"]) / f"fold_{fold}_val.txt")
        root = data_config["train_root"]
    else:
        ids = None
        root = data_config["test_root"]

    if arguments.ids_file:
        ids = read_ids(Path(arguments.ids_file))
        root = data_config["train_root"]

    dataset = MultiModalDataset(root, ids, size, False, data_config, {})
    if arguments.limit:
        dataset.ids = dataset.ids[: arguments.limit] if isinstance(dataset.ids, list) else dataset.ids[: arguments.limit]
    if arguments.target == "val" and arguments.limit:
        dataset.ids = dataset.ids[: arguments.limit]

    model = TriModalRTDETR(config, device)
    state = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema") or state["model"])
    model.eval()

    # 强制原始: 全局 0.001, 忽略配置里的 class_confidence
    postprocess = {**config["validation"], "confidence": 0.001, "class_confidence": {}}
    batch_size = arguments.batch or config["train"]["phases"][-1]["batch_size"]
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=config["train"].get("workers", 0), collate_fn=collate_fn,
    )

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predict_fn = predict_batch_tta if arguments.tta else predict_batch
    written = 0
    with torch.no_grad(), output.open("w", encoding="utf-8") as handle:
        for batch in loader:
            for key in ("img", "ir", "depth", "depth_valid"):
                batch[key] = batch[key].to(device, non_blocking=True)
            detections = predict_fn(model, batch, batch["meta"], postprocess)
            for image_id, result in zip(batch["image_ids"], detections):
                handle.write(json.dumps({
                    "image_id": image_id,
                    "boxes": result["boxes"].cpu().tolist(),
                    "scores": result["scores"].cpu().tolist(),
                    "labels": result["labels"].cpu().tolist(),
                }) + "\n")
                written += 1
            print(f"[predict_raw] {written}/{len(dataset)}  ({arguments.target})", flush=True)

    print(f"完成: {written} 张 -> {output}")


if __name__ == "__main__":
    main()
