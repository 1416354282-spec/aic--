#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aic_v7.config import load_config
from aic_v7.data.dataset import MultiModalDataset, collate_fn
from aic_v7.engine.evaluator import predict_batch, predict_batch_tta
from aic_v7.inference.submission import validate_submission, write_prediction
from aic_v7.models.multimodal_dfine import TriModalDFINE


def main():
    parser = argparse.ArgumentParser(description="AIC2026 第七版推理/提交入口")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs" / "submissions" / "predictions"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tta", action="store_true", help="启用 TTA (水平翻转测试时增强)")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    config = load_config(args.config, args.overlay, overrides).to_dict()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
    dataset = MultiModalDataset(config["data"]["test_root"], None, size, False, config["data"], {})
    model = TriModalDFINE(config, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema") or state["model"])
    model.eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset, batch_size=config["train"]["phases"][-1]["batch_size"], shuffle=False,
        num_workers=config["train"].get("workers", 0), collate_fn=collate_fn,
    )
    with torch.no_grad():
        predict_fn = predict_batch_tta if args.tta else predict_batch
        for batch in loader:
            for key in ("img", "ir", "depth", "depth_valid"):
                batch[key] = batch[key].to(device, non_blocking=True)
            detections = predict_fn(model, batch, batch["meta"], config["validation"])
            for image_id, result in zip(batch["image_ids"], detections):
                write_prediction(output / f"{image_id}.txt", result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu())
    errors = validate_submission(output, dataset.ids, config["model"]["num_classes"], config["validation"]["max_detections"])
    if errors:
        raise RuntimeError("提交检查失败:\n" + "\n".join(errors[:20]))
    print(f"完成: {len(dataset)} 张 -> {output}")


if __name__ == "__main__":
    main()
