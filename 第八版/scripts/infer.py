#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.inference.runtime import load_inference_checkpoint, build_inference_model, select_batch, prepare_prediction_output
from aic_v8.data.dataset import MultiModalDataset, collate_fn
from aic_v8.engine.evaluator import predict_batch, predict_batch_tta
from aic_v8.inference.submission import validate_submission, write_prediction
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=None)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", default=str(ROOT / "outputs/submissions/predictions"))
    parser.add_argument("--device")
    parser.add_argument("--tta", action="store_true", help="启用 TTA (水平翻转测试时增强)")
    arguments = parser.parse_args()

    config, state = load_inference_checkpoint(arguments.checkpoint, arguments.config, arguments.overlay, arguments.set)
    device = torch.device(arguments.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
    dataset = MultiModalDataset(config["data"]["test_root"], None, size, False, config["data"], {})
    model = build_inference_model(config, state, device)
    model.eval()
    output = Path(arguments.output)
    prepare_prediction_output(output, arguments.checkpoint, config, dataset.ids, arguments.tta)
    loader = DataLoader(
        dataset, batch_size=config["train"]["phases"][-1]["batch_size"], shuffle=False,
        num_workers=config["train"]["workers"], collate_fn=collate_fn,
    )
    with torch.no_grad():
        predict_fn = predict_batch_tta if arguments.tta else predict_batch
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
