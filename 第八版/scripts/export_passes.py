"""导出 TTA 两个分支的全量未截断预测, 供阈值扫描精确重放整条后处理流水线。

为什么需要它:
  production 流程是「按类别阈值筛 -> 按分数排序 -> 取 top-100」, 顺序决定了
  提高某类阈值可以腾出名额给其他类。要在验证集上搜这个阈值, 必须拿到截断之前的
  全量候选; predict_raw.py 的 max_detections=100 已经把候选砍掉了。

输出 JSONL, 每图一行:
  {"image_id", "meta": {...}, "passes": [{"boxes","scores","labels"}, {"boxes","scores","labels"}]}
  boxes 为 letterbox 归一化 cxcywh (与 _decode_outputs 输出一致, 未做逆变换),
  scores 已 sigmoid, 每个 pass 含全部 300 个 query, 按分数降序。
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.inference.runtime import load_inference_checkpoint, build_inference_model, select_batch, prepare_prediction_output
from aic_v8.data.dataset import MultiModalDataset, collate_fn
from aic_v8.engine.evaluator import _decode_outputs
from aic_v8.engine.trainer import read_ids
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--target", choices=["val", "test"], default="val")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    config, state = load_inference_checkpoint(arguments.checkpoint, arguments.config, [], arguments.set)
    device = torch.device(arguments.device)
    data_config = config["data"]
    size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
    if arguments.target == "val":
        ids = read_ids(Path(data_config["split_dir"]) / f"fold_{int(data_config['fold'])}_val.txt")
        root = data_config["train_root"]
    else:
        ids, root = None, data_config["test_root"]

    dataset = MultiModalDataset(root, ids, size, False, data_config, {})
    if arguments.limit:
        dataset.ids = dataset.ids[: arguments.limit]

    model = build_inference_model(config, state, device)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=arguments.batch, shuffle=False,
        num_workers=config["train"].get("workers", 0), collate_fn=collate_fn,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # conf=0.0 + 不截断: 保留全部 300 个候选 (分数已是 sigmoid, 恒 > 0)
    raw_postprocess = {"confidence": 0.0, "max_detections": 300, "class_confidence": {}}
    with torch.no_grad(), output.open("w", encoding="utf-8") as handle:
        for batch in loader:
            for key in ("img", "ir", "depth", "depth_valid"):
                batch[key] = batch[key].to(device, non_blocking=True)
            flipped = dict(batch)
            for key in ("img", "ir", "depth", "depth_valid"):
                flipped[key] = torch.flip(batch[key], [3])

            passes = []
            for current in (batch, flipped):
                detections = _decode_outputs(
                    model.predict_raw(current), current["img"].shape[0], device,
                    confidence=raw_postprocess["confidence"],
                    max_detections=raw_postprocess["max_detections"],
                )
                passes.append([
                    {"boxes": d["boxes"].cpu().tolist(),
                     "scores": d["scores"].cpu().tolist(),
                     "labels": d["labels"].cpu().tolist()}
                    for d in detections
                ])
            for index, image_id in enumerate(batch["image_ids"]):
                meta = batch["meta"][index]
                handle.write(json.dumps({
                    "image_id": image_id,
                    "meta": {"original_hw": list(meta.original_hw), "output_hw": list(meta.output_hw),
                             "ratio": float(meta.ratio), "pad_xy": list(meta.pad_xy)},
                    "passes": [passes[0][index], passes[1][index]],
                }) + "\n")
                written += 1
            print(f"[export_passes] {written}/{len(dataset)} ({arguments.target})", flush=True)
    print(f"完成: {written} 张 -> {output}")


if __name__ == "__main__":
    main()
