"""AIC2026 Phase 2 推理 (第六版 TriRT-DETR): 可断点续跑, 逐图落盘。

用法:
  python scripts/infer_phase2.py <checkpoint> --output <dir> [--tta] [--batch-size N]
  # 换测试集 (如复赛 PHASE_3): --set data.test_root=<绝对路径>
  # 换推理分辨率: --set train.phases.2.size='[1536, 864]'

注意: Windows 下 DataLoader 用 spawn, 必须保留 __main__ 守卫, 否则 worker 会重跑整个脚本。
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.dataset import MultiModalDataset, collate_fn
from aic_v6.engine.evaluator import predict_batch, predict_batch_tta
from aic_v6.inference.submission import validate_submission, write_prediction
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs/experiments/runs_x_phase2.yaml"))
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overlay, arguments.set)
    device = torch.device(arguments.device)
    size = tuple(int(v) for v in config["train"]["phases"][-1]["size"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = MultiModalDataset(config["data"]["test_root"], None, size, False, config["data"], {})
    done = {path.stem for path in output.glob("*.txt")}
    total = len(dataset)
    print(f"数据集 {total} 张 | 尺寸 {size} | 已完成 {len(done)} | TTA={arguments.tta}", flush=True)

    model = TriModalRTDETR(config, device)
    state = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("ema") or state["model"], strict=False)
    print(f"epoch={state.get('epoch')} best_metric={state.get('best_metric')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=arguments.batch_size, shuffle=False,
        num_workers=arguments.workers, collate_fn=collate_fn, pin_memory=True,
    )
    predict_fn = predict_batch_tta if arguments.tta else predict_batch
    start = time.time()
    processed = 0
    with torch.no_grad():
        for batch in loader:
            pending = [i for i, image_id in enumerate(batch["image_ids"]) if image_id not in done]
            if pending:
                indices = torch.tensor(pending, dtype=torch.long)
                for key in ("img", "ir", "depth", "depth_valid",
                            "ir_quality", "depth_quality", "depth_format"):
                    if key in batch and torch.is_tensor(batch[key]):
                        batch[key] = batch[key][indices].to(device, non_blocking=True)
                metas = [batch["meta"][i] for i in pending]
                detections = predict_fn(model, batch, metas, config["validation"])
                for slot, image_id in enumerate(batch["image_ids"][i] for i in pending):
                    result = detections[slot]
                    write_prediction(
                        output / f"{image_id}.txt",
                        result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu(),
                    )
                processed += len(pending)
            done_now = len(done) + processed
            if done_now and (done_now == total or done_now % 100 < arguments.batch_size):
                elapsed = time.time() - start
                rate = elapsed / max(processed, 1)
                print(f"  {done_now}/{total}  {elapsed / 60:.1f}min  {rate:.2f}s/图  "
                      f"预计剩余 {rate * (total - done_now) / 60:.1f}min", flush=True)

    errors = validate_submission(str(output), dataset.ids, config["model"]["num_classes"], config["validation"]["max_detections"])
    print(f"完成 {total} 张 -> {output}  用时 {(time.time() - start) / 60:.1f}min")
    print("提交检查:", "通过" if not errors else f"失败 {len(errors)} 项")
    for line in errors[:20]:
        print("  ", line)


if __name__ == "__main__":
    main()
