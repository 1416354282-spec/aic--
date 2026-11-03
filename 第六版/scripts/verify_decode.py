"""交叉验证 _normalize_detr_outputs 与 ultralytics 8.4.108 的 RTDETRDecoder.postprocess 是否等价。

用两个解释器分别执行:
  1) D:/miniconda3/python.exe (ultralytics 8.4.108) --mode ref   -> 调用官方 postprocess, 存 ref.pt
  2) envs/envname python  (ultralytics 8.4.43)  --mode mine      -> 调用本仓库实现, 对比
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["ref", "mine"], required=True)
parser.add_argument("--dump", default=str(ROOT / "outputs/sweep/decode_check.pt"))
arguments = parser.parse_args()

B, Q, NC = 2, 300, 12
generator = torch.Generator().manual_seed(0)
boxes = torch.rand(B, Q, 4, generator=generator)
scores = torch.rand(B, Q, NC, generator=generator)

if arguments.mode == "ref":
    from ultralytics.nn.modules.head import RTDETRDecoder

    class Mock:
        pass

    mock = Mock()
    mock.num_queries, mock.export, mock.nc = Q, False, NC
    reference = RTDETRDecoder.postprocess(mock, boxes.clone(), scores.clone())
    torch.save({"boxes": boxes, "scores": scores, "reference": reference}, arguments.dump)
    print(f"ref: {tuple(reference.shape)} -> {arguments.dump}")
else:
    import ultralytics

    from aic_v6.engine.evaluator import _normalize_detr_outputs

    payload = torch.load(arguments.dump, map_location="cpu", weights_only=True)
    raw = torch.cat([payload["boxes"], payload["scores"]], dim=-1)  # (B, Q, 4+nc)
    mine = _normalize_detr_outputs(raw)
    reference = payload["reference"]
    print(f"ultralytics {ultralytics.__version__}")
    print(f"ref.shape={tuple(reference.shape)}  mine.shape={tuple(mine.shape)}")
    if mine.shape != reference.shape:
        print("FAIL: shape mismatch")
        sys.exit(1)
    print(f"boxes  max|diff| = {(mine[..., :4] - reference[..., :4]).abs().max().item():.3e}")
    print(f"scores max|diff| = {(mine[..., 4] - reference[..., 4]).abs().max().item():.3e}")
    print(f"labels max|diff| = {(mine[..., 5] - reference[..., 5]).abs().max().item():.3e}")
    identical = torch.equal(mine, reference)
    print("完全一致 (torch.equal):", identical)
    # 6 列输入应原样通过 (旧版 ultralytics 路径)
    passthrough = _normalize_detr_outputs(reference)
    print("6 列输入原样通过:", torch.equal(passthrough, reference))
    sys.exit(0 if identical else 1)
