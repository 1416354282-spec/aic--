"""CPU 冒烟测试: config / 先验 / Relation Refiner / 模型构建 / 训练前向 / 推理 / 相位 / 数据管线 / 提交。

运行: pytest tests/test_smoke.py -v  (或 python tests/test_smoke.py)

注意: 模型构建类测试需要 weights/dfine_l_coco.pth (约 126MB), 缺失时自动跳过。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aic_v7.config import load_config
from aic_v7.constants import NUM_CLASSES
from aic_v7.data.dataset import MultiModalDataset, collate_fn
from aic_v7.engine.checkpoint import load_checkpoint, save_checkpoint
from aic_v7.engine.trainer import build_optimizer, split_fingerprint
from aic_v7.inference.submission import validate_submission, write_prediction
from aic_v7.models.relation_refiner import RelationAwareRefiner

WEIGHTS = ROOT / "weights" / "dfine_l_coco.pth"
TORCH_SEED = 0


def _seed():
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)


def _make_batch(device="cpu", size=(128, 128), num_gt=2, batch=1):
    """合成 v6 风格 batch (归一化 letterbox 空间, 含真实 GT)。"""
    _seed()
    width, height = size
    batch_data = {
        "img": torch.rand(batch, 3, height, width),
        "ir": torch.rand(batch, 1, height, width),
        "depth": torch.rand(batch, 3, height, width),
        "depth_valid": torch.ones(batch, 1, height, width),
        "boxes": torch.zeros(batch, num_gt, 4),
        "labels": torch.zeros(batch, num_gt, dtype=torch.long),
        "num_boxes": torch.full((batch,), num_gt, dtype=torch.long),
    }
    for i in range(batch):
        for j in range(num_gt):
            batch_data["boxes"][i, j] = torch.tensor([0.25 + 0.5 * (j + i) / num_gt, 0.5, 0.2, 0.3])
            batch_data["labels"][i, j] = (j + i) % NUM_CLASSES
    return {key: value.to(device) for key, value in batch_data.items()}


# ─────────────────────────── config ───────────────────────────

def test_config_base_and_experiments():
    base = load_config(str(ROOT / "configs" / "base.yaml")).to_dict()
    assert base["model"]["base"] == "dfine"
    assert base["model"]["mode"] == "tri"
    assert base["data"]["num_classes"] == NUM_CLASSES
    assert Path(base["model"]["weights"]).parent == ROOT / "weights"
    assert len(base["train"]["phases"]) == 3
    assert all(v % 32 == 0 for phase in base["train"]["phases"] for v in phase["size"])

    experiments = {
        "rgb_only.yaml": ("rgb_only", 0.0),
        "no_relation.yaml": ("tri", 0.0),
        "cooc_only.yaml": ("tri", 0.15),
        "spatial_only.yaml": ("tri", 0.15),
        "relation_app.yaml": ("tri", 0.15),
        "cmtf.yaml": ("tri", 0.15),
        "dfine_x.yaml": ("tri", 0.15),
        "no_conditional_dropout.yaml": ("tri", 0.15),
        "no_refiner.yaml": ("tri", 0.15),
    }
    for name, (mode, alpha_cooc) in experiments.items():
        config = load_config(str(ROOT / "configs" / "experiments" / name)).to_dict()
        assert config["model"]["mode"] == mode, name
        assert abs(config["model"]["relation"]["alpha_cooc"] - alpha_cooc) < 1e-9, name
    config = load_config(str(ROOT / "configs" / "experiments" / "no_conditional_dropout.yaml")).to_dict()
    assert config["augmentation"]["infrared_dropout"] == 0.0
    assert config["augmentation"]["depth_dropout"] == 0.0
    config = load_config(str(ROOT / "configs" / "experiments" / "no_refiner.yaml")).to_dict()
    assert config["model"]["target_refinement_levels"] == []
    config = load_config(str(ROOT / "configs" / "experiments" / "dfine_x.yaml")).to_dict()
    assert "dfine_hgnetv2_x_coco" in config["model"]["dfine_yaml"]


def test_config_validation_rejects_bad_inputs():
    from aic_v7.config import validate_config

    good = load_config(str(ROOT / "configs" / "base.yaml")).to_dict()
    bad = dict(good)
    bad["model"]["fusion"] = "nope"
    with pytest.raises(ValueError):
        validate_config(bad)
    bad = dict(good)
    bad["train"]["phases"] = []
    with pytest.raises(ValueError):
        validate_config(bad)
    bad = dict(good)
    bad["model"]["relation"]["alpha_geo"] = 2.0
    with pytest.raises(ValueError):
        validate_config(bad)


# ─────────────────────────── priors ───────────────────────────

def test_build_priors_and_refiner_load(tmp_path):
    from scripts.build_priors import collect_priors

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    # 4 张合成图: 类别共现 (0,1), (0,2), (2,3); 空间: class1 在上部, class2 在下部
    samples = [
        [(0, 0.5, 0.5), (1, 0.5, 0.2)],
        [(0, 0.4, 0.6), (2, 0.5, 0.8)],
        [(2, 0.5, 0.85), (3, 0.3, 0.3)],
        [(0, 0.6, 0.4), (1, 0.5, 0.25), (2, 0.5, 0.75)],
    ]
    for index, boxes in enumerate(samples):
        lines = [f"{cls} {cx} {cy} 0.1 0.1" for cls, cx, cy in boxes]
        (labels_dir / f"{index:08d}.txt").write_text("\n".join(lines), encoding="utf-8")

    cooc, mu, cov, counts, stats = collect_priors(labels_dir)
    assert cooc.shape == (NUM_CLASSES, NUM_CLASSES)
    assert mu.shape == (NUM_CLASSES, 2)
    assert cov.shape == (NUM_CLASSES, 2, 2)
    assert int(counts[0]) == 3 and int(counts[2]) == 3 and int(counts[1]) == 2
    # class1 在上部 (cy≈0.2), class2 在下部 (cy≈0.8)
    assert mu[1, 1] < 0.5 < mu[2, 1]
    # 共现: class1 出现时 class0 必现; class3 只与 class2 同图
    assert cooc[1, 0] > 0.9
    assert cooc[3, 2] > 0.9 and cooc[3, 0] < 0.2

    output = tmp_path / "priors.npz"
    np.savez_compressed(output, cooc=cooc, spatial_mu=mu, spatial_cov=cov, counts=counts)
    refiner = RelationAwareRefiner(num_classes=NUM_CLASSES)
    refiner.load_priors_npz(str(output))
    assert bool(refiner.has_cooc) and bool(refiner.has_spatial)


# ─────────────────────────── relation refiner ───────────────────────────

def test_relation_refiner_identity_and_warmup():
    refiner = RelationAwareRefiner(num_classes=NUM_CLASSES, alpha_cooc=0.15, alpha_spatial=0.10, alpha_geo=0.20)
    # 未加载先验 + 几何 MLP 零初始化 → 精炼前后恒等
    _seed()
    logits = torch.randn(2, 300, NUM_CLASSES)
    boxes = torch.rand(2, 300, 4) * 0.8 + 0.1
    refined = refiner.refine_logits(logits, boxes)
    assert torch.allclose(refined, logits, atol=1e-5)

    # 加载先验后: 共现/空间偏置生效, 但 epoch0 warmup=1/5 时幅度受限
    cooc = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    cooc[6, 11] = 0.8  # car → tricycle 强共现
    mu = torch.zeros(NUM_CLASSES, 2) + 0.5
    mu[10, 1] = 0.1  # uav 在画面上部
    cov = torch.eye(2).repeat(NUM_CLASSES, 1, 1) * 0.05
    refiner.set_priors(cooc=cooc, spatial_mu=mu, spatial_cov=cov)
    refiner.set_epoch(0)
    refined0 = refiner.refine_logits(logits, boxes)
    assert not torch.allclose(refined0, logits)
    refiner.set_epoch(4)
    refined4 = refiner.refine_logits(logits, boxes)
    # warmup: epoch4 的偏置幅度 > epoch0
    assert torch.norm(refined4 - logits) > torch.norm(refined0 - logits)

    # 几何消息: 训练一步后 geo_mlp 非零 → 输出变化; 坐标不变性由外部保证
    loss = refiner.varifocal_loss(refined4, boxes, [], [])
    loss.backward()
    # 回归断言: 末层零初始化 (恒等起点) 下末层必须有非零梯度 —— 末层外再套
    # ReLU 会把零初始化变成梯度死锁 (ReLU'(0)=0 截断, E4 消融假阴性)
    assert float(refiner.geo_mlp[-1].weight.grad.abs().sum()) > 0


def test_relation_refiner_varifocal_on_matched():
    refiner = RelationAwareRefiner(num_classes=NUM_CLASSES)
    refined = torch.randn(1, 6, NUM_CLASSES, requires_grad=True)
    boxes = torch.tensor([[[0.3, 0.4, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3], [0.7, 0.7, 0.1, 0.1]]])
    targets = [{"boxes": torch.tensor([[0.3, 0.4, 0.2, 0.2]]), "labels": torch.tensor([7])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    loss = refiner.varifocal_loss(refined, boxes, targets, indices)
    assert torch.isfinite(loss) and float(loss.detach()) > 0


# ─────────────────────────── model (需要官方权重) ───────────────────────────

requires_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="dfine_l_coco.pth 未下载")


@requires_weights
def _build_model(config_path="configs/base.yaml", device="cpu"):
    from aic_v7.models.multimodal_dfine import TriModalDFINE

    config = load_config(str(ROOT / config_path)).to_dict()
    model = TriModalDFINE(config, device=device)
    return model, config


@requires_weights
def test_model_build_and_weights():
    model, config = _build_model()
    core = model.core
    assert core.decoder.num_classes == NUM_CLASSES
    # 分类头随机初始化 (80→12), 其余全保留预训练: 加载统计中无未预期键
    assert not core.weights_info["unexpected"] if "unexpected" in core.weights_info else True
    excluded = core.weights_info.get("excluded", [])
    assert any("dec_score_head" in key for key in excluded)
    # 恒等初始化: Tri-CSSA residual_scale == 0
    for fusion in core.fusions:
        assert float(fusion.residual_scale) == 0.0
    # 逐角色参数存在
    roles = {model.parameter_role(name) for name, _ in model.named_parameters()}
    assert {"rgb", "detector", "classifier", "new"} <= roles


@requires_weights
def test_model_train_loss_and_backward():
    model, _ = _build_model()
    model.train()
    model.set_epoch(0)
    batch = _make_batch()
    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss), f"loss={loss}"
    assert metrics["detection"] >= 0
    assert "relation" in metrics
    loss.backward()
    # 梯度到达新分支与关系模块
    grad_names = {name for name, param in model.named_parameters() if param.grad is not None}
    assert any(name.startswith("core.ir_encoder") for name in grad_names)
    assert any(name.startswith("core.fusions") for name in grad_names)
    assert any(name.startswith("relation_refiner") for name in grad_names)
    # 回归断言: geo_mlp 末层梯度非零 (ReLU × 末层零初始化 = 梯度死锁, 已修复)
    geo_grad = model.relation_refiner.geo_mlp[-1].weight.grad
    assert geo_grad is not None and float(geo_grad.abs().sum()) > 0
    # 分类头参与 (Phase 默认从 epoch 0 起可训)
    assert any("dec_score_head" in name for name in grad_names)


@requires_weights
def test_model_predict_format():
    model, _ = _build_model()
    model.eval()
    model.set_epoch(4)
    with torch.no_grad():
        outputs = model.predict_raw(_make_batch(num_gt=0))
    assert outputs.shape == (1, 300, 6)
    boxes, scores, labels = outputs[0, :, :4], outputs[0, :, 4], outputs[0, :, 5]
    assert boxes.min() >= 0 and boxes.max() <= 1
    assert scores.min() >= 0 and scores.max() <= 1
    assert labels.min() >= 0 and labels.max() < NUM_CLASSES
    # 分数降序 (官方 postprocessor top-k 语义)
    assert torch.equal(scores, scores.sort(descending=True).values)


@requires_weights
def test_model_phases_and_optimizer():
    model, config = _build_model()
    # Phase 1: 冻结 backbone + encoder/decoder
    model.set_phase(1, config)
    for name, param in model.named_parameters():
        role = model.parameter_role(name)
        if role == "rgb":
            assert not param.requires_grad, name
        elif role == "detector":
            assert not param.requires_grad, name
    assert not model.core.backbone.training  # 冻结 backbone → eval (BN 统计固定)
    assert not model.core.encoder.training
    assert model.core.decoder.training  # decoder 必须保持 train (官方 criterion 要求)
    optimizer = build_optimizer(model, config["train"]["phases"][0])
    assert len(optimizer.param_groups) >= 2

    # model.train() 后冻结 BN 状态保持 (v7 修复)
    model.train()
    assert not model.core.backbone.training

    # Phase 3: 全解冻 (up/reg_scale 为官方定义不可训练的常量参数, 永远 requires_grad=False)
    model.set_phase(3, config)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any(name.startswith("core.backbone") for name in trainable)
    assert any(name.startswith("core.encoder") for name in trainable)
    assert any("dec_bbox_head" in name for name in trainable)
    assert all(param.requires_grad for param in model.relation_refiner.parameters())


@requires_weights
def test_rgb_only_backbone_branch():
    model, _ = _build_model()
    core = model.core
    batch = _make_batch()
    # rgb_only 分支: 包装器 mode 切换后直接透传官方 backbone 输出
    core.multimodal_backbone.mode = "rgb_only"
    features, auxiliary = core.multimodal_backbone(batch)
    assert len(features) == 3
    assert auxiliary == {}
    assert features[0].shape[-1] == batch["img"].shape[-1] // 8  # stride 8
    core.multimodal_backbone.mode = "tri"


@requires_weights
def test_nonsquare_letterbox_coordinates():
    """回归: 官方 postprocessor 的 orig_target_sizes 行约定是 (W, H) (x 用 s0
    缩放)。传 (H, W) 时非方形 letterbox 的 x 被按 H/W 压缩 —— Phase 2 (1024×576)
    val mAP 全零的直接原因; 正方形 (Phase 1 640×640) 交换不可见所以幸免。"""
    model, _ = _build_model()
    model.eval()
    pred_logits = torch.full((1, 300, NUM_CLASSES), -10.0)
    pred_logits[0, 5, 3] = 10.0  # 唯一高置信 query: 类别 3
    boxes = torch.full((1, 300, 4), 0.1)
    boxes[0, 5] = torch.tensor([0.9, 0.5, 0.2, 0.3])  # cx=0.9 靠右, 压缩 bug 下会变成 0.9*H/W
    orig_sizes = torch.tensor([[160, 128]])  # (W, H) 非方形
    with torch.no_grad():
        out = model.core.to_predictions(pred_logits, boxes, orig_sizes)
    assert out.shape == (1, 300, 6)
    assert int(out[0, 0, 5]) == 3 and float(out[0, 0, 4]) > 0.99  # top-1 = 高置信 query
    cx, cy, w, h = out[0, 0, :4].tolist()
    assert abs(cx - 0.9) < 1e-3, f"cx={cx} (非方形 (W,H) 约定被破坏)"
    assert abs(cy - 0.5) < 1e-3 and abs(w - 0.2) < 1e-3 and abs(h - 0.3) < 1e-3


@requires_weights
def test_eval_outputs_and_postprocess():
    model, _ = _build_model()
    model.eval()
    with torch.no_grad():
        outputs = model.core.eval_outputs(_make_batch(num_gt=0))
    assert "pred_logits" in outputs and "pred_boxes" in outputs and "pred_features" in outputs
    assert outputs["pred_logits"].shape == (1, 300, NUM_CLASSES)
    assert outputs["pred_boxes"].shape == (1, 300, 4)
    predictions = model.core.to_predictions(outputs["pred_logits"], outputs["pred_boxes"], outputs["orig_sizes"])
    assert predictions.shape == (1, 300, 6)


# ─────────────────────────── data pipeline ───────────────────────────

def test_dataset_pipeline_real_data():
    config = load_config(str(ROOT / "configs" / "base.yaml")).to_dict()
    split_dir = Path(config["data"]["split_dir"])
    val_ids_file = split_dir / f"fold_{config['data']['fold']}_val.txt"
    if not val_ids_file.exists():
        pytest.skip("split 文件缺失")
    ids = [line.strip() for line in val_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()][:2]
    dataset = MultiModalDataset(config["data"]["train_root"], ids, (128, 128), False, config["data"], {})
    assert len(dataset) == len(ids)
    batch = collate_fn([dataset[0], dataset[1]])
    assert set(batch) >= {"img", "ir", "depth", "depth_valid", "boxes", "labels", "num_boxes", "image_ids", "meta"}
    assert batch["img"].shape == (2, 3, 128, 128)
    assert batch["ir"].shape == (2, 1, 128, 128)
    assert batch["depth"].shape == (2, 3, 128, 128)
    assert batch["num_boxes"].sum() > 0  # 真实标注非空
    meta = batch["meta"][0]
    assert meta.original_hw[0] > 0 and meta.output_hw == (128, 128)


# ─────────────────────────── checkpoint / submission ───────────────────────────

def test_checkpoint_roundtrip(tmp_path):
    module = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, module, optimizer, scheduler, None, None, 3, 10, 0.5, {"k": 1}, "hash")
    module2 = torch.nn.Linear(4, 4)
    state = load_checkpoint(path, module2, None, None, None, None, expected_split_hash="hash")
    assert state["epoch"] == 3
    assert torch.equal(module.weight, module2.weight)
    with pytest.raises(ValueError):
        load_checkpoint(path, module2, None, None, None, None, expected_split_hash="other")


def test_split_fingerprint_stable():
    assert split_fingerprint(["a", "b"], ["c"]) == split_fingerprint(["b", "a"], ["c"])
    assert split_fingerprint(["a", "b"], ["c"]) != split_fingerprint(["a", "b"], ["d"])


def test_submission_roundtrip(tmp_path):
    write_prediction(tmp_path / "00000001.txt", [[0.5, 0.5, 0.2, 0.2]], [0.9], [3])
    errors = validate_submission(tmp_path, ["00000001"], num_classes=12, max_detections=100)
    assert errors == []
    write_prediction(tmp_path / "00000001.txt", [[0.5, 0.5, 2.0, 0.2]], [0.9], [3])
    assert validate_submission(tmp_path, ["00000001"], num_classes=12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
