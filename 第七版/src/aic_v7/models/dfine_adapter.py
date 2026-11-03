"""dfine_adapter: D-FINE 官方模型构建 + 三模态融合注入 + batch 适配层.

遵循第五版包装 RF-DETR 的外部包模式: vendored 官方仓库 (`third_party/D-FINE`)
只提供模型定义 (HGNetv2 backbone / HybridEncoder / DFINETransformer (OQD+FDR) /
DFINECriterion / HungarianMatcher / DFINEPostProcessor), 数据管线、训练循环、
评估与提交全部使用 aic_v7 自有实现。

关键设计 (与设计文档 §3 对应):
  - 官方组件经 YAMLConfig 从官方 yml 构建, 构建前覆盖:
      num_classes=12 (共享给 decoder/criterion/postprocessor),
      remap_mscoco_category=False (避免 12 类被重映射到 COCO 80 类),
      eval_spatial_size=None (多分辨率训练下锚点/位置编码动态生成),
      HGNetv2.pretrained=False / freeze_norm=False / freeze_at=-1
      (不联网、保留 affine BN 以对齐检查点、冻结统一由 set_phase 管理)。
  - 权重加载: 官方检查点 {"model": state_dict}; 分类头 (dec_score_head /
    enc_score_head / denoising_class_embed) 因 80→12 随机初始化, 连同缓存的
    640×640 锚点 buffer 一起排除, 其余严格对应。
  - 融合注入点与 v6 同构: Tri-CSSA/CMTF 输出替换 backbone 的 S3/S4/S5,
    AIFI + CCFF + decoder 全部保留预训练权重。
  - RGB 输入按 ImageNet 统计归一化 (官方预训练分布); 整个前向强制 fp32
    (v5/v6 验证过的 AMP 数值稳定方案)。
  - 训练损失 = 官方 DFINECriterion (varifocal 分类 + FDR 分布精炼 + L1 + GIoU
    + denoising 对比训练) + λ_target × 目标掩码辅助损失 (F3/F4, v3 配置)。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.ops import box_convert

from .cmtf import CMTF
from .modality_encoders import DepthEncoder, IREncoder
from .target_refinement import TargetAwareRefiner, build_soft_box_targets, target_mask_loss
from .tri_cssa import TriCSSA

VENDORED_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "D-FINE"
DEFAULT_YAML = VENDORED_ROOT / "configs" / "dfine" / "dfine_hgnetv2_l_coco.yml"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 80→12 类头随机初始化: 排除官方检查点中形状不兼容的键 (v5/v6 同策略)
CLASSIFIER_KEY_PREFIXES = (
    "decoder.dec_score_head",
    "decoder.enc_score_head",
    "decoder.denoising_class_embed",
)
# 官方训练在 640×640 下缓存了锚点 buffer; 我们动态生成, 直接排除
ANCHOR_BUFFER_KEYS = ("decoder.anchors", "decoder.valid_mask")


def _ensure_vendored_importable() -> Path:
    """把 vendored 官方仓库加入 sys.path (官方代码以 `src` 为顶层包名)。"""
    path = str(VENDORED_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return VENDORED_ROOT


def _import_official():
    """导入官方模型定义 (触发 @register 注册), 返回命名空间。"""
    _ensure_vendored_importable()
    from src.core.yaml_config import YAMLConfig
    from src.nn.backbone import HGNetv2  # noqa: F401  (注册 backbone)
    from src.zoo.dfine import (  # noqa: F401  (注册 DFINE/decoder/criterion/matcher/postprocessor)
        DFINE,
        DFINECriterion,
        DFINEPostProcessor,
        DFINETransformer,
        HungarianMatcher,
        HybridEncoder,
    )

    return YAMLConfig


def build_official_model(config: Dict[str, Any]):
    """从官方 yml 构建 (backbone, encoder, decoder, criterion, postprocessor)。

    num_classes 等由 config["model"] 传入并在构建前覆盖到 yaml 全局配置。
    """
    YAMLConfig = _import_official()
    model_cfg = config.get("model", {})
    yaml_path = Path(model_cfg.get("dfine_yaml") or DEFAULT_YAML)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"D-FINE 官方配置不存在: {yaml_path}（third_party/D-FINE 未 vendored, "
            f"或 model.dfine_yaml 路径有误）"
        )
    cfg = YAMLConfig(str(yaml_path))
    yaml_cfg = cfg.yaml_cfg
    yaml_cfg["num_classes"] = int(model_cfg.get("num_classes", 12))
    yaml_cfg["remap_mscoco_category"] = False
    yaml_cfg["eval_spatial_size"] = None
    for key, value in (
        ("pretrained", False),   # 不联网下载 stage1 初始化 (随后加载完整检查点)
        ("freeze_norm", False),  # 保留 affine BN, 与官方检查点键名一致
        ("freeze_at", -1),       # 冻结策略统一由 set_phase 管理
    ):
        yaml_cfg["HGNetv2"][key] = value
    return cfg.model, cfg.criterion, cfg.postprocessor, yaml_path


def load_official_weights(module: nn.Module, weights: str) -> Dict[str, Any]:
    """加载官方检查点, 排除分类头与锚点缓存; 返回加载统计。"""
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"D-FINE 权重不存在: {weights}（先用 hf-mirror 下载 dfine_l_coco.pth 并离线保存）")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
    filtered = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(CLASSIFIER_KEY_PREFIXES) and key not in ANCHOR_BUFFER_KEYS
    }
    result = module.load_state_dict(filtered, strict=False)
    # unexpected = 官方权重中有、模型中没有的键 → 真正的架构不匹配
    if result.unexpected_keys:
        raise RuntimeError(f"权重加载存在未预期键 (官方权重与模型架构不匹配): {result.unexpected_keys[:8]}")
    # missing 的组成: (a) 被排除的分类头/锚点 (80→12 随机初始化, 预期);
    # (b) 模型新增分支 (IR/Depth 编码器/融合/Refiner 等, 官方权重中本就不存在) — 均属正常
    return {
        "excluded": sorted({key for key in state_dict if key.startswith(CLASSIFIER_KEY_PREFIXES) or key in ANCHOR_BUFFER_KEYS}),
        "loaded": len(filtered),
    }


def modality_quality(infrared: torch.Tensor, depth_valid: torch.Tensor) -> torch.Tensor:
    """连续逐样本模态质量 (B, 3), 与 v6 一致: q_rgb=1, q_ir=非零占比, q_depth=有效掩码均值。"""
    batch = infrared.shape[0]
    rgb_quality = torch.ones(batch, device=infrared.device, dtype=infrared.dtype)
    ir_quality = (infrared.abs().mean((1, 2, 3)) > 1e-6).to(infrared.dtype)
    depth_quality = depth_valid.flatten(1).mean(1)
    return torch.stack((rgb_quality, ir_quality, depth_quality), dim=1)


class MultiModalBackbone(nn.Module):
    """RGB backbone (官方 HGNetv2) + IR/Depth 编码器 + 逐尺度融合 + Target-Aware Refiner。

    输出与官方 backbone 相同: List[Tensor] 三个尺度 (stride 8/16/32), 供官方
    HybridEncoder 直接消费 (与 v6 的注入点语义同构)。
    """

    def __init__(
        self,
        rgb_backbone: nn.Module,
        mode: str,
        fusion_name: str,
        ir_encoder: Optional[nn.Module],
        depth_encoder: Optional[nn.Module],
        fusions: nn.ModuleList,
        refiners: nn.ModuleDict,
        out_strides: Tuple[int, ...],
    ):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.mode = mode
        self.fusion_name = fusion_name
        self.ir_encoder = ir_encoder
        self.depth_encoder = depth_encoder
        self.fusions = fusions
        self.refiners = refiners
        self.out_strides = out_strides

    def forward(self, batch: Dict[str, torch.Tensor]):
        rgb = batch["img"].float()
        mean = torch.tensor(IMAGENET_MEAN, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        rgb_norm = (rgb - mean) / std

        features = self.rgb_backbone(rgb_norm)
        auxiliary: Dict[str, Any] = {}
        if self.mode != "tri" or self.ir_encoder is None:
            return features, auxiliary

        infrared = batch["ir"].float()
        depth = batch["depth"].float()
        depth_valid = batch["depth_valid"].float()
        quality = modality_quality(infrared, depth_valid)
        ir_features = self.ir_encoder(infrared)
        depth_features = self.depth_encoder(depth)
        fused, mask_logits = [], []
        for index, (rgb_feature, ir_feature, depth_feature) in enumerate(
            zip(features, ir_features, depth_features)
        ):
            feature = self.fusions[index](rgb_feature, ir_feature, depth_feature, quality)
            refiner = self.refiners[str(index)] if str(index) in self.refiners else None
            if refiner is not None:
                feature, logits = refiner(feature)
                mask_logits.append(logits)
            fused.append(feature)
        auxiliary["mask_logits"] = mask_logits
        auxiliary["quality"] = quality
        return fused, auxiliary


class DFINECore(nn.Module):
    """官方 D-FINE 模型包装为 aic_v7 batch 契约 (v6 TriModalRTDETR 的同位替换)。

    训练态: loss(batch) → (total, metrics, decoder_out, targets)
    推理态: eval_outputs(batch) → {pred_logits, pred_boxes, pred_features, orig_sizes}
            to_predictions(...) → (B, 300, 6) 归一化 [cx,cy,w,h,score,cls] (无 NMS)
    """

    def __init__(self, config: Dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__()
        model_cfg = config["model"]
        self.mode = model_cfg.get("mode", "tri")
        self.fusion_name = model_cfg.get("fusion", "tri_cssa")
        self.device = device

        official_model, criterion, postprocessor, yaml_path = build_official_model(config)
        self.backbone = official_model.backbone
        self.encoder = official_model.encoder
        self.decoder = official_model.decoder
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.loss_config = config.get("loss", {})

        self.ir_encoder = None
        self.depth_encoder = None
        self.fusions = nn.ModuleList()
        self.refiners = nn.ModuleDict()
        self.out_strides = tuple(int(s) for s in model_cfg.get("out_strides", [8, 16, 32]))
        if self.mode == "tri":
            rgb_channels = self._probe_rgb_channels(int(model_cfg.get("probe_size", 256)))
            self.ir_encoder = IREncoder(
                rgb_channels,
                width=float(model_cfg.get("ir_width", 0.375)),
                attention_groups=int(model_cfg.get("attention_groups", 8)),
                out_strides=self.out_strides,
            )
            self.depth_encoder = DepthEncoder(
                rgb_channels,
                width=float(model_cfg.get("depth_width", 0.375)),
                out_strides=self.out_strides,
                in_channels=int(model_cfg.get("depth_in_channels", 3)),
            )
            fusion_cls = CMTF if self.fusion_name == "cmtf" else TriCSSA
            if fusion_cls is CMTF:
                self.fusions = nn.ModuleList(
                    CMTF(
                        rgb_channels[i],
                        self.ir_encoder.out_channels[i],
                        self.depth_encoder.out_channels[i],
                        nhead=int(model_cfg.get("cmtf_nhead", 2)),
                        use_cross_attn=bool(model_cfg.get("cmtf_cross_attn", True)),
                        use_lite=bool(model_cfg.get("cmtf_lite", False)),
                    )
                    for i in range(len(self.out_strides))
                )
            else:
                self.fusions = nn.ModuleList(
                    TriCSSA(rgb_channels[i], self.ir_encoder.out_channels[i], self.depth_encoder.out_channels[i])
                    for i in range(len(self.out_strides))
                )
            refinement_levels = {int(i) for i in model_cfg.get("target_refinement_levels", [0, 1])}
            self.refiners = nn.ModuleDict({str(i): TargetAwareRefiner(rgb_channels[i]) for i in refinement_levels})
        self.multimodal_backbone = MultiModalBackbone(
            self.backbone, self.mode, self.fusion_name,
            self.ir_encoder, self.depth_encoder, self.fusions, self.refiners, self.out_strides,
        )

        self.weights_info = load_official_weights(self, str(model_cfg.get("weights", "")))
        self.to(device)
        self.phase = 1
        self._freeze_backbone = False
        self._freeze_decoder = False

    # -- construction helpers -------------------------------------------------
    def _probe_rgb_channels(self, size: int) -> Tuple[int, ...]:
        """运行官方 backbone 一次, 读取融合三尺度通道数 (复用 v6 探测思路)。"""
        training = self.backbone.training
        self.backbone.eval()
        parameter = next(self.backbone.parameters())
        with torch.no_grad():
            features = self.backbone(torch.zeros(1, 3, size, size, device=parameter.device, dtype=parameter.dtype))
        self.backbone.train(training)
        channels = tuple(int(feature.shape[1]) for feature in features)
        if len(channels) != len(self.out_strides):
            raise RuntimeError(f"官方 backbone 输出 {len(channels)} 个尺度, 配置期望 {len(self.out_strides)}: {channels}")
        strides = tuple(size // int(feature.shape[2]) for feature in features)
        if strides != self.out_strides:
            raise RuntimeError(
                f"官方 backbone 实际 strides={strides} != 配置 out_strides={self.out_strides} "
                f"(通道={channels}); 检查 model.dfine_yaml 与 out_strides 是否匹配"
            )
        return channels

    # -- batch adaptation -------------------------------------------------------
    @staticmethod
    def _make_targets(batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """v6 batch → 官方 criterion 的 targets 列表 (归一化 cxcywh + 类别)。"""
        targets = []
        for index, count in enumerate(batch["num_boxes"].tolist()):
            targets.append({
                "boxes": batch["boxes"][index, : int(count)].float(),
                "labels": batch["labels"][index, : int(count)].long(),
            })
        return targets

    def _orig_sizes(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 官方 postprocessor 以 orig_target_sizes.repeat(1,2)=(s0,s1,s0,s1) 缩放
        # xyxy → x 用 s0、y 用 s1, 行约定为 (W, H)。传 (H, W) 时非方形 letterbox
        # 的 x 会被按 H/W 压缩 (1024×576 实测 cx 上限 0.5625 → val mAP 塌 0;
        # 640×640 正方形时交换不可见, Phase 1 因此幸免)。
        height, width = batch["img"].shape[-2:]
        return torch.tensor([[width, height]] * batch["img"].shape[0], device=batch["img"].device)

    # -- forward paths ---------------------------------------------------------
    def _forward_features(self, batch: Dict[str, torch.Tensor]):
        """三模态融合 → 官方 HybridEncoder (AIFI + CCFF)。"""
        features, auxiliary = self.multimodal_backbone(batch)
        features = self.encoder(features)
        return features, auxiliary

    def forward_train(self, batch: Dict[str, torch.Tensor]):
        """训练前向: 返回官方 decoder 输出 dict (含 aux/dn) 与融合辅助信息。"""
        targets = self._make_targets(batch)
        features, auxiliary = self._forward_features(batch)
        outputs = self.decoder(features, targets)
        return outputs, targets, auxiliary

    def eval_outputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """推理前向: 官方 decoder eval 输出 (pred_logits/pred_boxes/pred_features) + letterbox 尺寸。"""
        features, _ = self._forward_features(batch)
        outputs = self.decoder(features, None)
        return {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
            "pred_features": outputs.get("pred_features"),
            "orig_sizes": self._orig_sizes(batch),
        }

    def to_predictions(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        orig_sizes: torch.Tensor,
        num_top_queries: int = 300,
    ) -> torch.Tensor:
        """官方 postprocessor 输出的逐图结果 → (B, 300, 6) 归一化 [cx,cy,w,h,score,cls]。"""
        results = self.postprocessor(
            {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
            orig_sizes,
        )
        batch = len(results)
        out = torch.zeros(batch, num_top_queries, 6, device=pred_logits.device, dtype=pred_logits.dtype)
        for index, result in enumerate(results):
            labels, boxes, scores = result["labels"], result["boxes"], result["scores"]
            count = min(len(scores), num_top_queries)
            cxcywh = box_convert(boxes[:count], in_fmt="xyxy", out_fmt="cxcywh")
            width, height = int(orig_sizes[index, 0]), int(orig_sizes[index, 1])  # (W, H) 行约定
            scale = torch.tensor([width, height, width, height], device=cxcywh.device, dtype=cxcywh.dtype)
            normalized = (cxcywh / scale).clamp(0, 1)
            out[index, :count, :4] = normalized
            out[index, :count, 4] = scores[:count]
            out[index, :count, 5] = labels[:count].float()
        return out

    def loss(self, batch: Dict[str, torch.Tensor]):
        """训练损失: 官方 DFINECriterion + λ_target × 目标掩码辅助损失。"""
        outputs, targets, auxiliary = self.forward_train(batch)
        loss_dict = self.criterion(outputs, targets)
        total = sum(loss_dict.values())
        metrics = {"detection": total.detach()}
        for key, value in loss_dict.items():
            metrics[f"loss_{key}"] = value.detach()

        mask_config = self.loss_config.get("target_mask", {})
        if self.mode == "tri" and mask_config.get("enabled", True) and auxiliary.get("mask_logits"):
            shapes = [tuple(logits.shape[-2:]) for logits in auxiliary["mask_logits"]]
            batch_indices, flat_boxes = self._flatten_boxes(batch)
            soft_targets = build_soft_box_targets(
                batch_indices, flat_boxes, batch["img"].shape[0],
                shapes, batch["img"].device, batch["img"].dtype,
                float(mask_config.get("center_sigma", 0.35)),
            )
            mask_loss = target_mask_loss(
                auxiliary["mask_logits"], soft_targets,
                gamma=float(mask_config.get("focal_gamma", 2.0)),
                dice_weight=float(mask_config.get("dice_weight", 1.0)),
            )
            total = total + batch["img"].shape[0] * float(mask_config.get("weight", 0.1)) * mask_loss
            metrics["target_mask"] = mask_loss.detach()
        metrics["total"] = total.detach()
        return total, metrics, outputs, targets

    @staticmethod
    def _flatten_boxes(batch):
        """Convert padded (B, N, 4) boxes to v3-style flat (T, 4) + batch indices."""
        boxes, indices = [], []
        for i, count in enumerate(batch["num_boxes"].tolist()):
            for j in range(int(count)):
                boxes.append(batch["boxes"][i, j])
                indices.append(i)
        if not boxes:
            return torch.zeros(0, device=batch["boxes"].device, dtype=torch.long), batch["boxes"].reshape(-1, 4)[:0]
        return (
            torch.tensor(indices, device=batch["boxes"].device, dtype=torch.long),
            torch.stack(boxes),
        )

    # -- training controls ------------------------------------------------------
    def parameter_role(self, name: str) -> str:
        if name.startswith("backbone."):
            return "rgb"
        if name.startswith(("encoder.", "decoder.")):
            if name.startswith("decoder.") and any(
                key in name for key in ("dec_score_head", "enc_score_head", "denoising_class_embed")
            ):
                return "classifier"
            return "detector"
        return "new"

    def set_phase(self, phase: int, config: dict):
        """冻结策略 (与 v6 同语义)。注意: decoder 必须保持 train 模式 (官方
        criterion 要求 aux_outputs/dn 输出), 且 decoder 只有 LayerNorm 无 BN
        统计漂移问题, 冻结只需 requires_grad=False; backbone/encoder 冻结时切
        eval 模式以固定 BN 统计 (v6 已验证方案)。
        """
        self.phase = phase
        phases = config.get("train", {}).get("phases", [])
        spec = phases[phase - 1] if 0 < phase <= len(phases) else {}
        freeze_backbone = bool(spec.get("freeze_backbone", phase == 1))
        freeze_decoder = bool(spec.get("freeze_decoder", phase == 1))
        freeze_classifier = bool(spec.get("freeze_classifier", False))
        for name, param in self.named_parameters():
            if param.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                continue  # 官方 reg_scale 等非浮点参数不可训练
            role = self.parameter_role(name)
            if role == "rgb":
                param.requires_grad_(not freeze_backbone)
            elif role == "detector":
                param.requires_grad_(not freeze_decoder)
            elif role == "classifier":
                param.requires_grad_(not freeze_classifier)
            else:
                param.requires_grad_(True)
        self._freeze_backbone = freeze_backbone
        self._freeze_decoder = freeze_decoder
        self._apply_frozen_eval()

    def _apply_frozen_eval(self):
        """冻结模块切 eval 固定 BN 统计; decoder 例外 (官方 criterion 要求
        train 模式输出 aux/dn, 且 decoder 只有无统计的 LayerNorm)。
        在 model.train() 之后重调, 保证冻结状态跨 epoch 持续生效 (v6 经验)。
        """
        if self._freeze_backbone:
            self.backbone.eval()
        if self._freeze_decoder:
            self.encoder.eval()
            # decoder 保持 train() (criterion assert aux_outputs); 参数已冻结
