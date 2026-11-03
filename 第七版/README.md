# 第七版：R-D-FINE —— 关系感知的三模态 D-FINE 检测架构

> AIC2026 面向城市场景的视觉多模态目标检测
>
> **第七版 = D-FINE（HGNetv2 + AIFI/CCFF + OQD 解码器 + FDR 细粒度分布回归，COCO 预训练）
> + 第三版 Tri-CSSA 融合（残差恒等 + 连续质量先验）+ 第六版注入点与训练管线
> + 全新 Relation-Aware Refiner（共现先验 + 空间先验 + 几何关系精炼，前六版首次目标间建模）**

完整设计见 [`第七版架构设计.md`](第七版架构设计.md)。

## 代码结构

```
第七版/
├── 第七版架构设计.md          # 设计文档 (§1-§7)
├── third_party/D-FINE/        # vendored 官方仓库 (仅模型定义, Apache-2.0; 补丁见 VENDORED.md)
├── weights/                   # dfine_l_coco.pth (L) / dfine_x_coco.pth (X), hf-mirror 下载
├── priors/relation_priors.npz # 关系先验 (build_priors.py 生成, 仅官方训练集统计)
├── configs/
│   ├── base.yaml              # D-FINE-L + Tri-CSSA + Relation Refiner (默认)
│   ├── cpu_mini.yaml          # CPU 三阶段迷你验证 (128×128)
│   ├── remote_westd.yaml      # AutoDL/5090D 路径覆盖
│   └── experiments/           # E0-E8 消融配置 (见设计文档 §5)
├── scripts/
│   ├── train.py               # 三阶段训练 (v6 管线迁移)
│   ├── validate.py            # 验证 mAP
│   ├── infer.py               # 测试集推理 + 提交
│   ├── build_priors.py        # 共现矩阵 + 空间高斯统计 (仅官方训练集)
│   └── make_split.py          # 当前数据集布局的五折划分
└── src/aic_v7/                # 自包含实现 (不依赖第六版)
    ├── config.py / constants.py
    ├── data/                  # v6 整包迁移 (dict batch, 无 COCO JSON 转换)
    ├── engine/                # trainer / evaluator / checkpoint (v6 迁移)
    ├── metrics/               # detection_map (v6 迁移)
    ├── inference/             # submission 管线 (v6 迁移)
    └── models/
        ├── dfine_adapter.py   # ★ 官方模型构建 + 权重加载 + 融合注入 + batch 适配
        ├── multimodal_dfine.py# 主包装器 (loss/相位/参数角色, v6 同位替换)
        ├── relation_refiner.py# ★ Relation-Aware Refiner (logit 空间可微 + warmup + varifocal 监督)
        └── tri_cssa / cmtf / modality_encoders / target_refinement / blocks / ema  # v6 迁移
```

## 快速开始

```bash
# 0. 准备 (一次)
#    - 下载权重: hf-mirror.com/Peterande/D-FINE/resolve/main/dfine_l_coco.pth → weights/
#    - 生成划分: python scripts/make_split.py --train-root <训练集> --output outputs/splits
#    - 生成先验: python scripts/build_priors.py \
#          --labels-dir <训练集>/new_labels_2000 --output priors/relation_priors.npz

# 1. CPU 冒烟 (三阶段全流程, 每阶段 1 epoch)
python scripts/train.py --config configs/cpu_mini.yaml

# 2. 正式训练 (5090D)
python scripts/train.py --config configs/base.yaml --device cuda:0
python scripts/train.py --overlay configs/remote_westd.yaml --device cuda:0

# 3. 验证 / 推理提交
python scripts/validate.py <checkpoint.pt> --device cuda:0
python scripts/infer.py <checkpoint.pt> --output outputs/submissions/predictions --tta --device cuda:0

# 4. 消融 (E0-E8)
python scripts/train.py --config configs/experiments/rgb_only.yaml        # E0
python scripts/train.py --config configs/experiments/no_relation.yaml     # E2-E4 基线
python scripts/train.py --config configs/experiments/cooc_only.yaml       # E2
python scripts/train.py --config configs/experiments/spatial_only.yaml    # E3
python scripts/train.py --config configs/experiments/relation_app.yaml    # E4
python scripts/train.py --config configs/experiments/cmtf.yaml            # E5
python scripts/train.py --config configs/experiments/dfine_x.yaml         # E6
python scripts/train.py --config configs/experiments/no_conditional_dropout.yaml  # E7
python scripts/train.py --config configs/experiments/no_refiner.yaml      # E8
```

## 测试

```bash
python -m pytest tests/test_smoke.py -v   # CPU 冒烟 (模型类测试需要 weights/dfine_l_coco.pth)
```

## 说明

- 第七版数据管线与第六版一致（dict batch `{img, ir, depth, depth_valid}`），官方仓库只
  vendored 模型定义；训练循环、评估、提交全部为 v7 自有实现。
- 模型接口契约与 v6 相同（`loss`/`predict_raw`/`set_phase`/`parameter_role`），
  evaluator 与提交管线零改动复用。
- `model.base: rtdetr` 可回退到第六版 RT-DETR 实现（需第六版代码可用）。
