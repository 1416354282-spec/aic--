# AIC2026 多模态目标检测：第二版

本目录是独立重构版，**不修改第一版代码和原始数据**。核心原则：先修正数据划分、框变换、解码和 NMS，再逐项做模型与增强实验。

## 目录

- `configs/`：基础、显存档位和单变量实验配置。
- `docs/`：旧版审查、数据报告、总体策略、实验矩阵、论文映射与检查清单。
- `scripts/`：审计、划分、可视化、训练、验证、推理、提交。
- `src/aic_v2/`：可复用的数据、模型、训练、指标和推理模块。
- `tests/`：不依赖正式训练的正确性测试。
- `outputs/`：机器生成的审计、图、split、训练和提交产物。

## 环境

当前 ZCode 终端没有发现 `python`、`py` 或 `conda`，因此本次交付未能执行 Python 测试。请在原计划的 `aic_multimodal` 环境中：

```bash
cd 第二版
python -m pip install -r requirements.txt
python scripts/smoke_test.py
pytest
```

必须记录实际 PyTorch、Torchvision、Ultralytics、CUDA 版本。模型输出结构随 Ultralytics 版本变化，正式训练/提交前必须完成 decode 对照测试。

## 1. 数据审计

```bash
python scripts/audit_data.py "../初赛数据集-面向城市场景的多模态目标检测/训练集/AIC2026_Train_2000" --output outputs/audit/train_manifest.jsonl
python scripts/audit_data.py "../初赛数据集-面向城市场景的多模态目标检测/测试集/AIC2026_PHASE_1_1000" --output outputs/audit/test_manifest.jsonl
python scripts/visualize_data.py outputs/audit/train_manifest.jsonl
```

哈希全量扫描较慢；开发时可加 `--no-hash`，正式划分必须保留哈希审计。

## 2. 防泄漏划分

```bash
python scripts/make_splits.py outputs/audit/train_manifest.jsonl --folds 3
```

划分按 group 约束，不允许明显序列或重复组跨 train/val。应检查每折类别框数，尤其 tricycle、ball、boat 和 uav。

## 3. 配置叠加

默认配置：`configs/base.yaml`。资源档位：

- `hardware_8gb.yaml`：640、batch 4、梯度累积 4；
- `hardware_12_24gb.yaml`：768、batch 8、梯度累积 2；
- `hardware_40gb_plus.yaml`：960、YOLOv8m、batch 16。

显式覆盖格式：

```bash
python scripts/smoke_test.py --overlay configs/hardware_12_24gb.yaml --set train.learning_rate=0.0003
```

CLI 未显式提供的值不会覆盖 YAML。不存在的本地权重会报错，不允许自动联网下载。

## 4. 推荐训练顺序

1. correctness baseline：letterbox、clip box、共享 decode、class-aware NMS；
2. 640/768/960；
3. YOLOv8s/m；
4. RGB、RGB+IR、RGB+Depth、三模态、Depth validity mask；
5. IR/Depth 编码；
6. 基础增强和 modal dropout；
7. Mosaic 与 rare Copy-Paste 分开测试；
8. Repeat-Factor Sampling 与长尾损失分开测试；
9. NMS/multi-label/分数校准；
10. 最后才做 gated stem 或中层融合。

禁止一次同时打开强类别权重、过采样和 rare Copy-Paste，否则无法归因且容易过拟合 25 个 tricycle 框。

## 5. 提交

推理应将 letterbox 框映射回原图归一化坐标。生成 TXT 后：

```bash
python scripts/make_submission.py outputs/submissions/predictions "../初赛数据集-面向城市场景的多模态目标检测/测试集/AIC2026_PHASE_1_1000" --zip outputs/submissions/submission.zip
```

脚本检查一图一文件、六列、类别/坐标/置信度、NaN/Inf、每图最多 100 框和 ZIP 根层结构。

## 当前实现状态

数据审计、group split、letterbox/仿射/框裁剪、Mosaic、rare Copy-Paste、数据集、Early Fusion、EMA/checkpoint、DFL decode、class-aware NMS、赛事 mAP 和提交校验模块已建立。由于当前终端无 Python，训练/验证/推理入口采用 fail-fast，避免在没有完成 Ultralytics 版本对照时静默生成错误结果。目标环境通过测试后，再把这些模块组装进正式长训练循环。
