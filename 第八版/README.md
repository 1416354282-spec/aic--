# 第八版：局部可靠性三模态 RT-DETR

本版以远端已完成训练的第六版稳定源码为工程基础。检测器仍为 RT-DETR，默认融合改为 `ReliabilityFusion`。第七版文档中的 D-FINE 在代码里只有回退适配器，未真正集成，因此这里不把它当作已验证底座。

## 核心改动

- 修复 HHA 数据管线：`depth_valid` 现在来自传感器深度的真实有效范围（300–19999 mm）；JPG 相对深度视作有效，letterbox/仿射填充区域无效。旧版错误地把 HHA 的高度通道当作有效掩码。
- 三个尺度各自根据 RGB、IR、Depth 特征和局部深度有效率预测空间模态分数。无效深度位置被硬屏蔽，部分有效区域按有效比例降权。
- 继承 Tri-CSSA 的通道评分、空间注意力与零残差初始化。全有效且局部卷积零初始化时，与原 Tri-CSSA 前向一致；训练初始仍保持预训练 RGB 路径。
- 保留远端经过训练验证的 FP32 前向、模态特征梯度限幅及梯度/参数有限性保护。

## 论文依据

| 论文 | 来源 | 本版采用的思想 |
|---|---|---|
| C²Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection | IEEE TGRS 2024, DOI: [10.1109/TGRS.2024.3376819](https://doi.org/10.1109/TGRS.2024.3376819) | 跨模态特征需按位置校准、互补；本版用轻量空间评分，不复现其 Transformer。 |
| Improving RGB-infrared object detection with cascade alignment-guided transformer | Information Fusion 2024, DOI: [10.1016/j.inffus.2024.102246](https://doi.org/10.1016/j.inffus.2024.102246) | 空间差异会影响融合；本版利用赛题已配准输入及深度有效掩码，做局部可靠性控制。 |

这些论文提供设计方向，并不构成本数据集上的精度证明。新模块需要同一训练/验证划分下的消融实验验证收益。

## 运行

依赖见 `requirements.txt`，还需离线准备与所选配置相符的 RT-DETR 权重。24GB RTX 4090 的独立验证实验：

```bash
python scripts/train.py --overlay configs/experiments/v8_4090_fold.yaml
python scripts/train.py --overlay configs/experiments/v8_4090_fold.yaml --overlay configs/experiments/tri_cssa.yaml
```

两组实验均使用原来的 1600/400 划分和 `new_labels_2000` 标注。全量 2000 张训练的远端第六版 `map50_95=0.867` 来自与训练重叠的验证集，不能用于比较或选模型。测试集推理仍用 `scripts/infer.py`，输出契约不变。

CPU 单元测试需额外安装 `pytest`：`python -m pytest tests/test_reliability.py`。完整训练和分数尚未在当前无 GPU 的远端实例上运行。

2026-09-28 新 RTX 5090 实例仅有官方 `labels/`，运行时在上述配置后再叠加 `configs/experiments/v8_5090_fold.yaml`，其结果与采用 `new_labels_2000` 的旧实验不可直接比较。
