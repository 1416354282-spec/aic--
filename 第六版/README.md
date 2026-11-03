# 第六版: TriRT-DETR —— 以 RT-DETR 为核心的三模态实时检测

架构设计见 `../第六版架构设计.md`。

## 环境

```bash
pip install -r requirements.txt   # 含 ultralytics (RT-DETR 官方集成)
```

## 权重（必须离线准备）

```bash
# github 直连被墙时用镜像代理 (已下载到项目根目录):
#   rtdetr-l.pt  66MB  (RT-DETR-L, 32M 参数, COCO 53.0 AP)
#   rtdetr-x.pt  135MB (RT-DETR-X, 65M 参数, COCO 54.8 AP)
curl -L -o rtdetr-l.pt "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt"
```

## 训练

```bash
# E0: RGB-only 基线
python scripts/train.py --overlay configs/experiments/rgb_only.yaml
# E1: 三模态 (Tri-CSSA 融合, 默认)
python scripts/train.py
# E2: CMTF 融合消融 (v5 跨模态注意力)
python scripts/train.py --overlay configs/experiments/cmtf.yaml
# E8: RT-DETR-X 底座
python scripts/train.py --overlay configs/experiments/rtdetr_x.yaml
# 泛化增强 (全增强 + 光度/擦除/mosaic, 见 ../第六版泛化能力提升方案.md)
python scripts/train.py --overlay configs/experiments/runs_x_phase2.yaml \
  --overlay configs/experiments/generalization.yaml
# 远程 5090D: 追加 --overlay configs/remote_westd.yaml (放在 generalization.yaml 之后)
# 断点续训:  --resume <run>/weights/last.pt --set train.phases.1.epochs=100
```

## 泛化增强（可选，默认关闭）

`configs/experiments/generalization.yaml` 把"每 epoch 一半样本无增强"改成"全增强"：

- 几何：`affine_probability 0.33→0.9`、`scale 0.20→0.45`、`translate 0.10→0.15`
- 光度：RGB/IR **同步**亮度/对比度/gamma 抖动（深度通道不动）
- 遮挡：随机擦除，三模态同步置 0，避开 GT 中心
- mosaic：4 图拼接，Phase 2 起生效，最后 20 epoch 关闭
- `augment_repeats: 2`：全增强下 epoch 长度与旧配置一致，epochs 不用改

默认配置里所有新旋钮都是 0 → 行为与旧版逐位相同。单测：

```bash
python tests/test_augmentation.py
```

## 深度质量先验（可选）

默认配置会修正标量质量（`q_depth` = 真实深度有效率、`q_ir` = 非零像素比例），
但仍保持 3 通道 HHA，旧 checkpoint 可加载。要启用"4 通道 + 空间质量图"：

```bash
python scripts/train.py --overlay configs/experiments/runs_x_phase2.yaml \
  --overlay configs/experiments/generalization.yaml \
  --overlay configs/experiments/depth_quality.yaml
```

`depth_in_channels: 4` 会改变 DepthEncoder 第一层形状，必须从 COCO 权重重训，
不能 resume 旧 run。设计与验收见 `../第六版深度质量先验修复.md`，单测：

```bash
python tests/test_depth_quality.py
```

## 验证 / 推理

```bash
python scripts/validate.py <checkpoint>          # 验证集 mAP@50-95
python scripts/infer.py <checkpoint> --tta       # 测试集提交 (无 NMS, 可选 TTA)
```

## CPU 冒烟

```bash
python tests/test_smoke.py                       # 模型前向/损失/残差恒等/phase
python scripts/train.py --config configs/cpu_mini.yaml   # 三阶段全流程迷你验证
```

## 关键约束

- 输入尺寸必须是 32 的倍数 (RT-DETR 约束)
- 分辨率 16:9 优先 (1280×736 / 1024×576 / 640×640)
- 权重为 AGPL-3.0 (ultralytics 发行), 比赛离线环境仅需权重文件本身
- 推理端到端无 NMS: RTDETRDecoder eval 直接输出 top-k 结果
