#!/bin/bash
# 同步第六版代码到远程训练服务器 (参照第五版 sync_to_5090d.sh)
# 用法: bash sync_to_5090d.sh
#
# 当前远端 (2026-09-27 起): AutoDL / RTX 4090 24GB / sm_89 / 驱动 570.124.04
#   已确认: torch 2.8.0+cu128 可用 (matmul 通过), miniconda base = Python 3.12.3
#   ⚠ 这台不是 5090D。显存 24GB (原 32GB) → Phase 3 @1536×864 的 batch 需重新探针确认。

REMOTE_HOST="root@connect.westb.seetacloud.com"
REMOTE_PORT="10673"
REMOTE_PATH="/root/aic/v6/第六版"
KEY="$HOME/.ssh/id_rsa"
LOCAL="$(dirname "$0")"

SCP="scp -o StrictHostKeyChecking=no -i $KEY -P $REMOTE_PORT"
SSH="ssh -o StrictHostKeyChecking=no -i $KEY -p $REMOTE_PORT"

echo "=== 确保远程目录存在 ==="
$SSH "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'" && echo "目录 ✓"

echo "=== 同步 src/ configs/ scripts/ tests/ 到远程 ==="
$SCP -r \
    "$LOCAL/src/" \
    "$LOCAL/configs/" \
    "$LOCAL/scripts/" \
    "$LOCAL/tests/" \
    "$LOCAL/requirements.txt" \
    "$LOCAL/pyproject.toml" \
    "$REMOTE_HOST:$REMOTE_PATH/" && echo "代码 ✓"

echo "=== 同步 RT-DETR 权重 (L=66MB, X=135MB, 离线加载) ==="
$SCP "$LOCAL/../rtdetr-l.pt" "$REMOTE_HOST:/root/aic/rtdetr-l.pt" && echo "rtdetr-l ✓"
$SCP "$LOCAL/../rtdetr-x.pt" "$REMOTE_HOST:/root/aic/rtdetr-x.pt" && echo "rtdetr-x ✓"

echo "=== 同步 splits (fold_0/1/2) ==="
$SSH "$REMOTE_HOST" "mkdir -p /root/aic/v6/第二版-outputs/splits"
$SCP -r "$LOCAL/../第二版/outputs/splits/"* "$REMOTE_HOST:/root/aic/v6/第二版-outputs/splits/" && echo "splits ✓"

echo "=== 同步 splits_full (全量重训用: fold_0_train = 2000 张) ==="
$SSH "$REMOTE_HOST" "mkdir -p /root/aic/v6/第二版-outputs/splits_full"
$SCP -r "$LOCAL/../第二版/outputs/splits_full/"* "$REMOTE_HOST:/root/aic/v6/第二版-outputs/splits_full/" && echo "splits_full ✓"

echo ""
echo "=== 远程训练命令 (5090D) ==="
echo "  ssh -i $KEY -p $REMOTE_PORT $REMOTE_HOST"
echo "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate base"
echo "  cd $REMOTE_PATH"
echo "  # 首次: pip install ultralytics   (依赖 GPU 版 torch, 已装则跳过)"
echo "  # E0 (RGB-only 基线):"
echo "  python scripts/train.py --config configs/base.yaml --overlay configs/experiments/rgb_only.yaml --overlay configs/remote_westd.yaml"
echo "  # E1 (三模态 Tri-CSSA):"
echo "  python scripts/train.py --overlay configs/remote_westd.yaml"
echo "  # 观察 2~3 个 epoch 后 Ctrl+C, 检查 outputs/runs/*/metrics.jsonl 的 loss 曲线与显存"
echo ""
echo "  # ★ X 底座注意 (实测已确认): remote_westd.yaml 的 weights 是 rtdetr-l.pt,"
echo "  #   它在 runs_x_phase2.yaml 之后应用会把 X 权重覆盖成 L (只迁移 126/1241 项, backbone 随机)。"
echo "  #   用 X 底座时务必: --overlay configs/experiments/full_data.yaml  (内含 X 权重路径, 最后应用)"
echo "  #   或追加 --set model.weights=/root/aic/rtdetr-x.pt"
echo "  #"
echo "  # ★ 本轮最终模型 = 无增强 + 全量 2000 张 (顺序不可乱: full_data.yaml 必须最后):"
echo "  #   python scripts/train.py --config configs/base.yaml \\"
echo "  #     --overlay configs/experiments/runs_x_phase2.yaml \\"
echo "  #     --overlay configs/remote_westd.yaml \\"
echo "  #     --overlay configs/experiments/no_aug.yaml \\"
echo "  #     --overlay configs/experiments/full_data.yaml"
echo "  #   提交用 outputs/runs_full/*/weights/last.pt —— 这个 run 的 best.pt 是按泄漏指标存的, 不要用"
