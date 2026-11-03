#!/bin/bash
# 推送第七版到远程 5090D/AutoDL（代码 + 配置 + 权重 + 先验）
# 用法: 修改下方 REMOTE/PORT/KEY 后执行 bash sync_to_5090d.sh
set -e

REMOTE="root@connect.westd.seetacloud.com"   # 5090D (westd)
PORT="49908"                                  # 5090D (westd)
KEY="$HOME/.ssh/id_rsa"
LOCAL="$(dirname "$0")"
REMOTE_PATH="/root/aic/v7"

SCP="scp -o StrictHostKeyChecking=no -i $KEY -P $PORT"
SSH="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT"

echo "=== 推送第七版 → ${REMOTE}:${REMOTE_PATH} ==="

$SSH "$REMOTE" "mkdir -p $REMOTE_PATH"

# 代码
$SCP -r "$LOCAL/src" "$LOCAL/scripts" "$LOCAL/configs" "$LOCAL/tests" \
    "$LOCAL/requirements.txt" "$LOCAL/README.md" "$LOCAL/第七版架构设计.md" \
    "$REMOTE:$REMOTE_PATH/" && echo "code ✓"

# vendored D-FINE (仅模型定义)
$SCP -r "$LOCAL/third_party/D-FINE" "$REMOTE:$REMOTE_PATH/third_party/" && echo "third_party ✓"

# 权重 (126MB L / 252MB X, 按需注释)
$SCP "$LOCAL/weights/dfine_l_coco.pth" "$REMOTE:$REMOTE_PATH/weights/" && echo "dfine_l_coco.pth ✓"
# $SCP "$LOCAL/weights/dfine_x_coco.pth" "$REMOTE:$REMOTE_PATH/weights/" && echo "dfine_x_coco.pth ✓"

# 关系先验 (远程重新生成亦可: scripts/build_priors.py --labels-dir ... )
$SCP "$LOCAL/priors/relation_priors.npz" "$REMOTE:$REMOTE_PATH/priors/" && echo "priors ✓"

echo "=== 完成: ${REMOTE}:${REMOTE_PATH} ==="
