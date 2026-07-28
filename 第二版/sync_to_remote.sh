#!/bin/bash
# 同步本地改动到远程 5090D 训练服务器
# 用法: bash sync_to_remote.sh

REMOTE_HOST="root@connect.weste.seetacloud.com"
REMOTE_PORT="13671"
REMOTE_PATH="/root/aic/v2/第二版"

echo "=== 同步 src/ configs/ scripts/ 到远程 ==="
scp -P "$REMOTE_PORT" -r \
    src/ \
    configs/ \
    scripts/ \
    "$REMOTE_HOST:$REMOTE_PATH/"

echo "=== 同步完成 ==="
echo ""
echo "训练命令:"
echo "  ssh -p $REMOTE_PORT $REMOTE_HOST"
echo "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate base"
echo "  cd $REMOTE_PATH"
echo "  python scripts/train.py --config configs/base.yaml --overlay configs/hardware_5090d.yaml"
