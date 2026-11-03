#!/usr/bin/env bash
# 分块上传 last.pt 到服务器并组装校验
set -u
HOST="root@connect.westd.seetacloud.com"
PORT=20679
LOCAL_PT="D:/AIC面对城市场景的视觉多模态目标检测/第六版/outputs/runs_x/last.pt"
REMOTE_PT="/root/aic/v6/第六版/outputs/runs_x_local/weights/last.pt"
REMOTE_PARTS="/root/aic/v6/第六版/outputs/runs_x_local/parts"

cd /tmp && rm -rf aic_upload_parts && mkdir aic_upload_parts && cd aic_upload_parts
echo "== 分块"
split -b 150M "$LOCAL_PT" last.pt.part_
ls -la last.pt.part_* | awk '{print $9, $5}'

ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=no $HOST "rm -rf $REMOTE_PARTS && mkdir -p $REMOTE_PARTS"

for part in last.pt.part_*; do
  local_size=$(stat -c %s "$part" 2>/dev/null || stat -f %z "$part")
  for attempt in 1 2 3 4 5; do
    remote_size=$(ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=no $HOST "stat -c %s $REMOTE_PARTS/$part 2>/dev/null || echo 0")
    if [ "$remote_size" = "$local_size" ]; then
      echo "== $part 已存在 ($remote_size B), 跳过"
      break
    fi
    echo "== 上传 $part (本地 $local_size, 远程 $remote_size) 第 $attempt 次"
    if scp -P $PORT -o StrictHostKeyChecking=no -o BatchMode=yes "$part" $HOST:$REMOTE_PARTS/$part; then
      remote_size=$(ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=no $HOST "stat -c %s $REMOTE_PARTS/$part 2>/dev/null || echo 0")
      if [ "$remote_size" = "$local_size" ]; then
        echo "== $part OK"
        break
      fi
    fi
    sleep 3
  done
done

echo "== 组装"
ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=no $HOST "cat $REMOTE_PARTS/last.pt.part_* > $REMOTE_PT && rm -rf $REMOTE_PARTS && md5sum $REMOTE_PT"
cd /tmp && rm -rf aic_upload_parts
