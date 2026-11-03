#!/usr/bin/env bash
# 分块下载 coco_corrected_ep103_all_100epochs (20261004_163628, best epoch 88) 的 best.pt。
# 逐块 + 整体 md5 校验, 断线重跑自动跳过已校验块。
set -u

HOST="root@connect.westc.seetacloud.com"
PORT=51287
RDIR="/root/v8dl_parts"
LOCAL_DIR="D:/AIC面对城市场景的视觉多模态目标检测/第八版/outputs/runs_corrected_all100/weights"
TMP="$LOCAL_DIR/.parts"
WHOLE_MD5="929291b78c732c5ec2cb12d3c09a76bc"
WHOLE_SIZE=1574928757

declare -A MD5=(
  [00]=4bfd0fb43ecdc36b94ee26726983921c
  [01]=e0a9ba68457f48f2ec7432b14172f298
  [02]=f439cb540959a22385179594a44246b9
  [03]=2a7e5c1efaaf7d07c827364c454671e3
  [04]=1f8053af18189f2037aab4b7d502c783
  [05]=6afa3e464c5227d07310fbc46594cd34
  [06]=4931e633747a1227524f6732c18bd5ac
  [07]=bc0870dbda16791b718e43c0ec91cb36
  [08]=c73a57a473f581ff4cc4e9620fb1a443
  [09]=77b6e86940bdb73b4046d63d36f93a17
  [10]=21f09c8164693018c15296e2a2781563
  [11]=3335cd424210e504ac345c2dac99bd40
  [12]=78c25d55b422a7a0a83073c9bc549cdd
  [13]=4eeb69c28482311868549e8abbee9e2c
  [14]=38c0a6e8fb295804cdf76c7b6d98b7b0
  [15]=9dc7156143184595bcc42b9d0a5d5fe7
)

mkdir -p "$TMP"

local_md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }

for key in $(echo "${!MD5[@]}" | tr ' ' '\n' | sort); do
  part="best.pt.part_$key"
  want="${MD5[$key]}"
  dest="$TMP/$part"
  got=""
  [ -f "$dest" ] && got="$(local_md5 "$dest")"

  if [ "$got" = "$want" ]; then
    echo "== $part 已存在且校验通过, 跳过"
    continue
  fi

  ok=0
  for attempt in 1 2 3 4 5 6 7 8; do
    echo "== 下载 $part 第 $attempt 次"
    if scp -P "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=25 \
        "$HOST:$RDIR/$part" "$dest" 2>&1 | tail -1; then
      got="$(local_md5 "$dest")"
      if [ "$got" = "$want" ]; then
        echo "   $part OK"
        ok=1
        break
      fi
      echo "   md5 不符 (得到 $got, 期望 $want), 重试"
      rm -f "$dest"
    fi
    sleep 5
  done

  if [ "$ok" != "1" ]; then
    echo "!! $part 8 次均失败, 中止"
    exit 1
  fi
done

echo "== 组装"
out="$LOCAL_DIR/best.pt"
cat "$TMP"/best.pt.part_* > "$out"
size=$(stat -c %s "$out" 2>/dev/null || stat -f %z "$out")
got="$(local_md5 "$out")"
echo "   大小 $size (期望 $WHOLE_SIZE)"
echo "   md5  $got (期望 $WHOLE_MD5)"
if [ "$size" = "$WHOLE_SIZE" ] && [ "$got" = "$WHOLE_MD5" ]; then
  echo "*** best.pt 下载完成且校验通过 ***"
  rm -rf "$TMP"
  ssh -p "$PORT" -o StrictHostKeyChecking=no "$HOST" "rm -rf $RDIR" && echo "远端分块已清理"
else
  echo "!! 整体校验失败"
  exit 1
fi
