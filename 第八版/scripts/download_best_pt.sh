#!/usr/bin/env bash
# 分块下载 v8 best.pt 并逐块 + 整体校验 md5。
# 断线可重跑: 已下载且 md5 正确的块会跳过。
set -u

HOST="root@connect.weste.seetacloud.com"
PORT=32174
RDIR="/root/v8dl_parts"
LOCAL_DIR="D:/AIC面对城市场景的视觉多模态目标检测/第八版/outputs/runs_1600_400_revised_phase3_batch3/weights"
TMP="$LOCAL_DIR/.parts"
WHOLE_MD5="90bbb5402b299433640e193117816ff2"
WHOLE_SIZE=1574522989

declare -A MD5=(
  [aa]=bd7d9db9050d964ca0433b5d20c2966f
  [ab]=ff05393de57beec36aaaf051f52a3fbf
  [ac]=efb8c452139da6296ed7b4ca6fdcf068
  [ad]=17ee7753969a1146ec3f75717f16936f
  [ae]=e68793c8c1f20607bc61dec6cb0ae2c5
  [af]=70ff5ec47570f620d9f9b6ba54a7e334
  [ag]=6395de167b5e4b00ee21378fd9a57dbb
  [ah]=47329b9e4bceeaf7258146e74f3eb1f3
  [ai]=93338c6d52c14ef9afa4de4d47d1ca9c
  [aj]=27221f61642b480e267da583efe9577b
  [ak]=b5ff547aeff5cb5433bf0a0abd5a10b0
  [al]=80e8d757aa84ea5e549db058e542e249
  [am]=4167326fc4c70dff0061222cd519995f
  [an]=8ba0b037c68deb900564a29c83a422d9
  [ao]=c57ff84109c5d4b5225108c900f7950c
  [ap]=e927fcd5ca66dc7b2772739af43b0944
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
else
  echo "!! 整体校验失败"
  exit 1
fi
