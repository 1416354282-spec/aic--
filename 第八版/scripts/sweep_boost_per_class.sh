#!/bin/bash
# per-class boost 消融 (远程验证集)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/aic/v6/第六版
OUT=/root/aic/v6/boost_per_class_results.txt
echo "per-class boost 消融 (animal=0.46, boost=1.0, min_hits=3, min_score=0.05)" > $OUT
echo "cls | overall_map | cls_AP | boosted" >> $OUT
for c in 0 1 2 3 4 5 6 7 8 9 10 11; do
  python scripts/temporal_post.py /root/aic/v6/outputs/sweep/temporal_val_raw_txt \
    --output /root/aic/v6/outputs/sweep/tv_boost_c$c \
    --images '/root/autodl-tmp/训练集/visible' \
    --labels '/root/autodl-tmp/训练集/labels' \
    --eval-ids /root/aic/v6/第二版-outputs/splits/fold_0_val.txt \
    --class-threshold 0=0.30 --class-threshold 1=0.41 --class-threshold 2=0.46 \
    --class-threshold 3=0.38 --class-threshold 4=0.38 --class-threshold 5=0.38 \
    --class-threshold 6=0.38 --class-threshold 7=0.78 --class-threshold 8=0.63 \
    --class-threshold 9=0.50 --class-threshold 10=0.70 --class-threshold 11=0.79 \
    --singleton-factor 1.0 --track-boost 1.0 --boost-classes $c \
    --report /root/aic/v6/outputs/sweep/tv_boost_c$c.json > /dev/null 2>&1
  python -c "
import json, numpy as np
r = json.load(open('/root/aic/v6/outputs/sweep/tv_boost_c$c.json'))
m = r['eval']['after']
overall = m['map50_95']
ap = np.mean(m['per_class']['$c'])
boosted = r['stats']['boosted']
print(f'$c | {overall:.4f} | {ap:.4f} | {boosted}')
" >> $OUT
done
echo DONE >> $OUT
