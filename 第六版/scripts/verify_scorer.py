"""校验 tune_thresholds 的快速 mAP 与 aic_v6.metrics.detection_map 完全等价。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aic_v6.data.manifest import read_labels
from aic_v6.engine.trainer import read_ids
from aic_v6.metrics.detection_map import detection_map
from tune_thresholds import make_scorer, replay

SPLIT_DIR = "D:/AIC面对城市场景的视觉多模态目标检测/第二版/outputs/splits"
TRAIN_ROOT = "D:/AIC面对城市场景的视觉多模态目标检测/第六版/outputs/tmp_val_root"
EXPORT = str(ROOT / "outputs/sweep/val_passes_last_newlabels.jsonl")
LIMIT = 60

images = []
with open(EXPORT, encoding="utf-8") as handle:
    for line in handle:
        images.append(json.loads(line))
images = images[:LIMIT]
val_ids = read_ids(Path(SPLIT_DIR) / "fold_0_val.txt")[:LIMIT]
ground_truth = []
for image_id in val_ids:
    labels, _ = read_labels(Path(TRAIN_ROOT) / "labels" / f"{image_id}.txt")
    for row in labels:
        ground_truth.append({"image_id": image_id, "class_id": int(row[0]), "bbox": [float(v) for v in row[1:5]]})

scorer = make_scorer(images, ground_truth, 12)
for name, class_thresholds, global_threshold in [
    ("基线 全部0.001", {}, 0.001),
    ("全局0.33", {}, 0.33),
    ("逐类 {1:0.5, 2:0.4}", {1: 0.5, 2: 0.4}, 0.05),
]:
    mine = scorer(class_thresholds, global_threshold, 0.7, 100)
    predictions = []
    for image in images:
        boxes, scores, labels = replay(image, class_thresholds, global_threshold, 0.7, 100)
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            predictions.append({"image_id": image["image_id"], "class_id": int(label),
                                "confidence": float(score), "bbox": box})
    reference = detection_map(predictions, ground_truth, 12)
    print(f"{name}: 预测 {len(predictions)} 框")
    print(f"   快速 map50_95={mine['map50_95']:.6f}   detection_map={reference['map50_95']:.6f}   "
          f"Δ={abs(mine['map50_95'] - reference['map50_95']):.2e}")
    print(f"   map50: 快速={mine['map50']:.6f} 参考={reference['map50']:.6f}   "
          f"map75: 快速={mine['map75']:.6f} 参考={reference['map75']:.6f}")
    worst = max(abs(a - b) for cls in range(12)
                for a, b in zip(mine["per_class"][cls], reference["per_class"][cls]))
    print(f"   逐类逐IoU 最大偏差: {worst:.2e}")
