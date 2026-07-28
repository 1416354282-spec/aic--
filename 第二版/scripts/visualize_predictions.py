#!/usr/bin/env python3
"""预测结果可视化：在图片上绘制检测框，支持 GT 对比。"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aic_v2.constants import CLASS_NAMES
from aic_v2.data.manifest import find_image, read_labels

# 12 个类别的可视化颜色
CLASS_COLORS = [
    (0, 255, 0),     # 0 person
    (255, 100, 0),   # 1 boat
    (0, 200, 255),   # 2 animal
    (255, 200, 0),   # 3 seat
    (200, 0, 255),   # 4 sign
    (100, 255, 150), # 5 bicycle
    (0, 100, 255),   # 6 car
    (255, 0, 100),   # 7 ball
    (255, 255, 0),   # 8 light
    (0, 150, 150),   # 9 garbage can
    (150, 0, 200),   # 10 uav
    (200, 100, 50),  # 11 tricycle
]


def draw_boxes(image: np.ndarray, boxes: list[dict], color: tuple, label_prefix: str = "", thickness: int = 2) -> np.ndarray:
    """在图像上绘制边界框。boxes: [{"bbox": [cx,cy,w,h] (归一化), "label": int, "score": float}, ...]"""
    h, w = image.shape[:2]
    for box in boxes:
        cx, cy, bw, bh = box["bbox"]
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{label_prefix}{CLASS_NAMES[box['label']]}"
        if box.get("score"):
            label += f" {box['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(image, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return image


def parse_predictions(pred_file: Path) -> list[dict]:
    """解析预测 .txt 文件。每行: class_id cx cy w h confidence"""
    boxes = []
    if pred_file.exists():
        for line in pred_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 6:
                boxes.append({
                    "label": int(parts[0]),
                    "bbox": [float(x) for x in parts[1:5]],
                    "score": float(parts[5]),
                })
    return boxes


def visualize_predictions(
    data_root: Path,
    pred_dir: Path | None,
    image_ids: list[str],
    output_dir: Path,
    n_samples: int = 16,
    confidence_threshold: float = 0.1,
    show_gt: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = random.sample(image_ids, min(n_samples, len(image_ids)))

    n_cols = min(4, len(samples))
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, image_id in enumerate(samples):
        ax = axes[idx // n_cols][idx % n_cols]
        vis_path = find_image(data_root / "visible", image_id)
        if vis_path is None:
            ax.text(0.5, 0.5, f"missing\n{image_id}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([]); continue

        image = cv2.cvtColor(cv2.imread(str(vis_path)), cv2.COLOR_BGR2RGB)

        # 绘制 GT（虚线框）
        if show_gt:
            gt_labels, _ = read_labels(data_root / "labels" / f"{image_id}.txt")
            if len(gt_labels):
                gt_boxes = [{"label": int(row[0]), "bbox": row[1:5].tolist()} for row in gt_labels]
                image = draw_boxes(image, gt_boxes, (255, 255, 255), "GT:", 1)

        # 绘制预测（实线框）
        if pred_dir:
            preds = parse_predictions(pred_dir / f"{image_id}.txt")
            preds = [p for p in preds if p["score"] >= confidence_threshold]
            for pred in preds:
                cls = pred["label"]
                image = draw_boxes(image, [pred], CLASS_COLORS[cls % len(CLASS_COLORS)], "", 2)

        ax.imshow(image); ax.set_title(image_id, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # 隐藏空子图
    for idx in range(len(samples), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout(pad=1)
    output_path = output_dir / "prediction_samples.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"可视化了 {len(samples)} 张 → {output_path}")

    # 同时导出单张大图
    _export_single_images(data_root, pred_dir, samples[:4], output_dir, confidence_threshold, show_gt)


def _export_single_images(data_root, pred_dir, image_ids, output_dir, conf_thresh, show_gt):
    """导出单独的大尺寸图，方便细看。"""
    for image_id in image_ids:
        vis_path = find_image(data_root / "visible", image_id)
        if vis_path is None:
            continue
        image = cv2.cvtColor(cv2.imread(str(vis_path)), cv2.COLOR_BGR2RGB)
        if show_gt:
            gt_labels, _ = read_labels(data_root / "labels" / f"{image_id}.txt")
            if len(gt_labels):
                gt_boxes = [{"label": int(row[0]), "bbox": row[1:5].tolist()} for row in gt_labels]
                image = draw_boxes(image, gt_boxes, (255, 255, 255), "GT:", 1)
        if pred_dir:
            preds = parse_predictions(pred_dir / f"{image_id}.txt")
            preds = [p for p in preds if p["score"] >= conf_thresh]
            for pred in preds:
                image = draw_boxes(image, [pred], CLASS_COLORS[pred["label"] % len(CLASS_COLORS)], "", 2)
        h, w = image.shape[:2]
        fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
        plt.imshow(image); plt.title(image_id, fontsize=10); plt.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(output_dir / f"single_{image_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="预测结果可视化")
    parser.add_argument("data_root", help="数据集根目录（含 visible/, labels/）")
    parser.add_argument("--predictions", "-p", default=None, help="预测 .txt 目录")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--n", type=int, default=16, help="采样数量")
    parser.add_argument("--confidence", type=float, default=0.1, help="置信度阈值")
    parser.add_argument("--no-gt", action="store_true", help="不显示 GT")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    visible_dir = data_root / "visible"
    if not visible_dir.exists():
        raise SystemExit(f"未找到 visible 目录: {visible_dir}")

    all_ids = sorted(p.stem for p in visible_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not all_ids:
        raise SystemExit(f"visible/ 中未找到图片")

    pred_dir = Path(args.predictions) if args.predictions else None
    output_dir = Path(args.output) if args.output else (pred_dir.parent if pred_dir else data_root.parent) / "visualizations"
    visualize_predictions(data_root, pred_dir, all_ids, output_dir, args.n, args.confidence, not args.no_gt)


if __name__ == "__main__":
    main()
