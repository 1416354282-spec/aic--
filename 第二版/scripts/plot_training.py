#!/usr/bin/env python3
"""训练可视化：读取 metrics.jsonl 生成多子图分析报告。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aic_v2.constants import CLASS_NAMES


def load_metrics(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot_training(log_path: Path, output_dir: Path, title_prefix: str = ""):
    records = load_metrics(log_path)
    train_entries = [r for r in records if r.get("type") == "train"]
    val_entries = [r for r in records if r.get("type") == "val"]
    init_entry = next((r for r in records if r.get("type") == "init"), {})

    if not train_entries:
        print("未找到训练记录"); return
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})
    prefix = f"{title_prefix}_" if title_prefix else ""

    # ── 图1: Loss + LR ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = [r["epoch"] for r in train_entries]
    ax1.plot(epochs, [r["loss"] for r in train_entries], "b-", alpha=0.7, lw=1.2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss"); ax1.set_title(f"{title_prefix}Training Loss".strip())
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, [r["lr"] for r in train_entries], "r-", lw=1.2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate"); ax2.set_title("LR Schedule")
    ax2.grid(True, alpha=0.3); ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}loss_lr.png"); plt.close(fig)

    # ── 图2: mAP 曲线 ──────────────────────────────────
    if val_entries:
        fig, ax = plt.subplots(figsize=(10, 5))
        val_epochs = [r["epoch"] for r in val_entries]
        ax.plot(val_epochs, [r["map50"] for r in val_entries], "g-o", ms=3, label="mAP@50", lw=1)
        ax.plot(val_epochs, [r["map75"] for r in val_entries], "b-s", ms=3, label="mAP@75", lw=1)
        ax.plot(val_epochs, [r["map50_95"] for r in val_entries], "r-^", ms=3, label="mAP@50:95", lw=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("mAP"); ax.set_title(f"{title_prefix}Validation mAP".strip())
        ax.legend(); ax.grid(True, alpha=0.3)
        best_epoch = max(val_entries, key=lambda r: r["map50_95"])
        ax.annotate(f"best {best_epoch['map50_95']:.4f}", (best_epoch["epoch"], best_epoch["map50_95"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8, color="red")
        fig.tight_layout(); fig.savefig(output_dir / f"{prefix}map_curves.png"); plt.close(fig)

        # ── 图3: 各阶段 per-class AP50  ──────────────────
        _plot_per_class_progression(val_entries, output_dir, prefix)

        # ── 图4: 最终 per-class AP50 柱状图 ──────────────
        last_val = val_entries[-1]
        if last_val.get("per_class_ap50"):
            fig, ax = plt.subplots(figsize=(12, 5))
            ap50 = last_val["per_class_ap50"]
            colors = ["#2ecc71" if v >= 0.5 else "#f39c12" if v >= 0.25 else "#e74c3c" for v in ap50]
            bars = ax.bar(range(len(ap50)), ap50, color=colors)
            ax.set_xticks(range(len(ap50))); ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("AP@50"); ax.set_title(f"{title_prefix}Per-Class AP@50 (epoch {last_val['epoch']})".strip())
            ax.set_ylim(0, max(1.0, max(ap50) * 1.1))
            for bar, val in zip(bars, ap50):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout(); fig.savefig(output_dir / f"{prefix}per_class_ap50.png"); plt.close(fig)

    # ── 汇总 JSON ────────────────────────────────────────
    summary = {
        "config": init_entry,
        "total_epochs": len(train_entries),
        "final_loss": train_entries[-1]["loss"] if train_entries else None,
        "best_map50_95": max(r["map50_95"] for r in val_entries) if val_entries else None,
        "best_epoch": max(val_entries, key=lambda r: r["map50_95"])["epoch"] if val_entries else None,
        "final_per_class_ap50": val_entries[-1].get("per_class_ap50") if val_entries else None,
    }
    (output_dir / f"{prefix}summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"可视化完成 → {output_dir}")
    print(f"  总轮次: {summary['total_epochs']}, 最终 loss: {summary['final_loss']:.4f}" if summary['final_loss'] else "")
    if val_entries:
        print(f"  最佳 mAP@50:95 = {summary['best_map50_95']:.4f} (epoch {summary['best_epoch']})")


def _plot_per_class_progression(val_entries: list[dict], output_dir: Path, prefix: str):
    """绘制各类别 AP50 随训练变化的热力图。"""
    if not val_entries or not val_entries[0].get("per_class_ap50"):
        return
    num_classes = len(val_entries[0]["per_class_ap50"])
    epochs = [r["epoch"] for r in val_entries]
    matrix = np.array([r.get("per_class_ap50", [0] * num_classes) for r in val_entries])

    fig, ax = plt.subplots(figsize=(14, max(6, num_classes * 0.45)))
    im = ax.imshow(matrix.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(0.7, matrix.max()))
    ax.set_xticks(range(len(epochs))); ax.set_xticklabels(epochs, rotation=45, fontsize=7)
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_title(f"{prefix}Per-Class AP@50 Progression".strip())
    cbar = fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label("AP@50")
    for i in range(num_classes):
        for j in range(len(epochs)):
            val = matrix[j, i]
            color = "white" if val < 0.3 else "black"
            if val > 0.001:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}per_class_heatmap.png"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="训练曲线可视化")
    parser.add_argument("log", help="metrics.jsonl 路径")
    parser.add_argument("--output", "-o", help="输出目录", default=None)
    parser.add_argument("--title", default="", help="图表标题前缀")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"未找到日志: {log_path}")

    output_dir = Path(args.output) if args.output else log_path.parent / "plots"
    plot_training(log_path, output_dir, args.title)


if __name__ == "__main__":
    main()
