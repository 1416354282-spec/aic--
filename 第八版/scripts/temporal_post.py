#!/usr/bin/env python3
"""时序后处理: 连续帧的 ByteTrack 风格跟踪 + 轨迹规则, 纯 numpy/scipy, 不依赖 torch。

背景: 测试集含大量同场景连续帧。逐帧独立推理对无人机等小目标存在
  - 漏检: 轨迹在相邻帧出现、中间帧缺失
  - 误检: 只闪现一帧、前后无对应物的孤立框
本脚本在预测层面利用时序一致性修正, 不训练、不改模型。

流程:
  1. 读入逐帧预测 (txt: cls cx cy w h conf, 归一化)
  2. (可选) 用可见光图像相邻帧像素差做场景切段, 避免跨场景错误关联
  3. 段内 ByteTrack 风格跟踪: 匀速运动模型 + IoU 门控 + 匈牙利匹配
  4. 轨迹规则 (均可开关/调参):
     - 插值: 轨迹在 t-1 与 t+1 有框、t 缺失 (间隔 <= interp_gap) → 线性补框
     - 分数持久化: 匹配到确认轨迹的框, 分数抬升到轨迹历史最高分 x track_boost
     - 孤立框抑制: 段内前后都匹配不上的单帧框, 分数 x singleton_factor
  5. (可选) 按类别阈值过滤 (语义与 apply_class_threshold 一致: conf > thr)
  6. (可选) --eval-ids + --labels 时计算处理前后的 mAP@50-95

用法:
  # 测试集: 原始预测 -> 时序规则 -> 类别阈值 -> 新提交
  python temporal_post.py raw_preds/ --output sub_temporal/ \
      --images 测试集/visible --class-threshold 10=0.70 ... --global-thr 0.38

  # 验证集: 只评估 eval-ids 内帧的处理前后 mAP
  python temporal_post.py val_preds/ --output val_temporal/ \
      --labels 训练集/labels --eval-ids fold_0_val.txt --report val_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# ---------------------------------------------------------------- 基础工具


def find_image(directory: Path, image_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = directory / f"{image_id}{suffix}"
        if path.exists():
            return path
    return None


def read_predictions(directory: Path, ids: list[str] | None = None) -> dict[str, np.ndarray]:
    """返回 {image_id: (N,6) float32 [cls, cx, cy, w, h, conf]} (按 image_id 排序)。"""
    frames: dict[str, np.ndarray] = {}
    files = sorted(directory.glob("*.txt")) if ids is None else [
        directory / f"{image_id}.txt" for image_id in ids if (directory / f"{image_id}.txt").exists()
    ]
    for path in files:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 6:
                rows.append([float(value) for value in fields])
        frames[path.stem] = np.asarray(rows, np.float32).reshape(-1, 6)
    return {image_id: frames[image_id] for image_id in sorted(frames)}


def read_ground_truth(labels_dir: Path, ids: list[str]) -> list[dict]:
    """GT 格式: [cls, cx, cy, w, h] (无 conf), 与 aic_v8.data.manifest.read_labels 一致。"""
    items = []
    for image_id in ids:
        path = labels_dir / f"{image_id}.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            row = [float(value) for value in fields]
            if not 0 <= int(row[0]) < NUM_CLASSES:
                continue
            items.append({"image_id": image_id, "class_id": int(row[0]), "bbox": row[1:5]})
    return items


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = np.asarray(boxes, np.float32).copy()
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return output


def box_iou(first: np.ndarray, second: np.ndarray, eps=1e-7) -> np.ndarray:
    """输入 xyxy (N,4)/(M,4), 输出 (N,M) IoU。"""
    intersection_size = np.maximum(
        0, np.minimum(first[:, None, 2:], second[None, :, 2:]) - np.maximum(first[:, None, :2], second[None, :, :2])
    )
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_first = np.maximum(0, first[:, 2] - first[:, 0]) * np.maximum(0, first[:, 3] - first[:, 1])
    area_second = np.maximum(0, second[:, 2] - second[:, 0]) * np.maximum(0, second[:, 3] - second[:, 1])
    return intersection / (area_first[:, None] + area_second[None, :] - intersection + eps)


def detection_map(predictions: list[dict], ground_truth: list[dict], num_classes=NUM_CLASSES):
    """与 aic_v8.metrics.detection_map 语义一致 (IoU 0.5:0.05:0.95 + 101 点插值)。"""
    thresholds = np.linspace(0.50, 0.95, 10)
    per_class = {}
    for cls in range(num_classes):
        predictions_for_class = sorted(
            [item for item in predictions if item["class_id"] == cls], key=lambda item: item["confidence"], reverse=True
        )
        truth_for_class = [item for item in ground_truth if item["class_id"] == cls]
        average_precisions = []
        for threshold in thresholds:
            matched = defaultdict(set)
            true_positive = np.zeros(len(predictions_for_class))
            false_positive = np.zeros(len(predictions_for_class))
            for prediction_index, prediction in enumerate(predictions_for_class):
                candidates = [
                    (index, truth) for index, truth in enumerate(truth_for_class)
                    if truth["image_id"] == prediction["image_id"] and index not in matched[prediction["image_id"]]
                ]
                if candidates:
                    predicted_box = cxcywh_to_xyxy(np.asarray([prediction["bbox"]], np.float32))
                    target_boxes = cxcywh_to_xyxy(np.asarray([truth["bbox"] for _, truth in candidates], np.float32))
                    overlaps = box_iou(predicted_box, target_boxes)[0]
                    best = int(np.argmax(overlaps))
                    if overlaps[best] >= threshold:
                        true_positive[prediction_index] = 1
                        matched[prediction["image_id"]].add(candidates[best][0])
                        continue
                false_positive[prediction_index] = 1
            recall = np.cumsum(true_positive) / max(len(truth_for_class), 1)
            precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive + false_positive), 1e-12)
            average_precisions.append(float(np.mean([
                precision[recall >= point].max() if np.any(recall >= point) else 0 for point in np.linspace(0, 1, 101)
            ])))
        per_class[cls] = average_precisions
    values = list(per_class.values())
    return {
        "map50": float(np.mean([value[0] for value in values])),
        "map75": float(np.mean([value[5] for value in values])),
        "map50_95": float(np.mean([np.mean(value) for value in values])),
        "per_class": {cls: [float(value) for value in per_class[cls]] for cls in range(num_classes)},
    }


# ---------------------------------------------------------------- 场景切段


def segment_frames(ids: list[str], images_dir: Path | None, cut_threshold: float, thumb: int,
                   max_frame_gap: int = 5) -> list[list[str]]:
    """按 文件名结构 (序列前缀+帧号) + 图像像素差 切段。

    - 有序列前缀的 id: 前缀相同且帧号差 <= max_frame_gap 判为同段 (每段一个视频)
    - 纯数字 id ('' 前缀, 含 JPG 抽帧与数字命名 PNG 的混合): 帧号差 <= max_frame_gap 且
      像素差 <= cut_threshold 才连, 避免不同视频的全局编号被误连
    - 图像差分只作安全网, 缺图时不阻断结构分段
    """
    if len(ids) < 2:
        return [list(ids)]
    thumbnails: dict[str, np.ndarray] = {}
    if images_dir is not None:
        for image_id in ids:
            path = find_image(images_dir, image_id)
            if path is None:
                continue
            try:
                with Image.open(path) as image:
                    gray = image.convert("L")
                    gray.thumbnail((thumb, thumb))
                    thumbnails[image_id] = np.asarray(gray, np.float32)
            except OSError:
                continue

    def pixel_diff(first: str, second: str) -> float | None:
        if first in thumbnails and second in thumbnails:
            return float(np.abs(thumbnails[first] - thumbnails[second]).mean())
        return None

    # 按序列前缀分组, 组内按帧号排序
    sequences: dict[str, list[tuple[int, str]]] = defaultdict(list)
    unparsed: list[str] = []
    for image_id in ids:
        parsed = parse_frame_number(image_id)
        if parsed is None:
            unparsed.append(image_id)
        else:
            sequences[parsed[0]].append((parsed[1], image_id))

    segments: list[list[str]] = []
    for sequence, items in sorted(sequences.items()):
        items.sort()
        current: list[str] = []
        previous_number: int | None = None
        previous_id: str | None = None
        for number, image_id in items:
            link = False
            if current:
                within_gap = number - previous_number <= max_frame_gap
                if sequence == "":
                    difference = pixel_diff(previous_id, image_id)
                    link = within_gap and (difference is None or difference <= cut_threshold)
                else:
                    link = within_gap
            if link:
                current.append(image_id)
            else:
                if current:
                    segments.append(current)
                current = [image_id]
            previous_number, previous_id = number, image_id
        if current:
            segments.append(current)
    # 未解析 id: 按原顺序做纯差分切段
    if unparsed:
        current = [unparsed[0]]
        for previous, image_id in zip(unparsed, unparsed[1:]):
            difference = pixel_diff(previous, image_id)
            if difference is not None and difference <= cut_threshold:
                current.append(image_id)
            else:
                segments.append(current)
                current = [image_id]
        segments.append(current)
    return segments


# ---------------------------------------------------------------- 跟踪


class Track:
    __slots__ = ("cls", "state", "velocity", "hits", "lost", "frames", "scores", "boxes")

    def __init__(self, cls: int, box: np.ndarray, score: float, frame_index: int):
        self.cls = cls
        self.state = box.copy()          # cxcywh
        self.velocity = np.zeros(4, np.float32)
        self.hits = 1
        self.lost = 0
        self.frames = [frame_index]
        self.scores = [score]
        self.boxes = [box.copy()]


def track_segment(frame_boxes: list[np.ndarray], iou_gate: float, max_lost: int, vel_alpha: float):
    """对一段视频逐帧跟踪。frame_boxes[i]: (N,6) [cls, cx, cy, w, h, conf]。

    返回 matches: list[list[Track|None]], 与每个框一一对应;
    以及 tracks: list[Track] (段内所有轨迹)。
    """
    tracks: list[Track] = []
    matches: list[list[Track | None]] = []
    for frame_index, detections in enumerate(frame_boxes):
        detections = np.asarray(detections, np.float32).reshape(-1, 6)
        frame_matches: list[Track | None] = [None] * len(detections)
        if len(detections):
            classes_present = np.unique(detections[:, 0].astype(int))
            matched_tracks: list[Track] = []
            matched_indices: list[int] = []
            for cls in classes_present:
                candidates = [track for track in tracks if track.cls == cls and track.lost <= max_lost]
                if not candidates:
                    continue
                class_mask = detections[:, 0].astype(int) == cls
                det_indices = np.where(class_mask)[0]
                # 匀速预测
                predicted = np.asarray([track.state + track.velocity for track in candidates], np.float32)
                det_boxes = cxcywh_to_xyxy(detections[det_indices, 1:5])
                pred_boxes = cxcywh_to_xyxy(predicted)
                cost = 1.0 - box_iou(pred_boxes, det_boxes)
                row_ind, col_ind = linear_sum_assignment(cost)
                for row, col in zip(row_ind, col_ind):
                    if 1.0 - cost[row, col] >= iou_gate:
                        track = candidates[row]
                        det_index = int(det_indices[col])
                        frame_matches[det_index] = track
                        matched_tracks.append(track)
                        matched_indices.append(det_index)
            # 更新匹配到的轨迹
            for track, det_index in zip(matched_tracks, matched_indices):
                box = detections[det_index, 1:5]
                observed_velocity = box - track.state
                track.velocity = vel_alpha * observed_velocity + (1 - vel_alpha) * track.velocity
                track.state = box
                track.hits += 1
                track.lost = 0
                track.frames.append(frame_index)
                track.scores.append(float(detections[det_index, 5]))
                track.boxes.append(box.copy())
            # 未匹配框 → 新轨迹
            for det_index in range(len(detections)):
                if frame_matches[det_index] is None:
                    track = Track(int(detections[det_index, 0]), detections[det_index, 1:5],
                                  float(detections[det_index, 5]), frame_index)
                    tracks.append(track)
                    frame_matches[det_index] = track
        # 本帧未匹配的轨迹 lost+1
        seen = {id(track) for track in frame_matches if track is not None}
        for track in tracks:
            if track.hits and id(track) not in seen:
                track.lost += 1
        matches.append(frame_matches)
    return matches, tracks


# ---------------------------------------------------------------- 轨迹规则


def parse_frame_number(image_id: str) -> tuple[str, int] | None:
    """'000013_034_00000001' -> ('000013_034', 1); 'shuming_909_00000270' -> ('shuming_909', 270);
    纯数字 '00000286' -> ('', 286)。解析失败返回 None。"""
    parts = image_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    if image_id.isdigit():
        return "", int(image_id)
    return None


def apply_temporal_rules(frames: dict[str, np.ndarray], segments: list[list[str]], arguments):
    """返回 (output, modified, stats, details):
      output:   {image_id: list[list[float]]} 应用规则后的行 (分数可修改, 可新增插值框)
      modified: {image_id: list[bool]} 与 output 逐行对应, True=时序规则生成/修改的行
    帧内每行用可变 list 表示, 轨迹记录 (帧, 分数) 引用, 规则按值定位修改。
    """
    output = {image_id: [list(map(float, row)) for row in np.asarray(value, np.float32)] for image_id, value in frames.items()}
    modified = {image_id: [False] * len(output[image_id]) for image_id in output}
    stats = {"interpolated": 0, "suppressed": 0, "boosted": 0}
    details: dict[str, dict] = {}
    boundary_margin = max(arguments.interp_gap, 1)
    for segment in segments:
        frame_boxes = [np.asarray(output[image_id], np.float32).reshape(-1, 6) if output[image_id] else np.zeros((0, 6), np.float32)
                       for image_id in segment]
        _, tracks = track_segment(frame_boxes, arguments.iou_gate, arguments.max_lost, arguments.vel_alpha)
        segment_length = len(segment)
        # 解析真实视频帧号, 用于跨采样间隔的插值比例
        numbers = [parse_frame_number(image_id) for image_id in segment]
        for track in tracks:
            if len(track.frames) < 2:
                continue
            hits = sorted(zip(track.frames, track.boxes, track.scores))
            track_max_raw = max(track.scores)
            # 1) 插值: 命中帧之间缺检的中间帧
            if arguments.interp_score is not None and arguments.interp_score > 0:
                for (f1, b1, s1), (f2, b2, s2) in zip(hits, hits[1:]):
                    gap = f2 - f1 - 1
                    if not (0 < gap <= arguments.interp_gap):
                        continue
                    if min(s1, s2) < arguments.interp_anchor_min:
                        continue
                    if f1 < boundary_margin or f2 > segment_length - 1 - boundary_margin:
                        continue
                    n1, n2 = numbers[f1], numbers[f2]
                    for step in range(1, gap + 1):
                        frame_index = f1 + step
                        if n1 is not None and n2 is not None and n1[0] == n2[0] and numbers[frame_index] is not None \
                                and numbers[frame_index][0] == n1[0] and n2[1] > n1[1]:
                            ratio = (numbers[frame_index][1] - n1[1]) / (n2[1] - n1[1])
                        else:
                            ratio = step / (f2 - f1)
                        ratio = float(np.clip(ratio, 0.0, 1.0))
                        box = [float(a * (1 - ratio) + b * ratio) for a, b in zip(b1, b2)]
                        score = min(s1, s2) * arguments.interp_score
                        image_id = segment[frame_index]
                        output[image_id].append([float(track.cls), *box, score])
                        modified[image_id].append(True)
                        stats["interpolated"] += 1
                        details.setdefault(image_id, {"added": 0, "suppressed": 0, "boosted": 0})
                        details[image_id]["added"] += 1
            # 2) 分数持久化: 确认轨迹上、且自身有一定证据的弱框抬分到 track_max x boost
            if arguments.track_boost is not None and arguments.track_boost > 0 \
                    and len(track.frames) >= arguments.min_track_hits \
                    and (arguments.boost_classes is None or track.cls in arguments.boost_classes):
                for frame_index, score in zip(track.frames, track.scores):
                    if not (arguments.boost_min_score <= score < track_max_raw * arguments.track_boost):
                        continue
                    image_id = segment[frame_index]
                    # 按位置+类别+原始分数定位该帧内属于本轨迹的行
                    for candidate in output[image_id]:
                        if int(candidate[0]) == track.cls and abs(float(candidate[5]) - score) < 1e-6:
                            candidate[5] = track_max_raw * arguments.track_boost
                            stats["boosted"] += 1
                            details.setdefault(image_id, {"added": 0, "suppressed": 0, "boosted": 0})
                            details[image_id]["boosted"] += 1
                            break
        # 3) 孤立框抑制: 只出现一帧、且不在段边界附近的轨迹 → 降级到队尾 (不删除)
        if arguments.singleton_factor is not None and arguments.singleton_factor < 1:
            for track in tracks:
                if track.hits != 1:
                    continue
                frame_index = track.frames[0]
                if frame_index < boundary_margin or frame_index > segment_length - 1 - boundary_margin:
                    continue
                image_id = segment[frame_index]
                for row_index, candidate in enumerate(output[image_id]):
                    if int(candidate[0]) == track.cls and abs(float(candidate[5]) - track.scores[0]) < 1e-6:
                        candidate[5] = track.scores[0] * arguments.singleton_factor
                        modified[image_id][row_index] = True
                        stats["suppressed"] += 1
                        details.setdefault(image_id, {"added": 0, "suppressed": 0, "boosted": 0})
                        details[image_id]["suppressed"] += 1
                        break
    # 每帧按置信度降序 (与原始提交约定一致)
    for image_id in output:
        order = np.argsort([-row[5] for row in output[image_id]])
        output[image_id] = [output[image_id][i] for i in order]
        modified[image_id] = [modified[image_id][i] for i in order]
    return output, modified, stats, details


def apply_thresholds(frames: dict[str, np.ndarray], global_thr: float, class_thr: dict[int, float]):
    """语义与 apply_class_threshold 一致: conf > threshold。"""
    output = {}
    for image_id, frame in frames.items():
        kept = []
        for row in frame:
            cls = int(row[0])
            threshold = class_thr.get(cls, global_thr)
            if float(row[5]) > threshold:
                kept.append(row)
        output[image_id] = np.asarray(kept, np.float32).reshape(-1, 6) if kept else np.zeros((0, 6), np.float32)
    return output


# ---------------------------------------------------------------- mAP 对比


def to_map_items(frames: dict[str, np.ndarray], ids: list[str]) -> list[dict]:
    items = []
    for image_id in ids:
        for row in frames.get(image_id, np.zeros((0, 6))):
            items.append({
                "image_id": image_id,
                "class_id": int(row[0]),
                "confidence": float(row[5]),
                "bbox": [float(value) for value in row[1:5]],
            })
    return items


# ---------------------------------------------------------------- 主流程


def parse_thresholds(pairs: list[str]) -> dict[int, float]:
    result = {}
    for pair in pairs or []:
        cls, thr = pair.split("=", 1)
        result[int(cls)] = float(thr)
    return result


def main():
    parser = argparse.ArgumentParser(description="连续帧时序后处理 (ByteTrack 风格跟踪 + 轨迹规则)")
    parser.add_argument("input", help="预测目录 (每图一个 txt)")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--images", help="可见光图像目录 (用于场景切段)")
    parser.add_argument("--labels", help="GT 标签目录 (配合 --eval-ids 计算 mAP)")
    parser.add_argument("--eval-ids", help="仅这些 id 参与评估 (txt 每行一个 id)")
    parser.add_argument("--report", help="变更报告 JSON 路径")
    # 轨迹规则参数
    parser.add_argument("--iou-gate", type=float, default=0.1, help="跟踪匹配 IoU 门限 (默认 0.1)")
    parser.add_argument("--max-lost", type=int, default=30, help="轨迹丢失后保留帧数 (默认 30)")
    parser.add_argument("--vel-alpha", type=float, default=0.1, help="速度 EMA 系数 (默认 0.1)")
    parser.add_argument("--interp-gap", type=int, default=1, help="允许补帧的最大间隔 (默认 1)")
    parser.add_argument("--interp-score", type=float, default=0.8, help="插值框分数 = min(邻居分)×系数 (默认 0.8, None 关)")
    parser.add_argument("--interp-anchor-min", type=float, default=0.05, help="插值锚点最低分数 (默认 0.05, 过滤噪声轨迹)")
    parser.add_argument("--min-track-hits", type=int, default=3, help="确认轨迹所需命中数 (默认 3)")
    parser.add_argument("--track-boost", type=float, default=0.0, help="分数持久化抬分比例 (默认 0 关, 如 0.85)")
    parser.add_argument("--boost-min-score", type=float, default=0.05, help="抬分仅作用于原始分>=该值的框 (默认 0.05)")
    parser.add_argument("--boost-classes", default=None, help="抬分生效的类别, 逗号分隔 (默认全部), 如 8,3,4")
    parser.add_argument("--singleton-factor", type=float, default=0.1, help="孤立框分数乘数 (默认 0.1, 1 关)")
    parser.add_argument("--cut-threshold", type=float, default=15.0, help="场景切段像素差阈值 (默认 15)")
    # 过滤参数
    parser.add_argument("--global-thr", type=float, default=None, help="全局置信度阈值 (严格 >)")
    parser.add_argument("--class-threshold", action="append", default=[], metavar="CLS=THR")
    arguments = parser.parse_args()

    input_dir = Path(arguments.input)
    ids = sorted(path.stem for path in input_dir.glob("*.txt"))
    eval_ids = None
    if arguments.eval_ids:
        eval_ids = [line.strip() for line in Path(arguments.eval_ids).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"读入 {len(ids)} 帧预测")

    frames = read_predictions(input_dir, ids)
    images_dir = Path(arguments.images) if arguments.images else None
    segments = segment_frames(ids, images_dir, arguments.cut_threshold, 64)
    same_scene_pairs = sum(len(segment) - 1 for segment in segments if len(segment) > 1)
    print(f"场景切段: {len(segments)} 段, 其中 {same_scene_pairs} 对同场景相邻帧")

    if arguments.boost_classes:
        arguments.boost_classes = {int(value) for value in arguments.boost_classes.split(",")}

    processed, modified, stats, details = apply_temporal_rules(frames, segments, arguments)
    print(f"轨迹规则: 插值补框 {stats['interpolated']}, 抬分 {stats['boosted']}, 孤立抑制 {stats['suppressed']}")

    class_thr = parse_thresholds(arguments.class_threshold)
    global_thr = arguments.global_thr if arguments.global_thr is not None else 0.0

    def keep_row(cls: int, score: float, is_modified: bool) -> bool:
        """被时序规则修改的行 (插值补框/孤立降级) 不参与阈值过滤:
        降级框保留在队尾不伤害 precision, 且若为漏检真框仍能在 recall 尾部匹配。"""
        if is_modified:
            return True
        return score > class_thr.get(cls, global_thr)

    # 阈值过滤 (语义: conf > thr, 与 apply_class_threshold 一致), 时序生成的行豁免
    filtered: dict[str, list[list[float]]] = {}
    for image_id in sorted(processed):
        kept = [row for row, flag in zip(processed[image_id], modified[image_id])
                if keep_row(int(row[0]), float(row[5]), flag)]
        filtered[image_id] = kept

    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id in sorted(filtered):
        # 类别 id 取整, 其余 8 位小数, 与原始提交 (predict_raw 输出) 格式一致
        lines = [" ".join([str(int(row[0])), *[f"{value:.8f}" for value in row[1:]]])
                  for row in filtered[image_id]]
        # CRLF 行尾, 与原始提交 (Windows 生成) 逐字节一致; newline='' 禁用文本模式换行转换
        (output_dir / f"{image_id}.txt").write_text("\r\n".join(lines), encoding="utf-8", newline="")
    final = {image_id: np.asarray(rows, np.float32).reshape(-1, 6) if rows else np.zeros((0, 6), np.float32)
             for image_id, rows in filtered.items()}

    report = {"frames": len(ids), "segments": len(segments), "stats": stats}
    if arguments.global_thr is not None or class_thr:
        print(f"阈值过滤后输出: {output_dir}")

    if arguments.labels and eval_ids:
        ground_truth = read_ground_truth(Path(arguments.labels), eval_ids)
        # 基线: 输入原始预测 + 同样阈值过滤 (与最终版仅在时序规则上有差别)
        baseline_frames = apply_thresholds(frames, global_thr, class_thr)
        before = detection_map(to_map_items(baseline_frames, eval_ids), ground_truth)
        after = detection_map(to_map_items(final, eval_ids), ground_truth)
        print("\n== 处理前 (输入+阈值) ==")
        print(f"  mAP@50-95: {before['map50_95']:.4f}   mAP@50: {before['map50']:.4f}   mAP@75: {before['map75']:.4f}")
        print("== 处理后 (时序规则+阈值) ==")
        print(f"  mAP@50-95: {after['map50_95']:.4f}   mAP@50: {after['map50']:.4f}   mAP@75: {after['map75']:.4f}")
        print(f"  差值: {after['map50_95'] - before['map50_95']:+.4f}")
        print("\n  各类别 mAP@50-95 (前/后/差):")
        for cls in range(NUM_CLASSES):
            delta = np.mean(after["per_class"][cls]) - np.mean(before["per_class"][cls])
            print(f"    {cls:2d} {CLASS_NAMES[cls]:12s}: {np.mean(before['per_class'][cls]):.4f} / "
                  f"{np.mean(after['per_class'][cls]):.4f} / {delta:+.4f}")
        report["eval"] = {"before": before, "after": after}
    if arguments.report:
        report["details"] = details
        Path(arguments.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告: {arguments.report}")
    print(f"\n完成: 输出 {output_dir}")


if __name__ == "__main__":
    main()
