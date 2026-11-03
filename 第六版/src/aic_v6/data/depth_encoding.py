"""
深度图 HHA 几何编码 — 将物理深度值转化为 CNN 友好的类图像表示。

参考: Gupta et al. "Learning Rich Features from RGB-D Images for
      Object Detection and Segmentation" (ECCV 2014)

三个通道:
  H (Horizontal Disparity):   水平视差, 近处亮/远处暗 — 替代原始深度值
  HA (Height Above Ground):   相对地面高度 — 物体垂直定位
  A  (Angle with Gravity):    表面法向量与重力夹角 — 物体形状/朝向

设计依据 (InfraParis WACV 2024):
  深度图的像素值是物理距离而非纹理, 卷积网络对绝对数值不敏感。
  HHA 编码将距离转化为视差、高度、角度三个通道, 与 RGB 的空间结构一致,
  CNN 更容易学习跨模态特征。
"""

from __future__ import annotations

import cv2
import numpy as np


def hha_encode(
    depth: np.ndarray,
    fx: float = 1.0,
    fy: float = 1.0,
    cx: float = 0.5,
    cy: float = 0.5,
    valid_min: float = 300.0,
    valid_max: float = 20000.0,
) -> np.ndarray:
    """
    HHA 编码: 单通道 uint16 深度图 → 三通道 float32 几何表示

    Args:
        depth:      (H, W) uint16 深度图, 单位 mm
        fx, fy:     相机焦距 (归一化坐标空间, 默认值适用)
        cx, cy:     相机主点 (归一化坐标空间)
        valid_min:  最小有效深度 (mm), 默认 300mm ≈ 30cm (ZED 最近距离)
        valid_max:  最大有效深度 (mm), 默认 20000mm = 20m

    Returns:
        (H, W, 3) float32 HHA 编码图, 各通道归一化到 [0, 1], 无效像素为 0
    """
    H, W = depth.shape
    depth_f = depth.astype(np.float32)

    # 有效性掩码
    valid = ((depth_f >= valid_min) & (depth_f < valid_max)).astype(np.float32)
    depth_clipped = np.where(valid > 0, depth_f, valid_max)

    # ── Channel 0: Horizontal Disparity (水平视差) ──
    # disparity ∝ 1/depth: 近处大(亮), 远处小(暗)
    disparity = 1.0 / np.maximum(depth_clipped, 1.0)
    valid_disparity = disparity[valid > 0]
    if len(valid_disparity) > 0:
        d_min, d_max = valid_disparity.min(), valid_disparity.max()
        h_ch = (disparity - d_min) / (d_max - d_min + 1e-8)
    else:
        h_ch = np.zeros_like(disparity)
    h_ch *= valid

    # ── Channel 1: Height Above Ground (相对地面高度) ──
    # 像素 y 坐标 × depth 加权: 远处同 y 的像素实际高度 > 近处
    y_coords = np.arange(H, dtype=np.float32).reshape(-1, 1).repeat(W, axis=1)
    height = (y_coords - cy * H) * depth_clipped / (fy * H * 1000.0)
    height = np.clip(height, -3, 3)  # clip 到合理范围
    ha_ch = (height - height.min()) / (height.max() - height.min() + 1e-8)
    ha_ch *= valid

    # ── Channel 2: Angle with Gravity (表面法向量与重力夹角) ──
    # 深度梯度 → 法向量 → 与 (0,0,1) 夹角
    dz_dx = cv2.Sobel(depth_clipped, cv2.CV_32F, 1, 0, ksize=3) / fx
    dz_dy = cv2.Sobel(depth_clipped, cv2.CV_32F, 0, 1, ksize=3) / fy
    grad_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    nz = 1.0 / grad_mag                     # 法向量 z 分量
    angle = np.arccos(np.clip(nz, -1, 1))   # 与重力夹角
    a_ch = angle / np.pi                     # 归一化到 [0, 1]
    a_ch *= valid

    return np.stack([h_ch, ha_ch, a_ch], axis=-1).astype(np.float32)


def depth_to_3channel_raw(
    depth: np.ndarray,
    valid_min: float = 300.0,
    valid_max: float = 20000.0,
    mode: str = "log",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    原始深度编码 (兼容旧版): 归一化深度值 + 有效性掩码 + 格式标记。

    返回三个独立通道 (H, W) float32:
      - depth_val:    归一化后的深度值 (log/inverse/linear)
      - depth_valid:  有效性掩码 (1.0=有效, 0.0=无效)
      - depth_format: 格式标记 (1.0=PNG度量深度, 0.0=JPG相对深度)
    """
    depth_f = depth.astype(np.float32)
    valid = ((depth_f >= valid_min) & (depth_f < valid_max)).astype(np.float32)

    if mode == "log":
        normalized = np.log1p(np.clip(depth_f, valid_min, valid_max)) / np.log1p(valid_max)
    elif mode == "inverse":
        normalized = np.where(valid > 0, valid_min / np.maximum(depth_f, valid_min), 0.0)
    else:  # "linear"
        normalized = np.clip(depth_f, valid_min, valid_max) / valid_max

    return (normalized * valid).astype(np.float32), valid, np.ones_like(valid, np.float32)


def jpg_depth_encode(gray: np.ndarray) -> np.ndarray:
    """
    JPG 相对深度编码: 将单通道 8-bit 相对深度转化为 3 通道互补表示。

    JPG 深度是相对值 (0=远, 255=近), 缺乏度量尺度, 无法计算 HHA。
    但通过生成 3 个互补视图, 让 Depth 编码器能从中学习:

      Ch0: 正向归一化深度    [0, 1]  — 直接深度值
      Ch1: 逆向深度 (类视差)  [0, 1]  — 近处亮/远处暗, 与 HHA 的 H 通道语义对齐
      Ch2: 梯度幅值 (边缘)    [0, 1]  — 深度突变处 = 物体边界

    三通道互补, 消除了简单复制三通道的信息冗余。

    Args:
        gray: (H, W) uint8 灰度图 (JPG 相对深度)

    Returns:
        (H, W, 3) float32, 各通道归一化到 [0, 1]
    """
    depth = gray.astype(np.float32) / 255.0

    # Ch0: 正向深度
    ch0 = depth

    # Ch1: 逆向深度 (类视差, 与 HHA 的 H 通道语义一致)
    ch1 = 1.0 - depth

    # Ch2: 梯度幅值 (Sobel 边缘 → 深度不连续处 = 物体轮廓)
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    ch2 = np.sqrt(grad_x ** 2 + grad_y ** 2)
    ch2 = ch2 / (ch2.max() + 1e-8)

    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)
