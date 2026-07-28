import numpy as np


def effective_number_weights(counts, beta=0.999, max_weight=5.0):
    counts = np.asarray(counts, dtype=np.float64)
    weights = (1 - beta) / np.maximum(1 - np.power(beta, counts), 1e-12)
    weights /= weights.mean()
    return np.minimum(weights, max_weight).astype(np.float32)

# 注意：不直接把该权重乘到每类所有 BCE 负样本上。
# 它用于正目标重权、采样或后续经过验证的分类损失实现。
