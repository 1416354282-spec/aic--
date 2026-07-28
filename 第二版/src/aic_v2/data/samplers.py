"""Dependency-light dataset samplers."""
from __future__ import annotations

import math
import random


class InfiniteSampler:
    def __init__(self, size, shuffle=True, seed=0):
        if size <= 0: raise ValueError("size must be positive")
        self.size, self.shuffle, self.seed = size, shuffle, seed
    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            order = list(range(self.size))
            if self.shuffle: rng.shuffle(order)
            yield from order
    def __len__(self): return self.size


class DistributedSampler:
    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False):
        if not 0 <= rank < num_replicas: raise ValueError("invalid rank")
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.shuffle, self.seed, self.drop_last, self.epoch = shuffle, seed, drop_last, 0
        self.num_samples = (len(dataset) // num_replicas) if drop_last else math.ceil(len(dataset) / num_replicas)
    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.num_samples
    def __iter__(self):
        ids = list(range(len(self.dataset)))
        if self.shuffle: random.Random(self.seed + self.epoch).shuffle(ids)
        total = self.num_samples * self.num_replicas
        ids = ids[:total] if self.drop_last else (ids + ids[:max(0, total - len(ids))])
        return iter(ids[self.rank:total:self.num_replicas])


class RepeatFactorSampler:
    """按图像类别频率重复尾类样本，不改变标签或引入外部数据。"""
    def __init__(self, sample_classes, threshold=0.01, seed=42):
        from collections import Counter
        self.seed, self.epoch = seed, 0
        classes = [set(int(c) for c in item) for item in sample_classes]
        counts = Counter(c for item in classes for c in item)
        n = max(1, len(classes))
        self.factors = [max([math.sqrt(threshold / max(counts[c] / n, 1e-12)) for c in item] or [1.0]) for item in classes]
        self.factors = [max(1.0, value) for value in self.factors]
    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return sum(math.ceil(value) for value in self.factors)
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        result = []
        for index, value in enumerate(self.factors):
            count = int(value) + int(rng.random() < value % 1)
            result.extend([index] * count)
        rng.shuffle(result)
        return iter(result)


class WeightedSampler:
    def __init__(self, weights, num_samples=None, replacement=True, seed=0):
        self.weights = [float(x) for x in weights]
        if not self.weights or any(x < 0 for x in self.weights) or sum(self.weights) <= 0: raise ValueError("weights must be non-negative with positive sum")
        self.num_samples = len(self.weights) if num_samples is None else num_samples
        self.replacement, self.seed, self.epoch = replacement, seed, 0
    def __len__(self): return self.num_samples
    def set_epoch(self, epoch): self.epoch = epoch
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        if self.replacement: return iter(rng.choices(range(len(self.weights)), weights=self.weights, k=self.num_samples))
        keys = [(rng.random() ** (1.0 / w), i) for i, w in enumerate(self.weights) if w > 0]
        return iter(i for _, i in sorted(keys, reverse=True)[:self.num_samples])
