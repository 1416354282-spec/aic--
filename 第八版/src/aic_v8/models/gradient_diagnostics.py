"""Sampled statistics at each modality feature's backward-hook entry."""

import math


class FeatureGradientStats:
    def __init__(self):
        self.levels = {}

    def record(self, name, maximum, scale):
        values = self.levels.setdefault(name, {
            "observed": 0, "finite": 0, "clipped": 0, "nonfinite": 0,
            "scale_sum": 0.0, "scale_min": 1.0, "gradient_max": 0.0,
        })
        values["observed"] += 1
        maximum = float(maximum)
        if not math.isfinite(maximum):
            values["nonfinite"] += 1
            return
        scale = float(scale)
        values["finite"] += 1
        values["clipped"] += int(scale < 1.0)
        values["scale_sum"] += scale
        values["scale_min"] = min(values["scale_min"], scale)
        values["gradient_max"] = max(values["gradient_max"], maximum)

    def pop(self):
        result = {}
        for name, values in self.levels.items():
            count = values["finite"]
            result[name] = {key: value for key, value in values.items() if key != "scale_sum"}
            result[name]["scale_mean"] = values["scale_sum"] / count if count else None
            result[name]["clip_fraction"] = values["clipped"] / count if count else None
            if not count:
                result[name]["scale_min"] = None
                result[name]["gradient_max"] = None
        self.levels.clear()
        return result
