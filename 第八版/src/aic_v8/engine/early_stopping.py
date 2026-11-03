"""Validation-count patience for phase transitions and final-phase stopping."""

from __future__ import annotations

import math


class EarlyStopping:
    def __init__(self, config, state=None):
        self.enabled = bool(config.get("enabled", False))
        self.patience = int(config.get("patience", 5))
        self.min_delta = float(config.get("min_delta", 0.001))
        if self.patience < 1 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early_stopping requires patience >= 1 and finite min_delta >= 0")
        state = state or {}
        self.reference = state.get("reference")
        self.bad_validations = int(state.get("bad_validations", 0))

    def update(self, score):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Early stopping received a non-finite validation mAP")
        if not self.enabled:
            return False
        # Compare against the last significant improvement so small gains can accumulate.
        if self.reference is None or score > self.reference + self.min_delta:
            self.reference = score
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        return self.bad_validations >= self.patience

    def state_dict(self):
        return {"reference": self.reference, "bad_validations": self.bad_validations}


# Keep existing phase-2 callers and checkpoint tests compatible.
Phase2EarlyStopping = EarlyStopping
