from __future__ import annotations

from copy import deepcopy

import torch


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000.0)).item())
        source = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])
