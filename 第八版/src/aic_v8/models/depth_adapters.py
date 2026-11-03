"""Type-specific shallow residuals followed by the existing shared depth encoder."""

import torch
from torch import nn


class DepthTypeAdapters(nn.Module):
    def __init__(self, channels=3, hidden=8):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.SiLU(),
                          nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
                          nn.Conv2d(hidden, channels, 1))
            for _ in range(2)
        ])
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, inputs, depth_type, depth_valid):
        if depth_type is None or depth_valid is None:
            raise ValueError("Depth type adapters require depth_type and depth_valid")
        types = torch.as_tensor(depth_type, device=inputs.device)
        if types.shape != (inputs.shape[0],) or types.dtype not in (torch.int32, torch.int64):
            raise ValueError("depth_type must be an integer tensor of shape [batch]")
        if not ((types == 0) | (types == 1)).all():
            raise ValueError("depth_type: 0=metric uint16, 1=relative uint8")
        output = inputs
        for kind, branch in enumerate(self.branches):
            indices = torch.nonzero(types == kind, as_tuple=False).flatten()
            if indices.numel():
                delta = branch(inputs.index_select(0, indices))
                output = output.index_add(0, indices, delta)
        # Residual biases must not turn sensor holes or dropped modalities into measurements.
        return output * depth_valid
