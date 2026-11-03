"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

# for register purpose
# [aic_v7 vendored trim] `data` is not imported eagerly: it pulls in
# faster-coco-eval / pycocotools which are only needed by the official
# training dataloader.  aic_v7 uses its own data pipeline (see VENDORED.md).
from . import nn, optim, zoo
