"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# [aic_v7 vendored trim] Only dist_utils is needed by the model/criterion path.
# logger / profiler / visualizer pull in loguru / calflops / PIL and belong to
# the official training loop, which aic_v7 does not use (see VENDORED.md).
from .dist_utils import setup_print, setup_seed
