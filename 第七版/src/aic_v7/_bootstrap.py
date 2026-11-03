from __future__ import annotations

from pathlib import Path
import sys


def ensure_paths() -> None:
    """把 第七版/src 与项目根加入 sys.path (第七版自包含, 不依赖第六版)。"""
    root = Path(__file__).resolve().parents[3]
    v7_src = root / "第七版" / "src"
    for path in (str(v7_src), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_paths()
