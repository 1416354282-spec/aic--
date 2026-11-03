from __future__ import annotations

from pathlib import Path
import sys


def ensure_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for path in (str(src), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_paths()

