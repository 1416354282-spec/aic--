from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_dotted(config: dict, dotted: str, value: Any) -> None:
    target = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _parse_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _resolve_paths(value: Any, config_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, config_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, config_dir, key) for v in value]
    if isinstance(value, str) and (key.endswith(("root", "dir", "path", "weights"))):
        path = Path(value)
        return str((config_dir / path).resolve()) if not path.is_absolute() else str(path)
    return value


def load_config(base: str | Path, overlays: Iterable[str | Path] = (), overrides: Iterable[str] = ()) -> dict:
    base_path = Path(base).resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for overlay in overlays:
        overlay_path = Path(overlay).resolve()
        with overlay_path.open("r", encoding="utf-8") as handle:
            config = _deep_merge(config, yaml.safe_load(handle) or {})
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖参数必须是 key=value: {item}")
        key, value = item.split("=", 1)
        _set_dotted(config, key, _parse_value(value))
    config = _resolve_paths(config, base_path.parent)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    fusion = config.get("model", {}).get("fusion", "early")
    if fusion not in {"early", "gated_stem"}:
        raise ValueError(f"未实现的融合策略: {fusion}")
    if config.get("augmentation", {}).get("mixup", {}).get("enabled", False):
        raise ValueError("MixUp 尚未通过消融验证，第二版基线禁止启用")
    if config.get("data", {}).get("num_folds", 3) < 2:
        raise ValueError("num_folds 必须至少为 2")


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def config_fingerprint(config: dict) -> str:
    import hashlib
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
