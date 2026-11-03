from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from .constants import NUM_CLASSES


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_config_tree(path: Path) -> dict:
    raw = _load_yaml(path)
    bases = raw.pop("_base_", None) or raw.pop("base_config", None)
    merged = {}
    if bases:
        if isinstance(bases, (str, Path)):
            bases = [bases]
        for base in bases:
            base_path = (path.parent / base).resolve()
            merged = _deep_merge(merged, _load_config_tree(base_path))
    merged = _deep_merge(merged, raw)
    return _resolve_paths(merged, path.parent)


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


def _resolve_paths(value: Any, config_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, config_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, config_dir, key) for v in value]
    if isinstance(value, str) and key.endswith(("root", "dir", "path", "weights", "priors", "yaml")):
        path = Path(value)
        return str((config_dir / path).resolve()) if not path.is_absolute() else str(path)
    return value


def validate_config(config: dict) -> None:
    data = config.get("data", {})
    model = config.get("model", {})
    if data.get("num_classes", NUM_CLASSES) != NUM_CLASSES:
        raise ValueError(f"data.num_classes 必须为 {NUM_CLASSES}")
    if data.get("depth_encoding", "hha") not in ("hha", "raw"):
        raise ValueError("data.depth_encoding 必须是 hha 或 raw")
    mode = model.get("mode", "tri")
    if mode not in ("tri", "rgb_only"):
        raise ValueError("model.mode 必须是 tri 或 rgb_only")
    if model.get("fusion", "tri_cssa") not in ("tri_cssa", "cmtf"):
        raise ValueError("model.fusion 必须是 tri_cssa 或 cmtf")
    if model.get("base", "dfine") not in ("dfine", "rtdetr"):
        raise ValueError("model.base 必须是 dfine 或 rtdetr")
    if not model.get("weights"):
        raise ValueError("model.weights 必须指向本地权重文件")
    relation = model.get("relation", {})
    for key in ("alpha_cooc", "alpha_spatial", "alpha_geo"):
        if not (0.0 <= float(relation.get(key, 0.0)) <= 1.0):
            raise ValueError(f"model.relation.{key} 必须在 [0, 1]")
    if float(config.get("loss", {}).get("relation", {}).get("weight", 0.0)) < 0:
        raise ValueError("loss.relation.weight 必须 >= 0")
    phases = config.get("train", {}).get("phases", [])
    if not phases:
        raise ValueError("train.phases 必须配置三阶段")
    for phase in phases:
        if int(phase.get("epochs", 0)) <= 0:
            raise ValueError("train.phases 中 epochs 必须大于 0")
        if int(phase.get("batch_size", 0)) <= 0:
            raise ValueError("train.phases 中 batch_size 必须大于 0")
        if tuple(int(v) for v in phase.get("size", ())) and any(
            int(v) % 32 for v in phase.get("size", ())
        ):
            raise ValueError(f"train.phases 中 size={phase.get('size')} 必须是 32 的倍数（D-FINE 尺寸约束）")


class Config:
    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "Config":
        return cls(copy.deepcopy(config_dict))

    def merge(self, other: "Config") -> "Config":
        return Config(_deep_merge(self._config, other._config))

    def update(self, updates: Dict) -> None:
        self._config = _deep_merge(self._config, updates)

    def get(self, key_chain: str, default: Any = None) -> Any:
        keys = key_chain.split(".")
        current = self._config
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
                if current is None:
                    return default
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(k)]
                except (ValueError, IndexError):
                    return default
            else:
                return default
        return current

    def set(self, key_chain: str, value: Any) -> None:
        keys = key_chain.split(".")
        current = self._config
        for k in keys[:-1]:
            if isinstance(current, dict) and k not in current:
                current[k] = {}
            if isinstance(current, dict):
                current = current[k]
            elif isinstance(current, list):
                current = current[int(k)]
        last_key = keys[-1]
        if isinstance(current, dict):
            current[last_key] = value
        elif isinstance(current, list):
            current[int(last_key)] = value

    def to_dict(self) -> Dict:
        return copy.deepcopy(self._config)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str, overlays: Iterable[str] = (), overrides: Optional[Dict] = None) -> Config:
    config_path = Path(config_path)
    # _load_config_tree 已按各文件所在目录解析过路径, 此处不得整体重解析
    # (实验配置继承 base 后按实验文件目录二次解析会破坏相对路径)。
    config = Config.from_dict(_load_config_tree(config_path))
    for overlay in overlays:
        config = config.merge(Config.from_dict(_resolve_paths(_load_yaml(Path(overlay)), Path(overlay).parent)))
    if overrides:
        for key_chain, value in overrides.items():
            # --set 覆盖项单独解析 (相对顶层配置文件目录)
            resolved = _resolve_paths(value, config_path.parent, key_chain.rsplit(".", 1)[-1])
            config.set(key_chain, resolved)
    validate_config(config.to_dict())
    return config


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
