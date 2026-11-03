from __future__ import annotations

import copy
import hashlib
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
    """按点分路径写入配置, 支持 dict 键与 list 下标混用。

    例: train.phases.0.epochs=2  (train.phases 是 list, 0 是下标)
    原先只支持 dict, 遇到 list 会 setdefault 报 AttributeError —— README 里记的
    断点续训 --set train.phases.1.epochs=... 因此一直是坏的。
    """
    target = config
    parts = dotted.split(".")
    for index, part in enumerate(parts[:-1]):
        if isinstance(target, list):
            target = target[int(part)]
            continue
        next_is_index = parts[index + 1].isdigit()
        if not isinstance(target.get(part), (dict, list)):
            target[part] = [] if next_is_index else {}
        target = target[part]
    last = parts[-1]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value


def _resolve_paths(value: Any, config_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, config_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, config_dir, key) for v in value]
    if isinstance(value, str) and key.endswith(("root", "dir", "path", "weights")):
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
        _set_dotted(config, key, yaml.safe_load(value))
    config = _resolve_paths(config, base_path.parent)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    data = config.get("data", {})
    model = config.get("model", {})
    depth_encoding = data.get("depth_encoding", "hha")
    if depth_encoding not in ("hha", "raw"):
        raise ValueError("data.depth_encoding 必须是 'hha' 或 'raw'")
    if depth_encoding == "raw":
        if not data.get("add_depth_validity_mask", False) or not data.get("add_depth_format_flag", False):
            raise ValueError("raw 编码需要 add_depth_validity_mask=true 且 add_depth_format_flag=true")
    for key in ("image_size", "width", "height"):
        value = data.get(key)
        if value is not None and value % 32:
            raise ValueError(f"data.{key}={value} 必须是 32 的倍数（RT-DETR 尺寸约束）")
    mode = model.get("mode", "rgb_only")
    if mode not in ("rgb_only", "tri"):
        raise ValueError("model.mode 必须是 'rgb_only' 或 'tri'")
    fusion = model.get("fusion", "tri_cssa")
    if fusion not in ("tri_cssa", "cmtf"):
        raise ValueError("model.fusion 必须是 'tri_cssa'（v3 通道评分）或 'cmtf'（v5 跨模态注意力）")
    weights = model.get("weights")
    if not weights:
        raise ValueError("model.weights 必须指向本地 RT-DETR 权重文件（rtdetr-l.pt / rtdetr-x.pt，开发机下载后离线保存）")
    phases = config.get("train", {}).get("phases", [])
    if not phases:
        raise ValueError("train.phases 必须配置三阶段（freeze_backbone/freeze_decoder/学习率）")
    dssm = config.get("loss", {}).get("dssm", {})
    if dssm.get("enabled") and dssm.get("implementation") != "proxy":
        raise ValueError("论文 DSSM 公式尚未核验；仅允许显式选择 implementation=proxy 的实验代理项")


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
