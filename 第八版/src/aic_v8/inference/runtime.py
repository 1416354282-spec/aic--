"""Checkpoint-owned architecture/preprocessing and safe resumable prediction output."""
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from aic_v8.config import _deep_merge, _resolve_paths, _set_dotted, validate_config
from aic_v8.data.provenance import preprocessing_config
from aic_v8.io_utils import atomic_output
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR


def inference_config(state, config_path=None, overlays=(), overrides=()):
    if not state.get('config'):
        raise ValueError('Checkpoint has no architecture config; provide a versioned training checkpoint')
    saved = copy.deepcopy(state['config'])
    config = copy.deepcopy(saved)
    for filename in ([config_path] if config_path else []) + list(overlays):
        path = Path(filename).resolve()
        values = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        config = _deep_merge(config, _resolve_paths(values, path.parent))
    for entry in overrides:
        key, value = entry.split('=', 1)
        _set_dotted(config, key, yaml.safe_load(value))
    # We load a complete learned state, so inference never requires the COCO file.
    config['model'] = saved['model']
    config['data'].update(preprocessing_config(saved['data']))
    for phase in config['train']['phases']:
        phase['amp'] = False  # Training-only scaler flags do not constrain old checkpoint inference.
    validate_config(config)
    return config


def load_inference_checkpoint(checkpoint, config_path=None, overlays=(), overrides=()):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return inference_config(state, config_path, overlays, overrides), state


def build_inference_model(config, state, device):
    model = TriModalRTDETR(config, device, load_pretrained=False)
    weights = state.get('ema')
    model.load_state_dict(weights if weights is not None else state['model'], strict=True)
    return model.eval()


def select_batch(batch, indices, device):
    """Slice every collated per-image field, including depth_type and metadata."""
    index = torch.as_tensor(indices, dtype=torch.long)
    selected = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            selected[key] = value.index_select(0, index.to(value.device)).to(device, non_blocking=True)
        elif isinstance(value, list):
            selected[key] = [value[i] for i in indices]
        else:
            selected[key] = value
    return selected


def prepare_prediction_output(output, checkpoint, config, image_ids, tta):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with Path(checkpoint).open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(chunk)
    manifest = {
        'version': 2, 'checkpoint_sha256': digest.hexdigest(), 'model': config['model'],
        'data_root': str(Path(config['data']['test_root']).resolve()),
        'preprocessing': preprocessing_config(config['data']),
        'size': config['train']['phases'][-1]['size'], 'postprocess': config['validation'],
        'image_ids': list(image_ids), 'tta': bool(tta),
    }
    # YAML permits integer class-threshold keys; compare in the JSON representation.
    manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
    path = output / 'run_manifest.json'
    done = {p.stem for p in output.glob('*.txt')}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Output belongs to a different checkpoint/config/dataset; use a new output directory')
    elif done:
        raise ValueError('Existing predictions have no run manifest; use a new output directory')
    else:
        with atomic_output(path) as temporary:
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if done - set(image_ids):
        raise ValueError('Output contains unexpected image IDs')
    # Validate existing files before treating them as completed; do not silently reuse truncation.
    if done:
        from .submission import validate_submission
        errors = validate_submission(output, image_ids, config['model']['num_classes'],
                                     config['validation'].get('max_detections', 100))
        errors = [error for error in errors if not error.startswith('缺少 ')]
        if errors:
            raise ValueError('Invalid saved predictions: ' + '; '.join(errors[:5]))
    return done
