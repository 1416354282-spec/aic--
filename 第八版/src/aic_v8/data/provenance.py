"""Version the input representation and selected labels independently of split IDs."""
import hashlib
import json
from pathlib import Path

PREPROCESS_DEFAULTS = {
    'depth_encoding': 'hha', 'depth_mode': 'log', 'hha_version': 'legacy',
    'infrared_mode': 'mean', 'add_depth_validity_mask': False, 'add_depth_format_flag': False,
}


def preprocessing_config(data):
    return {key: data.get(key, default) for key, default in PREPROCESS_DEFAULTS.items()}


def dataset_fingerprint(data, train_ids, val_ids):
    root = Path(data['train_root']).resolve()
    digest = hashlib.sha256(json.dumps({'root': str(root), 'preprocessing': preprocessing_config(data)},
                                     sort_keys=True).encode())
    for split, ids in (('train', train_ids), ('val', val_ids)):
        for image_id in sorted(ids):
            path = root / data.get('labels_subfolder', 'labels') / (image_id + '.txt')
            digest.update((split + '\0' + image_id + '\0').encode())
            digest.update(path.read_bytes() if path.exists() else b'<missing>')
            digest.update(b'\0')
    return digest.hexdigest()
