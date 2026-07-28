#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aic_v2.data.manifest import audit_dataset, save_records, summarize

p = argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--output", default=str(Path(__file__).parents[1] / "outputs/audit/train_manifest.jsonl")); p.add_argument("--no-hash", action="store_true")
a = p.parse_args(); records = audit_dataset(a.root, not a.no_hash); save_records(records, a.output)
summary = summarize(records); Path(a.output).with_name("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
