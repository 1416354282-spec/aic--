#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aic_v2.data.split import multilabel_group_folds, assert_no_group_leakage

p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("--folds",type=int,default=3); p.add_argument("--output",default=str(Path(__file__).parents[1]/"outputs/splits")); a=p.parse_args()
rows=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]; folds=multilabel_group_folds(rows,a.folds); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
for i,val in enumerate(folds):
 train=[r for j,f in enumerate(folds) if j!=i for r in f]; assert_no_group_leakage(train,val)
 (out/f"fold_{i}_train.txt").write_text("\n".join(r["image_id"] for r in train),encoding="utf-8"); (out/f"fold_{i}_val.txt").write_text("\n".join(r["image_id"] for r in val),encoding="utf-8")
 print(f"fold {i}: train={len(train)} val={len(val)}")
