#!/usr/bin/env python3
import argparse,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import matplotlib.pyplot as plt
from aic_v2.constants import CLASS_NAMES
p=argparse.ArgumentParser();p.add_argument("manifest");p.add_argument("--output",default=str(Path(__file__).parents[1]/"outputs/figures"));a=p.parse_args()
rows=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
counts=Counter(c for r in rows for c in r.get("classes",[]));plt.figure(figsize=(12,5));plt.bar(CLASS_NAMES,[counts[i] for i in range(12)]);plt.xticks(rotation=35,ha="right");plt.ylabel("images containing class");plt.tight_layout();plt.savefig(out/"class_image_distribution.png",dpi=180);plt.close()
plt.figure();plt.hist([r["num_boxes"] for r in rows],bins=30);plt.xlabel("boxes per image");plt.ylabel("images");plt.tight_layout();plt.savefig(out/"boxes_per_image.png",dpi=180);plt.close();print(out)
