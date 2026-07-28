#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from aic_v2.inference.submission import make_zip,validate_submission
p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("test_root");p.add_argument("--zip",default="submission.zip");a=p.parse_args()
ids=sorted(p.stem for p in (Path(a.test_root)/"visible").glob("*") if p.is_file());errors=validate_submission(a.directory,ids)
if errors: raise SystemExit("\n".join(errors[:50]));make_zip(a.directory,a.zip);print(f"校验通过: {len(ids)} 文件, ZIP={a.zip}")
