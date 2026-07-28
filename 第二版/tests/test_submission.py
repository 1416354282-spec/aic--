from pathlib import Path
from aic_v2.inference.submission import validate_submission

def test_submission_validation(tmp_path:Path):
    (tmp_path/"a.txt").write_text("0 0.5 0.5 0.2 0.2 0.9",encoding="utf-8")
    assert validate_submission(tmp_path,["a"])==[]
