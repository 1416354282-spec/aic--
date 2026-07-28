from aic_v2.data.split import multilabel_group_folds,assert_no_group_leakage

def test_group_never_crosses_fold():
    rows=[{"image_id":f"g{i//2}_{i}","group_id":f"g{i//2}","classes":[i%3]} for i in range(12)]
    folds=multilabel_group_folds(rows,3)
    assert_no_group_leakage([r for f in folds[1:] for r in f],folds[0])
