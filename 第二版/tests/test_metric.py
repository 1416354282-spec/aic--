from aic_v2.metrics.detection_map import detection_map

def test_perfect_map():
    gt=[{"image_id":"a","class_id":0,"bbox":[.5,.5,.2,.2]}]; pred=[{"image_id":"a","class_id":0,"bbox":[.5,.5,.2,.2],"confidence":.9}]
    result=detection_map(pred,gt,1); assert result["map50_95"]==1.0
