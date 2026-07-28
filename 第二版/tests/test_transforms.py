import numpy as np
from aic_v2.data.transforms import letterbox

def test_letterbox_keeps_aspect():
    images={"visible":np.zeros((100,200,3),np.uint8),"depth":np.zeros((100,200),np.float32)}
    labels=np.array([[0,.5,.5,.5,.5]],np.float32); result,out,meta=letterbox(images,labels,640)
    assert result["visible"].shape==(640,640,3) and meta.ratio==3.2 and len(out)==1
