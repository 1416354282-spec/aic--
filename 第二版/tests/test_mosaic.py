import numpy as np
from aic_v2.data.mosaic import mosaic4

def test_mosaic_does_not_duplicate_per_modality():
    sample=({"visible":np.zeros((10,20,3),np.uint8),"infrared":np.zeros((10,20),np.uint8),"depth":np.zeros((10,20),np.float32)},np.array([[0,.5,.5,.2,.2]],np.float32))
    _,labels=mosaic4([sample,sample,sample,sample],64)
    assert len(labels)<=4
