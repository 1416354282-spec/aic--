import numpy as np
from aic_v2.data.boxes import clip_labels

def test_clip_labels():
    labels=np.array([[0,0.0,0.5,0.4,0.2]],np.float32); out=clip_labels(labels)
    assert len(out)==1 and np.all(out[:,1:]>=0) and np.all(out[:,1:]<=1)
