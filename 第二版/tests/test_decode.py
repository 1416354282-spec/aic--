import torch
from aic_v2.inference.decode import decode_dfl

def test_decode_anchor_units():
    logits=torch.zeros(1,64,1); scores=torch.zeros(1,2,1); anchors=torch.tensor([[4.,4.]]); stride=torch.tensor([[8.]])
    boxes,_=decode_dfl(logits,scores,anchors,stride)
    assert boxes.shape==(1,1,4) and torch.allclose((boxes[0,0,:2]+boxes[0,0,2:])/2,torch.tensor([4.,4.]))
