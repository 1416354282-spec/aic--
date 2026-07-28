import torch
from aic_v2.models.ema import ModelEMA
from aic_v2.engine.checkpoint import save_checkpoint,load_checkpoint

def test_checkpoint_roundtrip(tmp_path):
    model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters()); sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1); scaler=None; ema=ModelEMA(model)
    path=tmp_path/"x.pt"; save_checkpoint(path,model,opt,sch,scaler,ema,2,3,.4,{},"abc")
    state=load_checkpoint(path,model,opt,sch,scaler,ema,"abc"); assert state["epoch"]==2 and state["global_step"]==3
