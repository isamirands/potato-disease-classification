from __future__ import annotations
import random, numpy as np, torch
def save_checkpoint(path, model, optimizer, scheduler, epoch, crops, config, metric):
    torch.save({"model_state":model.state_dict(), "optimizer_state":optimizer.state_dict() if optimizer else None, "scheduler_state":scheduler.state_dict() if scheduler else None, "epoch":epoch, "crops":crops, "config":config, "metric":metric, "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}},path)
def load_checkpoint(path, model, map_location="cpu"):
    state=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(state["model_state"]); return state
