from __future__ import annotations
import torch
class TemperatureScaler(torch.nn.Module):
    def __init__(self): super().__init__(); self.log_temperature=torch.nn.Parameter(torch.zeros(()))
    def forward(self, logits): return logits/self.log_temperature.exp().clamp(min=.05,max=10)
    def fit(self, logits, labels):
        optim=torch.optim.LBFGS(self.parameters(),lr=.1,max_iter=50)
        def closure(): optim.zero_grad(); loss=torch.nn.functional.cross_entropy(self(logits),labels); loss.backward(); return loss
        optim.step(closure); return float(self.log_temperature.exp().detach())
