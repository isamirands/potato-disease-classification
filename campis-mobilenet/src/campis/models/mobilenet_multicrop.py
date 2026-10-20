from __future__ import annotations
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MultiCropMobileNet(nn.Module):
    def __init__(self, crops: dict, pretrained: bool = True, dropout: float = .3):
        super().__init__(); self.crops = crops
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.features = base.features; self.avgpool = base.avgpool
        dim = base.classifier[0].in_features
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, len(meta["classes"]))) for name, meta in crops.items()})
    def embedding(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(images)), 1)
    def forward(self, images: torch.Tensor, crop: str | list[str]) -> torch.Tensor | dict[str, torch.Tensor]:
        emb = self.embedding(images)
        if isinstance(crop, str): return self.heads[crop](emb)
        result = {}
        for name in set(crop):
            idx = torch.tensor([i for i, c in enumerate(crop) if c == name], device=images.device)
            result[name] = (idx, self.heads[name](emb.index_select(0, idx)))
        return result
    def load_compatible_backbone(self, checkpoint: str) -> dict:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False).get("model_state", {})
        own = self.state_dict(); compatible = {k:v for k,v in state.items() if k in own and k.startswith(("features.", "avgpool.")) and own[k].shape == v.shape}
        self.load_state_dict(compatible, strict=False); return {"loaded": sorted(compatible), "skipped": sorted(set(state)-set(compatible))}
