from __future__ import annotations
import random
from torch.utils.data import Sampler
class HierarchicalCropSampler(Sampler):
    """Uniform crop -> uniform class -> random image sampler."""
    def __init__(self, frame, crops, n_samples=None, seed=42):
        self.n=n_samples or len(frame); self.rng=random.Random(seed); self.index={}
        for i,row in frame.reset_index(drop=True).iterrows(): self.index.setdefault(row.crop,{}).setdefault(row.disease,[]).append(i)
        self.crops=[c for c in crops if c in self.index]
    def __iter__(self):
        for _ in range(self.n):
            crop=self.rng.choice(self.crops); disease=self.rng.choice(list(self.index[crop])); yield self.rng.choice(self.index[crop][disease])
    def __len__(self): return self.n
