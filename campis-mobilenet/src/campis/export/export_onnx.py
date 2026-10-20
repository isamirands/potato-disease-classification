from __future__ import annotations
from pathlib import Path
import json, torch
from campis.inference import Predictor
class OneHead(torch.nn.Module):
    def __init__(self,m,c): super().__init__(); self.m,self.c=m,c
    def forward(self,x): return self.m(x,self.c)
def export_onnx(checkpoint, output_dir="artifacts/exports"):
    p=Predictor(checkpoint); p.model.eval(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); x=torch.randn(1,3,224,224)
    files=[]
    for crop in p.crops:
        file=out/f"campis_mobilenet_{crop}.onnx"; torch.onnx.export(OneHead(p.model,crop),x,file,input_names=["image"],output_names=["logits"],opset_version=17,dynamo=False); files.append(str(file))
    (out/"labels.json").write_text(json.dumps(p.crops,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"preprocessing.json").write_text(json.dumps({"size":224,"normalization":"ImageNet","padding":"square"}),encoding="utf-8")
    (out/"model_metadata.json").write_text(json.dumps({"model_version":p.version,"architecture":"MobileNetV3-Large shared backbone, crop-specific head","format":"One ONNX graph per manually selected crop","input":"float32 [1,3,224,224] RGB normalized with ImageNet statistics","outputs":"logits for the selected crop classes"},ensure_ascii=False,indent=2),encoding="utf-8")
    return files
