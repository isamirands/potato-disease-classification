from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from .preprocess import inference_transform
from campis.models import MultiCropMobileNet
from campis.training.checkpoint import load_checkpoint
DISPLAY = {"healthy":"Saludable","early_blight":"Tizón temprano","late_blight":"Tizón tardío","bacterial_spot":"Mancha bacteriana","septoria_leaf_spot":"Mancha foliar por Septoria","common_rust":"Roya común","northern_leaf_blight":"Tizón foliar del norte","gray_leaf_spot":"Mancha gris","coffee_leaf_rust":"Roya del café","sigatoka":"Sigatoka","cordana":"Cordana","blast":"Añublo","brown_spot":"Mancha parda","bacterial_leaf_blight":"Tizón bacteriano"}
class Predictor:
    def __init__(self, checkpoint, device="cpu", thresholds=None):
        state=torch.load(checkpoint,map_location=device,weights_only=False); self.crops=state["crops"]; self.model=MultiCropMobileNet(self.crops,pretrained=False).to(device); self.model.load_state_dict(state["model_state"]); self.model.eval(); self.device=device; self.thresholds=thresholds or {}; self.version="1.0.0"; self.transform=inference_transform()
    def predict(self, image, crop: str) -> dict:
        if crop not in self.crops: raise ValueError("Cultivo no admitido")
        im=Image.open(image) if isinstance(image,(str,Path)) else image
        x=self.transform(im).unsqueeze(0).to(self.device)
        with torch.inference_mode(): probs=torch.softmax(self.model(x,crop)[0],0).cpu().tolist()
        classes=self.crops[crop]["classes"]; ranked=sorted(zip(classes,probs),key=lambda x:x[1],reverse=True); best,confidence=ranked[0]; accepted=confidence>=self.thresholds.get(crop,0.0)
        return {"crop":crop,"crop_name":self.crops[crop]["display_name"],"prediction":best if accepted else None,"prediction_name":DISPLAY.get(best,best) if accepted else None,"confidence":confidence,"status":"prediction" if accepted else "uncertain","warning":None if accepted else "No se puede clasificar con suficiente confianza.","top_predictions":[{"class":c,"probability":p} for c,p in ranked[:3]],"model_version":self.version,"disclaimer":"Clasificación visual; no es un diagnóstico fitopatológico definitivo."}
