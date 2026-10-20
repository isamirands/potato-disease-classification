from __future__ import annotations
import os, tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from campis.inference import Predictor
app=FastAPI(title="CAMPIS")
_predictor=None
def get_predictor():
    global _predictor
    if _predictor is None:
        path=os.getenv("CAMPIS_CHECKPOINT","artifacts/checkpoints/best_model.pt")
        if not os.path.exists(path): raise RuntimeError("No existe un checkpoint configurado")
        _predictor=Predictor(path)
    return _predictor
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/crops")
def crops():
    p=get_predictor(); return [{"id":k,"name":v["display_name"]} for k,v in p.crops.items()]
@app.get("/crops/{crop}/diseases")
def diseases(crop:str):
    p=get_predictor()
    if crop not in p.crops: raise HTTPException(404,"Cultivo no encontrado")
    return p.crops[crop]["classes"]
@app.post("/predict")
async def predict(crop:str=Form(...),image:UploadFile=File(...)):
    if image.content_type not in {"image/jpeg","image/png","image/webp"}: raise HTTPException(415,"Formato no admitido")
    data=await image.read()
    if len(data)>10*1024*1024: raise HTTPException(413,"Archivo demasiado grande")
    try:
        with tempfile.NamedTemporaryFile(suffix=".img") as f: f.write(data); f.flush(); return get_predictor().predict(f.name,crop)
    except ValueError as e: raise HTTPException(400,str(e))
    except Exception as e: raise HTTPException(422,"Imagen inválida o corrupta") from e
