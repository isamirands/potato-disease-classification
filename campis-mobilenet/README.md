# CAMPIS MobileNet

Pipeline PyTorch reproducible para clasificación visual foliar de tomate, maíz, café, plátano y arroz. El usuario siempre selecciona primero el cultivo. No es un diagnóstico fitopatológico.

## Inicio

```powershell
cd C:\CAMPIS\campis-mobilenet
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python scripts\prepare_data.py --config configs\datasets.yaml
.\.venv\Scripts\python scripts\train_model.py --config configs\train.yaml
```

`prepare_data.py` crea `data/raw/DOWNLOAD_INSTRUCTIONS.json`. PlantVillage puede clonarse desde su repositorio oficial; RoCoLe, Banana Leaf y Paddy Doctor requieren verificar licencia/versión o autenticación y descargarse manualmente. No se entrena un cabezal hasta que existan todas sus clases requeridas.

Para probar código sin datasets reales: `python -m pytest -q`. Para predicción: `python scripts/predict_image.py --crop tomato --image hoja.jpg`. API: `uvicorn campis.api.main:app --reload`. Interfaz: `streamlit run app.py`.

Los datos crudos, modelos y exportaciones se excluyen de Git. Los manifiestos registran hash, fuente, versión, país y split determinista (seed 42). La exportación genera un ONNX por cultivo, conservando el concepto de backbone compartido para despliegue.
