# Datos

Este proyecto usa imágenes de hojas de papa clasificadas en 3 clases: `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy` (subset del dataset [PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)).

## 1. Descargar

Descarga el dataset (ej. desde Kaggle) y coloca las carpetas de clase directamente dentro de `data/raw/`, de forma que quede así:

```
data/raw/
├── Potato___Early_blight/
│   ├── img001.jpg
│   └── ...
├── Potato___Late_blight/
│   └── ...
└── Potato___healthy/
    └── ...
```

`data/raw/` y `data/processed/` están en `.gitignore` — las imágenes nunca se suben al repositorio.

## 2. Preparar (split train/val/test)

```bash
python -m potato_disease.data.prepare
```

Esto lee `data/raw/<clase>/*.jpg` y genera:

```
data/processed/
├── train/<clase>/...
├── val/<clase>/...
└── test/<clase>/...
```

en formato compatible con `torchvision.datasets.ImageFolder`, listo para usar por `potato_disease/data/dataset.py`.
