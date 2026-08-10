# Potato Disease Classification

Clasificación de enfermedades en hojas de papa (Early Blight, Late Blight, Healthy) a partir de imágenes, usando PyTorch. El proyecto entrena y compara varias arquitecturas sobre la misma tarea.

## Arquitecturas soportadas

| Arquitectura     | Config                          | Notas                          |
|------------------|----------------------------------|---------------------------------|
| CNN simple       | `configs/simple_cnn.yaml`       | Baseline desde cero             |
| ResNet18         | `configs/resnet18.yaml`         | Transfer learning (ImageNet)    |
| MobileNetV2      | `configs/mobilenet_v2.yaml`     | Transfer learning (ImageNet)    |
| EfficientNet-B0  | `configs/efficientnet_b0.yaml`  | Transfer learning (ImageNet)    |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Datos

Ver [`data/README.md`](data/README.md) para instrucciones de descarga y preparación del dataset.

```bash
python -m potato_disease.data.prepare
```

## Entrenamiento

```bash
python -m potato_disease.train --config configs/resnet18.yaml
```

## Evaluación

```bash
python -m potato_disease.evaluate --config configs/resnet18.yaml --checkpoint checkpoints/resnet18/<run>/best.pt
```

## Tests

```bash
pytest
```

## Estructura del proyecto

```
configs/            # un YAML por arquitectura (hiperparámetros, modelo, datos)
data/                # raw/ y processed/ (gitignored, ver data/README.md)
notebooks/           # exploración (EDA); no se importa desde potato_disease/
potato_disease/       # paquete: config, data, models, train, evaluate, utils
  models/            # una arquitectura por archivo + registry (get_model)
tests/               # tests unitarios (datos sintéticos, sin depender del dataset real)
checkpoints/         # pesos entrenados (gitignored)
experiments/         # logs de métricas por corrida (gitignored)
```

## Resultados

_Completar a medida que se entrenen y comparen las arquitecturas._

| Arquitectura     | Val Accuracy | Notas |
|------------------|--------------|-------|
| CNN simple       | -            |       |
| ResNet18         | -            |       |
| MobileNetV2      | -            |       |
| EfficientNet-B0  | -            |       |
