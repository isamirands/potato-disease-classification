# CAMPIS Vision

Pipeline reproducible para clasificar hojas de papa en `early_blight`,
`healthy` y `late_blight` usando dos dominios de adquisición:

- `dataset_campis`: 2.312 imágenes tomadas con cámara profesional.
- `dataset_irish`: 58.709 imágenes tomadas con celular.

Irish representa el 96,21 % de las 61.021 imágenes. Por eso el proyecto no
mezcla todos los archivos de forma ingenua: divide por fuente y clase, conserva
los grupos relacionados en un solo split, balancea referencias durante el
entrenamiento y reporta métricas globales y separadas por cámara.

## Qué cambió respecto a Iteración 5

Se conserva la secuencia original —auditoría, split 70/15/15, aumento,
MobileNetV3Small, entrenamiento, evaluación, Keras, TFLite e inferencia— con
estas correcciones:

- No se mueve ni copia ninguna imagen original. El split se guarda en CSV.
- Todas las rutas vienen de TOML; ya no se mezclan `dataset_v4`, `dataset_v5`,
  Colab y una unidad `G:`.
- El código se puede importar: no monta Drive, grafica, entrena ni termina el
  proceso como efecto lateral.
- El aumento se aplica online solo a train; no crea `train_augmented`.
- El balance predeterminado es por `fuente + clase`, no solo por clase, y se
  resuelve dentro de `tf.data`: cada época vuelve a muestrear.
- Existe una fase de cabeza congelada y otra de fine-tuning real del 30 %.
- Keras y TFLite reciben exactamente RGB `float32` en rango `0..255`; el
  preprocesamiento MobileNetV3 vive dentro del modelo.
- La exportación incluye etiquetas, metadatos y una prueba automática de
  paridad Keras/TFLite sobre 64 imágenes repartidas por todo el test.

La paridad se juzga por la decisión que se despliega, no por un valor suelto.
Falla si cambia la clase predicha en una imagen cuyo margen entre las dos
primeras clases superaba la tolerancia, si la desviación media excede su
límite, o si la desviación en el percentil 99 excede el suyo. El máximo se
reporta pero no decide: con cuantización dinámica la distribución del error es
muy sesgada —mediana en 0,00001 y algún caso aislado en 0,19— y el máximo de
una muestra sesgada crece al comparar más imágenes, así que un umbral sobre el
máximo se endurece solo. Un cambio de clase sobre un empate técnico se acepta;
sobre una predicción confiada, no.
- Cada ejecución guarda configuración, manifest, checkpoints, historial,
  matrices de confusión, predicciones y métricas por fuente.

## Estructura

```text
CAMPIS/
├── configs/                 # configuración completa y smoke
├── datasets/                # datos raw; el pipeline nunca los modifica
├── src/campis/
│   ├── audit.py             # integridad, distribución y duplicados
│   ├── config.py            # dataclasses y validación TOML
│   ├── data.py              # catálogo, split, balance y tf.data
│   ├── modeling.py          # MobileNetV3Small y fine-tuning
│   ├── training.py          # callbacks, checkpoints e historial
│   ├── evaluation.py        # métricas globales y por fuente
│   ├── exporting.py         # Keras, TFLite, metadatos y paridad
│   ├── reporting.py         # figuras de diagnóstico de una ejecución
│   ├── inference.py         # predicción de archivos
│   ├── pipeline.py          # orquestación
│   └── cli.py               # comandos
├── tests/                   # pruebas unitarias y de integración
├── artifacts/               # manifests, reportes, modelos y runs generados
├── legacy/                  # scripts originales, preservados como referencia
├── pyproject.toml
└── run.py
```

## Dataset esperado

```text
datasets/
├── dataset_campis/
│   ├── early_blight/
│   ├── healthy/
│   └── late_blight/
└── dataset_irish/
    ├── early_blight/
    ├── healthy/
    └── late_blight/
```

Las imágenes apartadas por la auditoría viven en `datasets/dataset_fallido/`,
que replica esa misma composición bajo un nivel por fuente. No es una fuente
configurada, así que el pipeline la ignora; `artifacts/reports/dataset_fallido.csv`
registra origen, destino, razón y hash de cada archivo movido.

El orden de clases está fijado en `configs/default.toml` y se exporta junto al
modelo. Las extensiones admitidas por la configuración actual son JPG, JPEG,
PNG y BMP.

## Instalación

Con el entorno virtual existente en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python run.py doctor --config configs/default.toml
```

También se puede instalar exactamente el entorno verificado:

```powershell
python -m pip install -r requirements.txt
```

En este equipo TensorFlow 2.21 detectó CPU, no GPU. TensorFlow nativo de
Windows muestra que las versiones actuales no usan CUDA directamente; para un
entrenamiento completo con GPU conviene ejecutar el mismo proyecto desde WSL2.

## Validación rápida

`smoke.toml` usa solo 12 archivos por combinación fuente/clase, no descarga
ImageNet y entrena una época. Sirve para comprobar el software, no para medir
calidad del modelo.

```powershell
python run.py audit --config configs/smoke.toml --quick
python run.py prepare --config configs/smoke.toml --force
python run.py train --config configs/smoke.toml --run-name smoke
```

## Pipeline completo

Primero puede hacerse un conteo rápido:

```powershell
python run.py audit --config configs/default.toml --quick
```

La auditoría completa descomprime cada imagen y calcula SHA-256. Tardará más,
pero detecta archivos corruptos, duplicados exactos y contenido idéntico con
etiquetas incompatibles.

La verificación descomprime a propósito, no solo valida marcadores: el
`verify()` de Pillow aprueba un JPEG cortado a mitad del scan, que después
reventaría dentro de `tf.data` con el entrenamiento ya avanzado. Por eso
`scan_dataset` llama también a `load()`, y el proyecto fija
`ImageFile.LOAD_TRUNCATED_IMAGES = False` para que Pillow nunca rellene en
silencio los bytes que faltan.

Aun así, Pillow y TensorFlow no comparten decodificador de JPEG. `--decode-tf`
pasa cada imagen por `tf.io.decode_image`, que es literalmente el que usará
`tf.data`, y es la única comprobación concluyente. Reparte el trabajo en hilos
—las operaciones eager liberan el GIL al descomprimir— y tarda alrededor de un
minuto sobre 40.000 imágenes, frente a los minutos de entrenamiento que cuesta
descubrir el mismo archivo a mitad de una época:

```powershell
python run.py audit --config configs/default.toml --decode-tf
python run.py prepare --config configs/default.toml
python run.py train --config configs/default.toml --run-name baseline-mixto
```

El manifest contiene ruta, fuente, tipo de cámara, clase, split, SHA-256,
`group_id`, dimensiones y tamaño. El split es determinista con semilla 42,
estratificado por fuente/clase y mantiene en una sola partición los duplicados
exactos y las sesiones de captura CAMPIS. Una sesión se reconoce por tres vías:
la marca temporal del nombre (`20250107_093439`), las marcas epoch de 10 o 13
dígitos, y las corridas del contador de la cámara (`IMG_0051`, `IMG_0052`,
`IMG_0054`) separadas por menos de `group_sequence_gap` números. El contador es
global a la cámara, así que una ráfaga que cambió de clase sigue siendo una
sola sesión. Sin esta última regla, 1.202 de las 2.312 fotos CAMPIS quedaban
sueltas y podían repartirse entre train y test.

El balance no crea imágenes ni congela un subconjunto: cada grupo `fuente +
clase` se baraja y se repite por separado dentro de `tf.data`, y la época se
arma en turnos estrictos entre grupos. Así una época contiene exactamente
`samples_per_source_class` referencias de cada combinación, pero la fuente
mayoritaria aporta imágenes **distintas** en cada vuelta hasta agotar su
corpus, en lugar de mostrar siempre las mismas 2.000. La minoritaria se repite
y las capas de aumento generan las variaciones en memoria. Validación y test
conservan su distribución natural.

Antes de redimensionar a 224×224 se recorta el cuadrado central
(`crop_to_square`). CAMPIS entrega fotos 1:1 e Irish 4:3; sin el recorte solo
Irish se deformaba y la relación de aspecto quedaba como pista espuria del
dominio. El mismo recorte se aplica al entrenar, evaluar, exportar y predecir.

Validación conserva la distribución natural, donde Irish es el 96 % de las
imágenes. Para que el checkpoint y el early stopping no queden gobernados por
esa sola cámara, el entrenamiento monitorea `val_balanced_loss`: el promedio
simple entre dominios. `history.csv` guarda además `val_campis_*` y
`val_irish_*` en cada época. La validación se calcula en una sola pasada.

## Evaluar, exportar y predecir

```powershell
python run.py evaluate `
  --config configs/default.toml `
  --model artifacts/runs/MI_RUN/model.keras

python run.py export `
  --config configs/default.toml `
  --model artifacts/runs/MI_RUN/model.keras

python run.py predict `
  --config configs/default.toml `
  --model artifacts/runs/MI_RUN/model.keras `
  --input ruta/a/una_imagen.jpg
```

`predict` también acepta una carpeta o un `.tflite`. Para un bundle movido
fuera del run se puede pasar `--labels ruta/model.labels.json`.

## Artefactos de una ejecución

Cada carpeta `artifacts/runs/<fecha>-<nombre>/` contiene, como mínimo:

- `config.resolved.json` y `manifest.csv`;
- `best_head.keras`, `best_fine_tune.keras` cuando corresponde y `model.keras`;
- `history.csv` con métricas globales y por cámara, y resumen del entrenamiento;
- `evaluation/metrics.json`, predicciones y matrices global/CAMPIS/Irish;
- `model.tflite`, etiquetas, metadatos y `tflite_parity.json`;
- `figures/` con las seis figuras de diagnóstico descritas abajo.

## Figuras

`train` genera las figuras al final, después de exportar. Se dibujan leyendo
archivos ya escritos, así que un fallo al graficar no invalida el modelo y el
reporte se puede repetir sobre cualquier ejecución —incluso antigua— sin
volver a entrenar:

```powershell
python run.py report --config configs/default.toml --run artifacts/runs/MI_RUN
```

En `figures/`:

- `training_curves.png` — pérdida y acierto por época, con las dos fases
  encadenadas y la validación desglosada por cámara. La brecha entre las dos
  líneas punteadas es la señal de que el modelo se está especializando en un
  dominio.
- `class_metrics.png` — precision, recall y F1 por clase, con el soporte de
  cada una, en tres paneles: global, CAMPIS e Irish.
- `confidence_distribution.png` — confianza de aciertos frente a errores. Si
  los errores se acumulan cerca de 1,0, ningún umbral de rechazo los filtra.
- `sample_predictions.png` — muestras del test tal como las ve el modelo, ya
  recortadas y redimensionadas. Sirve además para comprobar que el recorte
  cuadrado no está cortando la hoja.
- `top_errors.png` — los errores más confiados, que suelen delatar etiquetas
  dudosas antes que ruido estadístico.
- `dataset_distribution.png` — composición del corpus por fuente, clase y
  partición.

En `evaluation/` siguen las tres matrices de confusión (`confusion_global.png`
y una por cámara).

## Pruebas

```powershell
python -m unittest discover -s tests -t . -v
```

Las pruebas no descargan pesos de ImageNet. Incluyen configuración, split sin
intersecciones, agrupación de duplicados, balance virtual, `tf.data`, modelo,
fine-tuning, evaluación por fuente y exportación/paridad TFLite.

## Límites que deben revisarse con el conocimiento del dataset

SHA-256 detecta duplicados exactos, no recortes, recomprensiones o fotografías
casi iguales. Para CAMPIS la sesión se infiere del nombre del archivo, porque
solo 23 de 400 archivos `IMG_####` muestreados conservan `DateTimeOriginal` en
EXIF: el resto fue despojado y no hay una fecha real que usar. Agrupar por
contador reduce los grupos independientes de 1.408 a 244 y el mayor llega a
294 fotos; el split sigue alcanzando 70/15/15 por clase, pero el test CAMPIS
son ~347 imágenes de pocas sesiones y su intervalo de confianza es amplio.

Irish no expone un identificador de planta/sesión y sus nombres correlativos
(`healthy0`, `healthy1`) no son evidencia de una ráfaga, así que no se agrupan;
si se conoce esa relación, debe incorporarse como `group_id` antes de
interpretar el test como completamente independiente.

Los scripts exportados de Colab se conservaron en `legacy/`. Los dos nombres de
Iteración 5 que permanecen en la raíz son wrappers de compatibilidad hacia el
nuevo CLI.
