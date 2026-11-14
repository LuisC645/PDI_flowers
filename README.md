# Conteo y clasificación de flores en una cama de cultivo (Piloto PDI)

Proyecto de **Procesamiento Digital de Imágenes**. A partir de un video que recorre desde arriba una cama de flores de 32 m × 1.35 m, se busca:

1. Segmentar la cama.
2. Contar las flores (blancas, rojas, amarillas y naranjas).
3. Decidir cuáles están listas para cortar, según su tamaño y su color.

Primero se normaliza el video: se segmenta la cama, se endereza y se fija su escala. Después se analizan las flores.

**Equipo:** Luis Castillo y Benjamin Ruiz.

> **Estado (14 nov 2026).** Preprocesamiento (Bloque A) **cerrado en su parte automática**: A7 abre la puerta en `63-40A` (`results/metrics/qc_block_a_63-40A.json`). Faltan dos tareas manuales que el código no puede hacer: las máscaras a mano (F1.4) y las medidas de la moneda (F1.3). El Bloque B corre de punta a punta como **borrador** sin calibrar. Los métodos siguen como *propuesta* en `config/decisions.md`. El avance por día y lo pendiente están en `docs/worklog.md`.

## Cómo correrlo

```bash
conda activate pdi
python scripts/standardize_videos.py   # videos_raw -> videos_processed (10 fps, cuadro más nítido)
python scripts/inventory.py            # inventario + mosaico de muestra
python scripts/block_a.py              # A1-A7: normalización del video y control de calidad
python scripts/block_b.py              # B1-B5: solo corre si A7 abre la puerta (--ignore-gate para explorar)
```

Cada script acepta `--steps` (por ejemplo `python scripts/block_a.py --steps a4 a56 a7`) y `--video 63-40A`. Cada paso lee de disco lo que guardó el anterior, así que se puede repetir uno solo. Los parámetros están en `config/params.yaml`.

Tareas manuales pendientes:

```bash
python scripts/export_gt_frames.py     # F1.4: exporta ~20 cuadros a data/ground_truth/frames/
# pintar la cama en blanco y guardar en data/ground_truth/masks/<video>_<cuadro>.png
python scripts/evaluate_block_a.py     # IoU, Dice y contaminación contra las máscaras a mano
# F1.3: llenar data/measurements/measurements.csv (moneda con pie de rey, cama, moneda en el dosel)
```

## Qué entrega el Bloque A (contrato para el Bloque B)

Se lee con `src/common/normalized.py`:

```python
from src.common.normalized import NormalizedVideo
nv = NormalizedVideo("63-40A")
nv.video_path        # video normalizado: cama horizontal, ancho W = 1080 px (8 px/cm)
nv.valid_frames      # cuadros utilizables (medidos o interpolados)
nv.roi(i)            # región útil del cuadro i: franja de la cama, con imagen, sin camarógrafo
nv.matrix(i)         # matriz 2x3 cuadro original -> normalizado
nv.px_per_cm         # escala
```

En el cuadro normalizado, x va a lo largo de la cama e y a lo ancho; la cama ocupa las filas `[margin, margin + W)`.

## Pipeline

| Paso | Módulo | Qué hace (borrador) | Salida principal |
|---|---|---|---|
| Estandarización | `scripts/standardize_videos.py` | 60 → 10 fps, conserva el cuadro más nítido de cada ventana de 6 | `data/videos_processed/`, `frame_index_*.csv`, `sharpness_*.csv` |
| A1 Nitidez | `src/normalization/sharpness.py` | Varianza del Laplaciano; borroso si < 70 % de la mediana móvil | `results/tables/frame_quality_*.csv` |
| A2 Máscara | `src/normalization/bed.py` | Brillo L suavizado (sombra entre camas) + paralaje de movimiento, fusionados en el tiempo → Otsu → morfología → componente central. Si el paralaje rompe la máscara, se usa solo el brillo | `data/normalized/masks/*_bed_masks.npz` |
| A2 Camarógrafo | `src/normalization/operator.py` | Píxeles que casi no cambian en el tiempo y tocan el borde inferior → caja fija por video | `data/normalized/masks/*_operator_mask.png` |
| A3 Bordes | `src/normalization/bed.py` | Borde superior e inferior por columna (cerca de las rectas del cuadro anterior) → 2 rectas con RANSAC → ángulo, centro, ancho | `data/normalized/transforms/*_geometry_raw.csv` |
| A4 Estabilización | `src/normalization/tracking.py` | Atípicos con filtro de Hampel, interpolación de huecos ≤ 1 s, Savitzky-Golay | `*_geometry.csv` |
| A5 Geometría | `src/normalization/geometry.py` | Rotación + escala (ancho = 1080 px = 8 px/cm) + centrado; matriz guardada | `data/normalized/video/*.mp4`, `*_transforms.csv` |
| A6 Color | `src/normalization/color.py` | Iguala la media Lab del follaje verde con la del video | (columnas en `*_transforms.csv`) |
| A7 Calidad | `scripts/block_a.py` | Vuelve a medir la cama en el video normalizado (ancho, centro, ángulo, sacudida lateral, color del follaje) y compara con las metas de `params.qc` | `results/metrics/qc_block_a_*.json` |
| B1/B2 Detección | `src/analysis/detection.py` | Umbral HSV del perfil → morfología → transformada de distancia → watershed → diámetro | `results/tables/detections_*.csv` |
| B3 Contar una vez | `src/analysis/counting.py` | Correlación de fase → coordenada global (u, v) → enlace cuadro a cuadro (húngaro) | `shifts_*.csv`, `flowers_*.csv` |
| B4 Cortar | `src/analysis/counting.py` | Diámetro ≥ moneda **y** saturación ≥ umbral | columna `cut` en `flowers_*.csv` |
| B5 Resultados | `scripts/block_b.py` | Totales, % a cortar, flores por metro, mosaico marcado | `summary_*.json`, `per_meter_*.csv`, figuras `b5_*` |

## Estructura de carpetas

```
proy/
├── README.md
├── environment.yml          # entorno conda "pdi"
├── config/                  # params.yaml y decisions.md
├── data/
│   ├── videos_raw/          # videos originales (entrada; se conservan)
│   ├── videos_processed/    # videos estandarizados: 10 fps, sin audio (base de trabajo)
│   ├── measurements/        # medidas de referencia (moneda, cama)
│   ├── ground_truth/        # máscaras y conteos hechos a mano
│   └── normalized/          # máscaras, transformaciones y video normalizado
├── src/
│   ├── common/              # utilidades compartidas (rutas, video)
│   ├── normalization/       # Bloque A
│   └── analysis/            # Bloque B
├── scripts/                 # ejecución de cada bloque
├── notebooks/               # exploración
├── tests/
├── results/                 # tablas, métricas y figuras
└── report/
```

Todos los nombres de archivos, carpetas, variables y columnas están en inglés.

## Datos

Los videos y todo lo que hay en `data/` **no se suben a git** por su peso; solo se conserva la estructura de carpetas. Tampoco se suben `results/figures/` ni `results/tables/detections_*.csv` (se regeneran con los scripts).
