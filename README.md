# Conteo y clasificación de flores en una cama de cultivo (Piloto PDI)

Proyecto de **Procesamiento Digital de Imágenes**. A partir de un video que recorre desde arriba una cama de flores de 32 m × 1.35 m, se busca:

1. Segmentar la cama.
2. Contar las flores (blancas, rojas, amarillas y naranjas).
3. Decidir cuáles están listas para cortar, según su tamaño y su color.

Primero se normaliza el video: se segmenta la cama, se endereza y se fija su escala. Después se analizan las flores.

**Equipo:** Luis Castillo y Benjamin Ruiz.

> **Estado (24 nov 2026). Código completo y comparado con los datos de la empresa.** El pipeline corre de punta a punta en las 9 camas: estandarización → Bloque A (normalización, A7 pasa en las 9) → Bloque B (detección, conteo y cuándo se corta: hoy / mañana / otros días). Cada cama se entrega en flores y en tallos (unidad de la empresa), y se compara con el reporte de recepción de la empresa (`scripts/compare_company.py`). Corrección del 24 nov: `63-40A.mp4` era una copia de la cama 63-41A; el verdadero 63-40A es `63_40A.mp4`. Falta el informe. Detalle en `README_roadmap.md`, `config/decisions.md` y `docs/worklog.md`.

## Resultados (versión final)

Las cifras absolutas frente a la empresa (tallos, variedades y fechas) están en `README_empresa.md`. Ese archivo es local y no se sube a git. Aquí solo van conteos propios y porcentajes.

**Conteo de flores (9 camas, sin conteo manual: miden consistencia, no error absoluto)**

| Métrica | Valor |
|---|---|
| Flores totales | 219 672 (media 24 408 por cama; 19 986 – 32 605) |
| Densidad | media 565 flores/m² (463 – 755) |
| Línea de conteo vs seguimiento | diferencia media 10.6 %, máx. 22 %; 4 de 9 camas dentro de ±10 % |
| Largo recorrido frente a 32 m | error medio 5.9 % |
| Variación por metro dentro de la cama | CV mediano 9.1 % (máx. 41.7 %, densidad real mayor al final de la cama) |
| Camas del mismo color | CV 3.1 – 7.3 % |
| Reparto (criterio moneda) | hoy 56.9 %, mañana 11.0 %, otros días 32.1 % |

| Cama | Color | Flores (línea) | Flores (seguimiento) | Diferencia | Hoy | Mañana | Otros | Flores/m² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 63-38B | blanca | 22 029 | 21 834 | +0.9 % | 32.2 % | 11.9 % | 55.8 % | 510 |
| 63-39A | naranja | 21 197 | 22 194 | −4.5 % | 58.0 % | 12.7 % | 29.3 % | 491 |
| 63-39B | blanca | 24 418 | 24 684 | −1.1 % | 41.4 % | 11.9 % | 46.7 % | 565 |
| 63-40A | naranja | 19 986 | 23 790 | −16.0 % | 60.4 % | 10.8 % | 28.8 % | 463 |
| 63-41A | naranja | 22 486 | 27 211 | −17.4 % | 59.9 % | 10.5 % | 29.5 % | 521 |
| 63-58A | amarilla | 27 349 | 24 364 | +12.3 % | 75.2 % | 9.6 % | 15.2 % | 633 |
| 63-59A | amarilla | 26 174 | 22 515 | +16.3 % | 74.5 % | 7.9 % | 17.6 % | 606 |
| 63-63A | roja | 23 428 | 22 239 | +5.3 % | 45.2 % | 18.1 % | 36.7 % | 542 |
| 64-45B | rosada | 32 605 | 26 736 | +22.0 % | 59.4 % | 7.3 % | 33.4 % | 755 |

**Comparación con la empresa (solo porcentajes; 5 camas válidas)**

| Prueba | Error medio por cama | Error del total |
|---|---:|---:|
| Tallos, factor flores/tallo dejando la cama fuera (línea de conteo) | 25.0 % (sesgo +6.3 %) | +1.4 % |
| Lo mismo con seguimiento | 35.9 % | −3.7 % |
| Sin la cama atípica 63-63A | 20.8 % | −4.1 % |
| Factor solo de la misma variedad (única variedad con 3 camas) | 14.2 % | −1.8 % |
| Día de corte con la moneda (2.03 cm) | 2.8 días (adelanta) | — |
| Día de corte con diámetro calibrado (1.48 × moneda) | 1.6 días | — |

- El factor flores/tallo varía 32 % entre variedades y es la mayor fuente de error.
- Con más camas por variedad, el error por cama baja a unos 14 %.
- La línea de conteo es más estable que el seguimiento.

## Cómo correrlo

```bash
conda activate pdi
python scripts/standardize_videos.py   # videos_raw -> videos_processed (10 fps, cuadro más nítido)
python scripts/inventory.py            # inventario + mosaico de muestra
python scripts/block_a.py              # A1-A7: normalización del video y control de calidad
python scripts/block_b.py --steps b12 b3   # B1-B3: solo corre si A7 abre la puerta (--ignore-gate para explorar)
python scripts/calibrate_colors.py     # color maduro por cama (necesita B3 de todas las camas)
python scripts/block_b.py --steps b4 b5    # B4-B5: cuándo se corta, resultados y summary_all_videos.csv
python scripts/import_company.py       # datos de la empresa (instr/data_oficial) -> data/ground_truth/company/
python scripts/compare_company.py      # flores -> tallos y comparación con el reporte de la empresa
```

O todo en orden con un solo comando, que se detiene en el primer error y se retoma con `--from`:

```bash
python scripts/run_all.py              # por ejemplo: python scripts/run_all.py --from calibrate
python -m unittest discover tests      # pruebas mínimas (no necesitan videos)
```

**Datos de la empresa: no se suben a git.** `import_company.py` los deja en `data/ground_truth/company/` (`reception_report.csv` y `beds_company.csv`, con variedad y fechas de corte), y todo lo que sale de ellos (`company_*`, `beds_flowers_stems.csv`) está en `.gitignore`. `data/videos.csv` solo tiene datos del proyecto: `video_id, bed_id, flower_color, record_date, rotate, exclude`. Las métricas del conteo de flores, que no usan datos de la empresa, están en `results/metrics/flower_count_metrics.json`.

Cada script acepta `--steps` (por ejemplo `python scripts/block_a.py --steps a4 a56 a7`) y `--video 63_40A`. Cada paso lee de disco lo que guardó el anterior, así que se puede repetir uno solo. Los parámetros están en `config/params.yaml`.

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
nv = NormalizedVideo("63_40A")
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
| B1/B2 Detección | `src/analysis/detection.py` | Perfil HSV por color → morfología → sin formas alargadas → transformada de distancia + watershed → racimos repartidos por área; diámetro, circularidad y color (HSV, Lab) | `results/tables/detections_*.csv` |
| B3 Contar una vez | `src/analysis/counting.py` | Correlación de fase → coordenada global → seguimiento (asignación húngara) + unión de rastros → se cuentan los rastros que cruzan una línea de conteo | `shifts_*.csv`, `flowers_line_*.csv` (principal), `flowers_*.csv` (verificación) |
| B4 Cuándo cortar | `src/analysis/counting.py`, `scripts/calibrate_colors.py` | Diámetro frente a la moneda de 100 COP (20.3 mm) + color calibrado por cama → hoy / mañana / otros días | columna `stage` |
| B5 Resultados | `scripts/block_b.py` | Totales por grupo, flores por metro, mosaico marcado, resumen de todas las camas y métricas del conteo (acuerdo entre métodos, densidad, CV por metro, error del largo recorrido) | `summary_*.json`, `summary_all_videos.csv`, `flower_count_metrics.json`, figuras `b5_*` |
| Comparación con la empresa | `scripts/compare_company.py` | Flores por tallo en las camas cortadas (variedad → tipo de flor → todas) → tallos estimados por cama y por grupo; error dejando una cama fuera; día de corte predicho frente al de la empresa y diámetro de corte calibrado con esas fechas | `beds_flowers_stems.csv`, `company_*.csv`, `company_comparison.json`, figuras `company_*` (fuera de git) |

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
│   ├── ground_truth/        # máscaras y conteos hechos a mano; company/ = datos de la empresa (fuera de git)
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
