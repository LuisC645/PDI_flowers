# Bitácora del proyecto

Registro por día: qué se hizo, qué quedó a medias y qué sigue. Las decisiones y su evidencia están en `config/decisions.md`.

## 14 nov 2026

### Hecho

**Código existente (arreglos)**
- El tiempo de cada cuadro sale del contenedor (`CAP_PROP_POS_MSEC`), porque el video del celular tiene fps variable.
- Escritor de video compartido en `src/common/video.py`; el video se etiqueta como BT.601 (igual que lo decodifica OpenCV).
- A1 completo: etiqueta nítido/borroso (4 de 1117 cuadros borrosos).

**Bloque A, preprocesamiento (`scripts/block_a.py`)**
- A2: máscara de la cama con brillo + paralaje de movimiento (no se ve suelo café, así que la propuesta de "no suelo" no sirve). Fusión temporal y respaldo de solo brillo cuando la cámara gira sin avanzar.
- A2: zona del camarógrafo, detectada por los píxeles que no cambian en el tiempo.
- A3: bordes con RANSAC, buscados cerca de las rectas del cuadro anterior. 94.7 % de cuadros válidos.
- A4: filtro de Hampel + interpolación de huecos ≤ 1 s + Savitzky-Golay.
- A5/A6: video normalizado a 1080 px de ancho de cama (8 px/cm) y color igualado con el follaje.
- A7: **puerta abierta** en `63-40A`. Ancho CV 1.9 %, centro 1.0 % de W, ángulo 1.75° (meta ≤ 2°), inválidos + interpolados 9.3 % (meta ≤ 10 %).
- Contrato para el Bloque B: `src/common/normalized.py` (`NormalizedVideo`).

**Bloque B, borrador sin calibrar (`scripts/block_b.py`)**
- B1/B2: umbral HSV + watershed; unas 800 flores por cuadro.
- B3: largo recorrido 32.0 m (la cama mide 32 m). Conteo: 27 885 con seguimiento y 20 393 con línea de conteo.
- B4/B5: 48.7 % a cortar (no confiable), flores por metro y mosaico de la cama.

**Apoyo para las tareas manuales**
- `scripts/export_gt_frames.py`: exportó 18 cuadros a `data/ground_truth/frames/` (lista en `data/ground_truth/frames_63-40A.csv`).
- `scripts/evaluate_block_a.py`: IoU, Dice y contaminación contra las máscaras a mano.
- Plantilla de medidas: `data/measurements/measurements.csv`.

### A medias

**Experimento de aprendizaje profundo (rama `exp/dl-detector`, carpeta `experiments/dl_detector/`)**
- Listo: scripts (`build_dataset.py`, `train.py`, `compare.py`) y el conjunto de recortes en `data/dl/` (703 de entrenamiento y 141 de validación, con pre-etiquetas del detector clásico).
- **No se entrenó ni se comparó:** la instalación de torch/ultralytics se quedó colgada por internet lento. El entorno `pdi-dl` existe pero está incompleto.
- Para retomarlo:
  ```bash
  conda activate pdi-dl
  pip install ultralytics pandas pyyaml tqdm scipy scikit-image scikit-learn
  python experiments/dl_detector/train.py
  python experiments/dl_detector/compare.py
  ```
- Ojo: entrenado con pre-etiquetas, el detector aprende a imitar al clásico. La comparación real necesita recortes corregidos a mano (`data/dl/review/`) y el conteo manual.

### Pendiente para mañana (en orden)

1. **F1.4, máscaras a mano.** Pintar la cama en los 18 cuadros de `data/ground_truth/frames/`, guardarlas en `data/ground_truth/masks/<video>_<cuadro>.png` y correr `python scripts/evaluate_block_a.py`. Es lo único que falta para cerrar A7 del todo.
2. **F1.3, medidas.** Llenar `data/measurements/measurements.csv`: moneda con pie de rey (5 medidas), cama, y la moneda a la altura del dosel en el centro y en el borde (decisión 7).
3. **Conteo manual de 4–6 tramos de 1 m** sobre `results/figures/b5_mosaic_full_63-40A.jpg` (decisión 10): mitad para calibrar y mitad para probar. Sin esto no se puede elegir entre seguimiento y línea de conteo, que difieren un 27 %.
4. **Características por flor (B1/B2):** agregar forma (circularidad, solidez) y color en Lab a `detections_*.csv`.
5. **Calibrar el criterio de corte** (decisión 8): con saturación ≥ 150 pasa el 96 % de las flores, no discrimina.
6. Terminar el experimento de aprendizaje profundo, si el internet lo permite y el profesor lo aprueba (decisión 2).

### Puntos abiertos para discutir en equipo
- Las metas de A7 las propuso Claude y pasan por poco en ángulo e inválidos. Acordarlas y confirmarlas con otro video.
- El ancho de la cama, al volver a medirlo, da +2.7 % (1109 px frente a 1080): unos 0.06 cm en una flor de 2.3 cm, justo donde está el umbral de la moneda.
- El diámetro cambia con la posición a lo ancho (2.0–2.2 cm en el centro, 2.6–2.9 cm en el borde inferior): perspectiva o efecto real.
- Cronograma: hoy debería estar la Fase 3 (A5–A7 y B1). A y un borrador de B están; faltan las tareas manuales de la Fase 1.

## 17–18 nov 2026

### Hecho
- **8 videos nuevos** procesados (`63_38B`, `63_39A`, `63_39B`, `63_41A`, `63_58A`, `63_59A`, `63_63A`, `64_45B`). `63_40A` se excluyó por ser la misma cama que `63-40A`.
- **Estandarización:** ahora lee con ffmpeg, pasa el HDR del iPhone a SDR (libplacebo) y gira automáticamente los videos grabados en horizontal (`63_40A`, `64_45B`) para dejar al camarógrafo abajo. Giro, HDR y confianza quedan en `processing_log.csv`.
- **Zona del camarógrafo:** ya no exige tocar el borde inferior (en los videos nuevos aparece la mano o el zapato debajo del pantalón).
- **Bloque A con un solo modo para todos los videos**, elegido con `scripts/sweep_block_a.py` (compara configuraciones sobre los 9 videos a la vez): RANSAC con pendiente acotada, un punto de borde por columna y medición con un solo borde. Inválidos + interpolados: 14.9 % → 7.5 % en promedio.
- **A7:** mide con los dos bordes, usa dispersión robusta y cuenta los cuadros malos solo en el recorrido de la cama (`off_bed` aparte). **Los 9 videos abren la puerta.** Estos cambios se definieron después de ver los datos: están en `config/decisions.md` para que el equipo los apruebe.

### Pendiente
- Llenar en `data/videos.csv` la variedad, el color y las fechas de los videos nuevos (`white`, `orange`, `yellow`, `red`, `pink`).
- Aprobar en equipo los cambios en el cálculo de A7.
- Las tareas manuales siguen pendientes (`README_roadmap.md`, pasos 1–3); conviene hacer las máscaras a mano de al menos un video por color.
- Perfiles de color para blanca, amarilla, roja y rosada (Bloque B).
- Recortar los cuadros de transición al inicio y al final de cada cama, que hoy quedan medidos pero incluyen pasillo.

## 18 nov 2026 (Bloque B en los 9 videos)

### Hecho
- **Perfiles de color** de los 5 colores, calibrados con histogramas tono-saturación (`config/params.yaml`, `profiles`). Admiten varios rangos de tono (rojo y rosado dan la vuelta al círculo). El tono medio por flor ahora es circular.
- **Detección más robusta:** filtro de circularidad, eliminación de formas alargadas (poste blanco) y reparto de racimos por área. Nuevas características por flor: circularidad, solidez y color en Lab.
- **Conteo:** unión de rastros partidos y conteo principal por **rastros que cruzan una línea de conteo** ubicada donde la región útil está completa. El seguimiento se reporta como verificación.
- **Moneda:** 100 COP actual = 20.30 mm (Banco de la República).
- **Cuándo cortar:** hoy / mañana / otros días, con el tamaño frente a la moneda y el color calibrado por cama (`scripts/calibrate_colors.py`). Supuesto: el diámetro crece 10 % por día.
- Resumen de las 9 camas: `results/tables/summary_all_videos.csv`.

### Pendiente
- Conteo manual de tramos para medir el error del conteo (los dos métodos difieren hasta un 22 %).
- Grabar la misma cama dos días seguidos para calibrar el crecimiento diario.
- Flores etiquetadas (lista / no lista) para calibrar el color con curva ROC.
- Revisar si el diámetro de corte debe ser distinto por flor (la blanca es una margarita pequeña, de 1.8 cm de mediana).
