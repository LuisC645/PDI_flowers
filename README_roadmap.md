# Hoja de ruta: qué falta para terminar el piloto

Estado al 14 nov 2026. Cierre del proyecto: **25 nov 2026**.
Complementa a `README.md` (qué es y cómo se corre), `docs/worklog.md` (qué se hizo cada día) y `config/decisions.md` (decisiones y evidencia).

## ¿Con esto se termina el piloto?

Sí. El piloto queda terminado cuando se cumplan estas seis condiciones:

- [ ] El preprocesamiento está validado contra máscaras hechas a mano (IoU reportado).
- [ ] El conteo automático está comparado con el conteo manual en tramos de prueba (error reportado).
- [ ] El criterio cortar / no cortar está calibrado con ejemplos etiquetados y evaluado.
- [ ] Las 12 decisiones de `config/decisions.md` están cerradas, con su evidencia.
- [ ] El código está pasado a limpio y los resultados se regeneran con los scripts.
- [ ] El informe final y la presentación están entregados.

Alcance del piloto, para dejarlo explícito:

- **Cinco colores, 9 camas:** blanca (2), naranja (3), amarilla (2), roja (1) y rosada (1, perfil propio). Hay pocos videos por color: la calibración de cada perfil depende de una o dos camas.
- El Bloque A usa una sola configuración para todos los videos, elegida comparando sobre los 9 a la vez.
- Un piloto terminado no significa que el conteo sea exacto: significa que se sabe **cuánto se equivoca y por qué**, y qué cambiar en la versión grande.

## Dónde estamos

| Parte | Estado |
|---|---|
| Fase 1: inventario, cuadros, estructura, nitidez | Hecho |
| Fase 1: medidas (F1.3) y referencia a mano (F1.4) | **Falta (manual)** |
| Bloque A: normalización (A1–A7) | Hecho; A7 abre la puerta. Falta validar contra las máscaras a mano |
| Bloque B: detección, conteo, corte, resultados | Borrador que corre de punta a punta, **sin calibrar** |
| Validación y análisis de errores | Falta |
| Informe y presentación | Falta |

Ramas: el pipeline vive en `draft-pipeline`; las pruebas en `exp/...`. `main` no se toca hasta que el equipo lo decida.

---

## Paso a paso

Cada paso dice qué hacer, quién, y cuándo está listo. Los pasos 1 a 3 no dependen entre sí y bloquean el resto.

### Paso 0. Agregar los videos nuevos · equipo + código · 15 nov

Llegan más videos el 15 nov. Por cada uno:

1. Copiarlo a `data/videos_raw/` con el nombre de la cama (por ejemplo `63-41B.mp4`).
2. Agregar su fila en `data/videos.csv` (`video_id, bed_id, variety, flower_color, cut_start_date, cut_end_date`).
3. Correr `python scripts/standardize_videos.py`, `python scripts/inventory.py` y `python scripts/block_a.py --video <id>`.
4. Revisar `results/metrics/qc_block_a_<id>.json`: si A7 no abre la puerta, se ajusta el Bloque A antes de seguir.
5. Si el color no es naranja, falta su perfil en `profiles` de `config/params.yaml` (decisión 8) antes de correr `block_b.py`.

**Estado (18 nov):** hecho. 8 videos nuevos procesados (uno excluido por duplicado); A7 abre la puerta en los 9. Falta llenar la variedad, el color y las fechas en `videos.csv`, y que el equipo apruebe los cambios en el cálculo de A7 (`config/decisions.md`).

**Listo cuando:** todos los videos están en el inventario y se sabe en cuáles pasa A7. Con esto cambia el alcance (más de un video y, quizá, más de un color) y se confirman las metas de A7. Los pasos 1 a 3 se repiten para los videos que se vayan a validar.

### Paso 1. Máscaras de la cama a mano (F1.4) · equipo · 15–16 nov

1. Abrir los 18 cuadros de `data/ground_truth/frames/` (lista en `data/ground_truth/frames_63-40A.csv`). Si no están en este computador: `python scripts/export_gt_frames.py`.
2. Pintar la cama en blanco (255) y el resto en negro (0), del mismo tamaño que el cuadro (LabelMe, CVAT o GIMP).
3. Guardar en `data/ground_truth/masks/63-40A_<cuadro>.png` (mismo nombre que el cuadro).
4. Correr `python scripts/evaluate_block_a.py`.

**Listo cuando:** existe `results/metrics/mask_eval_63-40A.json` con IoU, Dice y contaminación, y el equipo decide si el resultado es aceptable (fijar la meta **antes** de mirar el número; propuesta: IoU ≥ 0.85 en la franja).

### Paso 2. Medidas de referencia (F1.3) · equipo · 15–16 nov

1. Medir la moneda de 100 COP con pie de rey, 5 veces.
2. Confirmar ancho (1.35 m) y largo (32 m) de la cama.
3. Anotar el diámetro mínimo de corte de la flor naranja respecto a la moneda.
4. Si se puede volver al cultivo: grabar o fotografiar la moneda a la altura del dosel, en el centro y en el borde de la cama (para el paralaje, decisión 7).
5. Llenar `data/measurements/measurements.csv`.

**Listo cuando:** el archivo no tiene valores vacíos (o se anota por qué falta alguno) y `profiles.orange.min_cut_diameter_cm` en `config/params.yaml` sale de ahí.

### Paso 3. Conteo manual por tramos (decisión 10) · equipo · 15–16 nov

1. Elegir 4–6 tramos de 1 m repartidos por la cama, sobre `results/figures/b5_mosaic_full_63-40A.jpg` (se regenera con `python scripts/block_b.py --steps b5`).
2. En cada tramo, contar todas las flores abiertas y cuántas se cortarían.
3. Marcar la mitad de los tramos como **calibración** y la otra mitad como **prueba**.
4. Guardar en `data/ground_truth/counts/segments_63-40A.csv` con columnas `start_m, end_m, total, cut, split`.

**Listo cuando:** el archivo existe y los tramos de prueba no se han usado para ajustar nada.

### Paso 4. Características por flor (B1/B2) · código · 17 nov

**Estado (18 nov):** hecho. Perfiles de los 5 colores, circularidad, solidez y color en Lab por flor.


- Agregar a `detections_*.csv`: circularidad, solidez y color medio en Lab.
- Corregir el diámetro con el factor de paralaje medido en el paso 2 (`normalize.parallax_k`).

**Listo cuando:** la tabla tiene las columnas nuevas y una muestra revisada a ojo coincide con las flores.

### Paso 5. Elegir el método de conteo (decisión 9) · código + equipo · 18 nov

- Escribir `scripts/evaluate_block_b.py`: compara el conteo automático con el manual, por tramo.
- Medir los dos métodos (seguimiento: 27 885; línea de conteo: 20 393) en los tramos de **calibración**.

**Listo cuando:** se elige un método con su error en calibración y se anota en `config/decisions.md`.

### Paso 6. Calibrar cortar / no cortar (decisión 8) · código + equipo · 18–19 nov

**Estado (18 nov):** versión sin etiquetas hecha. Moneda de 100 COP = 20.3 mm; grupos hoy / mañana / otros días; color calibrado por cama. Falta calibrar con flores etiquetadas y con dos grabaciones de la misma cama.


- Tamaño: umbral = diámetro de la moneda medido (paso 2).
- Color: elegir el rasgo y el umbral con los tramos de calibración (curva ROC). El umbral actual (saturación ≥ 150) deja pasar el 96 % de las flores.

**Listo cuando:** los umbrales están en `config/params.yaml` con su justificación en `config/decisions.md`.

### Paso 7. Validación y análisis de errores · código + equipo · 20–21 nov

Solo con los tramos de **prueba**:

- Error de conteo absoluto y relativo por tramo.
- Matriz de confusión, precisión, *recall* y F1 de cortar / no cortar.
- Subconteo por oclusión, reportado aparte.
- Para cada error grande: causa (oclusión, máscara, luz, blur, doble conteo) y qué se aprendió.

**Listo cuando:** las métricas están en `results/metrics/` y el análisis de errores está escrito.

### Paso 8. Cerrar decisiones · equipo · 21 nov

Revisar `config/decisions.md` y pasar cada fila a "Decidida". Las que requieren acuerdo o consulta:

- Decisión 2: ¿el curso permite aprendizaje profundo?
- Decisión 7: compensación de paralaje (depende del paso 2).
- Decisión 11: estructura exacta del informe.
- Metas de A7 (las propuso Claude; pasan por poco en ángulo e inválidos).

**Listo cuando:** no queda ninguna fila en "Pendiente" ni en "Propuesta".

### Paso 9. Pasar a limpio · código · 21–22 nov

- Quitar las marcas `[BORRADOR]` de los módulos cuyo método quedó decidido.
- Un notebook por paso con los resultados intermedios (regla 6 de las instrucciones).
- Pruebas mínimas en `tests/`.
- Correr todo desde cero y confirmar que los resultados se regeneran.

**Listo cuando:** `standardize_videos.py → block_a.py → block_b.py` corre sin `--ignore-gate` y reproduce las tablas.

### Paso 10. Informe y presentación · equipo · 23–25 nov

- Informe con las 8 secciones de `instr/README.md` (sección 6). Metodología y resultados salen de `config/decisions.md`, `docs/worklog.md` y `results/`.
- Recomendaciones para la versión grande: altura fija, velocidad constante, marcas cada metro, marcador de escala en el dosel, más de un video por color.
- Presentación.

**Listo cuando:** informe y presentación entregados el 25 nov.

---

## Opcional (solo si sobra tiempo)

- **Experimento de aprendizaje profundo** (rama `exp/dl-detector`): scripts y recortes listos, sin entrenar. Solo aporta con recortes corregidos a mano y con permiso del profesor. Cómo retomarlo: `docs/worklog.md`.
- **Mejoras del Bloque A:** usar los postes blancos como referencia; homografía en lugar de transformación afín; suavizar con el tiempo real de cada cuadro.
- **Otros colores:** requieren videos nuevos.

## Riesgos para la fecha

| Riesgo | Qué hacer |
|---|---|
| Las tareas manuales (pasos 1–3) se atrasan | Son la ruta crítica: sin ellas no hay calibración ni validación. Empezar por ahí |
| No se puede volver al cultivo a medir la moneda en el dosel | Reportar el paralaje como limitación y dar el diámetro con su incertidumbre |
| El conteo manual no favorece claramente a ningún método | Reportar los dos con su error; es un resultado válido para un piloto |
| El ancho re-medido tiene un sesgo de +2.7 % | Afecta el criterio de tamaño (≈ 0.06 cm en 2.3 cm). Incluirlo en el análisis de errores |
