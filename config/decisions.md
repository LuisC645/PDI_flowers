# Registro de decisiones

Registro de las decisiones del proyecto: opciones probadas, evidencia y decisión final.

Estados: **Pendiente** (sin trabajar) · **Propuesta** (implementada en el borrador, falta decidir en equipo) · **Decidida**.
Evidencia del borrador: video `63-40A`, corrida del 14 nov 2026.

| # | Tema | Opciones probadas | Métrica / evidencia | Decisión y razón | Fecha | Estado |
|---|---|---|---|---|---|---|
| 1 | Colores por video/cama | — | `63-40A` tiene un solo color (naranja, M21104-Reeses) según `data/videos.csv` | Un perfil por video, elegido con `flower_color`. Confirmar con los demás videos | 14 nov | Propuesta |
| 2 | Lenguaje, herramientas y aprendizaje profundo | Python + OpenCV 5.0, NumPy, SciPy, scikit-image, scikit-learn | El pipeline completo corre en ~5 min por video | Métodos clásicos, sin aprendizaje profundo (pocas referencias hechas a mano) | 14 nov | Propuesta |
| 3 | Segmentación de cama y estimación de bordes | (a) "no suelo" con ExG/Lab: **descartada**, no se ve suelo café entre camas. (b) brillo L suavizado + paralaje de movimiento, Otsu, componente central; bordes con RANSAC. (c) = (b) + fusión temporal del puntaje, respaldo de solo brillo si el paralaje rompe la máscara, y bordes buscados cerca de las rectas del cuadro anterior. Zona del camarógrafo: píxeles estáticos en el tiempo | Cuadros válidos en A3: (b) 87 % → (c) 94.7 %. Peso del paralaje probado: 0 → 54 %, 0.3 → 83 %, 0.5 → 87 %, 0.7 → 83 %, 1.0 → 15 % | (c) con peso 0.5. Falta medir IoU contra F1.4 (`scripts/evaluate_block_a.py`) | 14 nov | Propuesta |
| 4 | Seguimiento y estabilización | Atípicos con MAD global (k=4, k=5) y con filtro de Hampel (MAD local, k=5); Savitzky-Golay (21 cuadros) | MAD global k=4: 14.7 % de cuadros no medidos; k=5: 10.5 %; Hampel k=5: 9.3 %. El MAD global marcaba giros rápidos reales. `center_y` oscila con periodo ~2 s (paso del camarógrafo): es real y se compensa | Hampel k=5. Pendiente: suavizar con `time_s`; usar los postes blancos como referencia | 14 nov | Propuesta |
| 5 | Cuadros donde la cama no se detecta bien | Hueco ≤ 10 cuadros (1 s) → interpolar; más largo → inválido (cuadro negro, se conserva el índice) | 1.9 % inválidos (inicio, final y un giro en los cuadros 165–176) + 7.4 % interpolados = 9.3 % (meta A7: ≤ 10 %) | — | 14 nov | Propuesta |
| 6 | Ancho fijo de la cama normalizada | W = 1080 px (≈ ancho medido, ~1170 px); margen de 0.3·W arriba y abajo | 8 px/cm; una flor (~2.3 cm) mide ~18 px. Al volver a medir en el video normalizado el ancho da 1109 px (+2.7 %): la medida del ancho tiene un sesgo de ese orden según el contexto. El largo recorrido (32.0 m frente a 32 m) respalda la escala a lo largo | — | 14 nov | Propuesta |
| 7 | Compensación de paralaje | Sin compensar (k = 1) | El diámetro medio cambia con la posición a lo ancho: 2.0–2.2 cm en el centro y 2.6–2.9 cm en el borde inferior. Puede ser perspectiva (cámara inclinada) o un efecto real | Medir con la moneda en el centro y en los bordes, a la altura del dosel (F1.3). Considerar homografía en A5 | — | Pendiente |
| 8 | Criterio de color de corte por flor | Naranja: saturación media ≥ 150 | El 96 % de las flores pasa: con este umbral el color no discrimina | Calibrar con ejemplos etiquetados (curva ROC) | — | Pendiente |
| 9 | Conteo único entre cuadros | (a) DBSCAN sobre coordenada global: **descartado**, encadena flores vecinas (337 flores). (b) Enlace cuadro a cuadro con asignación húngara, rastro ≥ 10 cuadros: 27 885. (c) Franja central (línea de conteo): 20 393 | El largo recorrido sumado da 32.0 m (cama = 32 m) ✔. (b) y (c) difieren un ~27 % | Decidir con el conteo manual de los tramos (F1.4) | 14 nov | Propuesta |
| 10 | Tramos manuales y métricas de evaluación | — | — | — | — | Pendiente |
| 11 | Estructura exacta del informe | — | — | — | — | Pendiente |
| 12 | Rango de variación de altura de cámara | — | El ancho de la cama varía entre ~1100 y ~1240 px (±6 %) en la parte central del video | Usar como evidencia del rango de altura | 14 nov | Propuesta |

## Decisiones fuera de la tabla 2.2

| Tema | Decisión | Razón | Fecha |
|---|---|---|---|
| Estandarización a 10 fps | Se conserva 1 de cada 6 cuadros (el más nítido de la ventana). Excepción a la regla 4 ("todo cuadro se conserva") | El original se conserva en `videos_raw`, la nitidez de los 6697 cuadros está en `sharpness_*.csv` y cada cuadro conservado apunta a su origen en `frame_index_*.csv`. Efecto secundario: el tiempo entre cuadros no es uniforme (17–183 ms); usar `time_s` cuando importe | 2 nov |
| Color del video estandarizado | Se etiqueta como BT.601 (smpte170m) | OpenCV decodifica con BT.601 y ffmpeg codifica con BT.601: así los reproductores muestran el mismo color que OpenCV. Aplica a los videos que se procesen desde ahora | 14 nov |
| Metas de A7 | Ancho: CV ≤ 5 %. Centro: desviación ≤ 3 % de W. Ángulo: desviación ≤ 2°. Sacudida lateral: ≤ 1 % de W. Color: la desviación del L del follaje baja tras A6. Inválidos + interpolados ≤ 10 % | Son metas de borrador propuestas por Claude, **sin acordar en equipo**. Se fijaron antes de la corrida que pasó, pero el método se ajustó hasta cumplirlas en el único video disponible: el ángulo (1.75°) y los inválidos (9.3 %) pasan por poco. Confirmar con otro video | 14 nov |
| Resultado de A7 en `63-40A` | Puerta abierta | Ancho CV 1.9 %; centro 1.0 % de W; ángulo 1.75°; sacudida 3.6 px (0.45 cm); follaje L 1.81 → 1.42, a 1.02 → 0.38, b 1.17 → 0.36; región útil = 92.5 % de la franja. Pendiente manual: contaminación de la máscara (F1.4) | 14 nov |
