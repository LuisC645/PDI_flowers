# Experimento: detector de aprendizaje profundo frente al clásico

Rama `exp/dl-detector`. Es una **prueba**, no parte del pipeline: no cambia la decisión 2.

Pregunta: ¿un detector pequeño (YOLO nano por recortes) cuenta distinto que el clásico
(HSV + watershed) sobre el mismo video normalizado?

Limitación de fondo: no hay etiquetas hechas a mano. El detector se entrena con
**pre-etiquetas del detector clásico**, así que aprende a imitarlo. La prueba sirve para
(1) dejar montado todo el flujo, (2) ver dónde difieren los dos y (3) dejar listos los
recortes para corregir a mano. La comparación real necesita esas correcciones y el
conteo manual por tramos (F1.4).

## Pasos

```bash
conda activate pdi-dl                               # entorno aparte (torch + ultralytics)
python experiments/dl_detector/build_dataset.py     # recortes + etiquetas YOLO -> data/dl/
python experiments/dl_detector/train.py             # entrena -> data/dl/runs/
python experiments/dl_detector/compare.py           # conteo y acuerdo -> results/
```

- Recortes de 320 px (se entrenan a 640: una flor de ~18 px queda en ~36 px).
- Entrenamiento y validación se separan por **posición en la cama** (cada flor se ve en
  ~40 cuadros seguidos; mezclar cuadros vecinos sería hacer trampa).
- El conteo usa la línea de conteo (franja central), igual para los dos detectores.
- `data/dl/review/`: recortes de validación con sus pre-etiquetas, para corregir a mano
  (formato YOLO; se abren con LabelImg, CVAT o Roboflow).
