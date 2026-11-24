# Conteo y clasificación de flores en camas de cultivo (Piloto PDI)

Proyecto de **Procesamiento Digital de Imágenes** (Luis Castillo y Benjamin Ruiz). A partir de un video cenital de una cama de flores de 32 m × 1.35 m, el pipeline normaliza la cama (segmenta, endereza y fija la escala), cuenta las flores, decide cuáles se cortan hoy, mañana u otros días, y estima los tallos para compararlos con el reporte de la empresa.

Todo se corre con `python scripts/run_all.py` en el entorno conda `pdi` (`environment.yml`).

## Carpetas

```
proy/
├── config/        # params.yaml (parámetros) y decisions.md (decisiones tomadas)
├── data/          # videos y datos (fuera de git, salvo videos.csv y medidas)
│   ├── videos_raw/          # videos originales
│   ├── videos_processed/    # videos estandarizados (10 fps)
│   ├── normalized/          # máscaras, transformaciones y video normalizado
│   ├── measurements/        # medidas de referencia (moneda, cama)
│   └── ground_truth/        # cuadros para etiquetar; company/ = datos de la empresa (fuera de git)
├── src/
│   ├── common/              # rutas, metadatos y lectura de video
│   ├── normalization/       # Bloque A: normalización de la cama
│   └── analysis/            # Bloque B: detección, conteo y etapa de corte
├── scripts/       # un script por bloque, run_all.py y la comparación con la empresa
├── tests/         # pruebas unitarias (python -m unittest discover tests)
├── results/       # tablas, métricas y figuras generadas
├── docs/          # bitácora de trabajo
└── report/        # material del informe
```
