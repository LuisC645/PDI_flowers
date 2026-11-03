# Conteo y clasificación de flores en una cama de cultivo (Piloto PDI)

Proyecto de **Procesamiento Digital de Imágenes**. A partir de un video que recorre desde arriba una cama de flores de 32 m × 1.35 m, se busca:

1. Segmentar la cama.
2. Contar las flores (blancas, rojas, amarillas y naranjas).
3. Decidir cuáles están listas para cortar, según su tamaño y su color.

Primero se normaliza el video: se segmenta la cama, se endereza y se fija su escala. Después se analizan las flores.

**Equipo:** Luis Castillo y Benjamin Ruiz.

## Estructura de carpetas

```
proy/
├── README.md
├── environment.yml          # entorno conda "pdi"
├── config/                  # params.yaml y decisions.md
├── data/
│   ├── video/               # videos originales (no se modifican)
│   ├── frames/              # cuadros extraídos
│   ├── measurements/        # medidas de referencia (moneda, cama)
│   ├── ground_truth/        # máscaras y conteos hechos a mano
│   └── normalized/          # máscaras, transformaciones y video normalizado
├── src/
│   ├── common/              # utilidades compartidas
│   ├── normalization/       # normalización del video
│   └── analysis/            # análisis de flores
├── scripts/                 # ejecución de cada paso
├── notebooks/               # exploración
├── tests/
├── results/                 # tablas, métricas y figuras
└── report/
```

Todos los nombres de archivos, carpetas, variables y columnas están en inglés.

## Datos

Los videos y todo lo que hay en `data/` **no se suben a git** por su peso; solo se conserva la estructura de carpetas.