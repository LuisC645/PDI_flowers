"""Constantes y utilidades del experimento."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.paths import DATA_DIR  # noqa: E402

DL_DIR = DATA_DIR / "dl"
RUNS_DIR = DL_DIR / "runs"
TILE = 320          # px del recorte (video normalizado, 8 px/cm -> 40 cm)
IMGSZ = 640         # tamaño de entrenamiento (escala x2)
BOX_FACTOR = 1.2    # lado de la caja = BOX_FACTOR x diámetro equivalente
VAL_RANGE = (500, 600)   # cuadros de validación
BUFFER = 50         # cuadros que se descartan a cada lado de la validación (traslape)


def masked(frame, roi):
    """Cuadro con lo que está fuera de la región útil en negro."""
    out = frame.copy()
    out[roi == 0] = 0
    return out


def center_strip(nv):
    """Recorte central (x0, x1, y0, y1) que usa la línea de conteo."""
    cx = nv.out_w // 2
    return cx - TILE // 2, cx + TILE // 2, nv.margin, nv.out_h - nv.margin


def match(a, b, radius):
    """Empareja puntos de a y b a menos de radius px. Devuelve (n pares, solo a, solo b)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    if len(a) == 0 or len(b) == 0:
        return 0, len(a), len(b)
    d = cdist(a, b)
    d[d > radius] = 1e6
    r, c = linear_sum_assignment(d)
    n = int((d[r, c] < 1e6).sum())
    return n, len(a) - n, len(b) - n
