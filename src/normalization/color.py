"""A6. Normalización de color con el follaje como referencia.  [BORRADOR]

El follaje verde no depende del color de la flor y casi no cambia. Por cada cuadro se
calcula la media Lab del follaje dentro de la cama (en A2) y se corrige:
  L: ganancia  = L_ref / L_cuadro      (luz más o menos fuerte)
  a, b: desplazamiento = ref - cuadro  (dominante de color)
La referencia es la mediana de todo el video y la corrección se suaviza en el tiempo.
"""
import cv2
import numpy as np
import pandas as pd


def color_corrections(stats, window):
    """stats: DataFrame con foliage_L, foliage_a, foliage_b por cuadro. Devuelve gain_L, off_a, off_b."""
    ref = stats[["foliage_L", "foliage_a", "foliage_b"]].median()
    smooth = (stats[["foliage_L", "foliage_a", "foliage_b"]].interpolate(limit_direction="both")
              .rolling(window, center=True, min_periods=1).median())
    return pd.DataFrame({
        "frame": stats["frame"],
        "gain_L": (ref["foliage_L"] / smooth["foliage_L"]).round(4),
        "off_a": (ref["foliage_a"] - smooth["foliage_a"]).round(3),
        "off_b": (ref["foliage_b"] - smooth["foliage_b"]).round(3),
    })


def apply_correction(frame, gain_L, off_a, off_b):
    """Aplica la corrección en Lab (escala OpenCV de 8 bits) y vuelve a BGR."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[..., 0] *= gain_L
    lab[..., 1] += off_a
    lab[..., 2] += off_b
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
