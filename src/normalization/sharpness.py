"""A1. Índice de nitidez de un cuadro y etiqueta nítido / borroso."""
import cv2
import numpy as np


def laplacian_variance(frame, downscale=0.5):
    """Varianza del Laplaciano en gris. Valores más altos indican un cuadro más nítido."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if downscale != 1:
        gray = cv2.resize(gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def label_blur(sharpness, window=31, rel_threshold=0.7):
    """Etiqueta cada cuadro comparando su nitidez con la mediana móvil de sus vecinos.

    El umbral es relativo porque la nitidez depende de la escena (más flores = más bordes).
    Devuelve (nitidez relativa, etiqueta "sharp"/"blurry").
    """
    ref = sharpness.rolling(window, center=True, min_periods=1).median()
    rel = sharpness / ref
    return rel.round(3), np.where(rel < rel_threshold, "blurry", "sharp")
