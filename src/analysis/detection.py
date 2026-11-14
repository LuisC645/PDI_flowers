"""B1 perfil de color y B2 detección y medición de flores.  [BORRADOR]

Umbral HSV del perfil -> apertura y cierre -> relleno de huecos (centro amarillo) ->
transformada de distancia -> máximos locales como semillas -> watershed (separa flores
pegadas) -> diámetro equivalente d = sqrt(4A/π).
"""
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


def profile_mask(frame, profile, open_px, close_px, roi=None):
    """B1. Máscara binaria de los píxeles que cumplen el perfil de color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(profile["hsv_lo"]), np.array(profile["hsv_hi"]))
    if roi is not None:
        m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px,) * 2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px,) * 2))
    return ndi.binary_fill_holes(m > 0), hsv


def detect_flowers(frame, profile, cfg, px_cm, parallax_k=1.0, roi=None):
    """B2. Tabla de flores de un cuadro: posición (px), diámetro (px y cm) y color medio (HSV)."""
    binary, hsv = profile_mask(frame, profile, cfg["open_px"], cfg["close_px"], roi)
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    min_r = cfg["min_diameter_cm"] * px_cm / 2
    peaks = peak_local_max(dist, min_distance=max(1, int(cfg["peak_min_distance_cm"] * px_cm / 2)),
                           threshold_abs=min_r * 0.5, labels=ndi.label(binary)[0],
                           exclude_border=False)
    markers = np.zeros(dist.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=binary)

    rows = []
    for r in regionprops(labels, intensity_image=hsv):
        d_px = np.sqrt(4 * r.area / np.pi)
        d_cm = d_px / px_cm / parallax_k
        if not cfg["min_diameter_cm"] <= d_cm <= cfg["max_diameter_cm"]:
            continue
        h, s, v = r.intensity_mean
        rows.append({"x": r.centroid[1], "y": r.centroid[0], "d_px": d_px, "d_cm": d_cm,
                     "hue": h, "saturation": s, "value": v})
    return pd.DataFrame(rows, columns=["x", "y", "d_px", "d_cm", "hue", "saturation", "value"])
