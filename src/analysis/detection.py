"""B1 perfil de color y B2 detección y medición de flores.

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


def profile_pixels(hsv, profile):
    """Píxeles (uint8 0/255) que cumplen el perfil de color.

    profile: hue_ranges = [[lo, hi], ...] (escala OpenCV 0-180; el rojo usa dos rangos porque
    da la vuelta al círculo), s_min, s_max, v_min, v_max. La flor blanca usa todo el tono con
    saturación baja (s_max) y brillo alto (v_min).
    """
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in profile["hue_ranges"]:
        m |= cv2.inRange(hsv, np.array([lo, profile.get("s_min", 0), profile.get("v_min", 0)]),
                         np.array([hi, profile.get("s_max", 255), profile.get("v_max", 255)]))
    return m


def profile_mask(frame, profile, open_px, close_px, roi=None):
    """B1. Máscara binaria de los píxeles que cumplen el perfil de color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = profile_pixels(hsv, profile)
    if roi is not None:
        m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px,) * 2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px,) * 2))
    return ndi.binary_fill_holes(m > 0), hsv


def _drop_elongated(binary, max_len_px, max_elongation):
    """Quita componentes mucho más largos que anchos y más largos que una flor (postes, tallos)."""
    lab = ndi.label(binary)[0]
    out = binary.copy()
    for r in regionprops(lab):
        if r.axis_major_length > max_len_px and \
                r.axis_minor_length / max(r.axis_major_length, 1) < max_elongation:
            out[tuple(r.coords.T)] = False
    return out


def _split_clumps(labels, max_d_px, typical_d_px=None):
    """Racimos: regiones más grandes que una flor (pétalos de flores vecinas que se tocan sin
    espacio) se reparten en k = área / área típica partes con k-medias sobre sus píxeles.
    El área típica es la mediana de las flores aisladas y bien formadas del mismo cuadro.
    Es una aproximación: el número de flores del racimo se estima por su área."""
    props = regionprops(labels)
    max_area = np.pi * (max_d_px / 2) ** 2
    # área típica: flores aisladas y bien formadas (redondas y compactas), no pedazos de flor
    if typical_d_px:  # fijado en el perfil (flores cuyo perfil solo capta pedazos de pétalo)
        typical = np.pi * (typical_d_px / 2) ** 2
    else:
        good = np.array([r.area for r in props if r.area <= max_area and r.solidity >= 0.9
                         and 4 * np.pi * r.area / max(r.perimeter, 1) ** 2 >= 0.75])
        if len(good) < 10:
            return labels
        typical = np.median(good)
    out = labels.copy()
    next_label = labels.max() + 1
    for r in props:
        if r.area <= max_area:
            continue
        k = int(round(r.area / typical))
        if k < 2:
            continue
        pts = r.coords.astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, assign, _ = cv2.kmeans(pts, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        for j in range(k):
            out[tuple(r.coords[assign.ravel() == j].T)] = next_label
            next_label += 1
    return out


FEATURES = ["x", "y", "d_px", "d_cm", "area_px", "circularity", "solidity",
            "hue", "saturation", "value", "L", "a", "b"]


def _circular_hue(h_values):
    """Tono medio circular (escala OpenCV 0-180): el promedio de 179 y 1 es 0, no 90."""
    ang = h_values.astype(float) * (2 * np.pi / 180)
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return (m * 180 / (2 * np.pi)) % 180


def detect_flowers(frame, profile, cfg, px_cm, parallax_k=1.0, roi=None):
    """B2. Tabla de flores de un cuadro con sus características.

    Por flor: posición (px), diámetro equivalente (px y cm), área, circularidad (4πA/P², 1 =
    círculo), solidez (área / envolvente convexa), color medio en HSV (tono circular) y Lab.
    Un perfil puede sobrescribir los parámetros de detección (open_px, close_px,
    peak_min_distance_cm, min/max_diameter_cm, min_circularity).
    """
    c = {**cfg, **{k: v for k, v in profile.items() if k in cfg}}
    binary, hsv = profile_mask(frame, profile, c["open_px"], c["close_px"], roi)
    binary = _drop_elongated(binary, c["max_diameter_cm"] * px_cm, c["max_elongation"])
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    min_r = c["min_diameter_cm"] * px_cm / 2
    peaks = peak_local_max(dist, min_distance=max(1, int(c["peak_min_distance_cm"] * px_cm / 2)),
                           threshold_abs=min_r * 0.5, labels=ndi.label(binary)[0],
                           exclude_border=False)
    markers = np.zeros(dist.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-dist, markers, mask=binary)

    clump_d = profile.get("clump_diameter_cm")
    labels = _split_clumps(labels, c["max_diameter_cm"] * px_cm, clump_d * px_cm if clump_d else None)

    rows = []
    for r in regionprops(labels, intensity_image=np.dstack([hsv, lab])):
        d_px = np.sqrt(4 * r.area / np.pi)
        d_cm = d_px / px_cm / parallax_k
        if not c["min_diameter_cm"] <= d_cm <= c["max_diameter_cm"]:
            continue
        circ = 4 * np.pi * r.area / max(r.perimeter, 1) ** 2
        if circ < c["min_circularity"]:  # descarta formas alargadas (postes, tallos, brillos)
            continue
        _, s_, v_, L, a, b = r.intensity_mean
        hue = _circular_hue(hsv[..., 0][tuple(r.coords.T)])
        rows.append({"x": r.centroid[1], "y": r.centroid[0], "d_px": d_px, "d_cm": d_cm,
                     "area_px": r.area, "circularity": min(circ, 1.0), "solidity": r.solidity,
                     "hue": hue, "saturation": s_, "value": v_, "L": L, "a": a, "b": b})
    return pd.DataFrame(rows, columns=FEATURES)
