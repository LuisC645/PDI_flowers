"""A5. Normalización geométrica.  [BORRADOR: decisiones 6 y 7 sin cerrar]

Cada cuadro se rota (bordes horizontales), se escala (ancho de cama = W px) y se centra.
En el cuadro normalizado: x = a lo largo de la cama, y = a lo ancho. La cama ocupa las
filas [margin_px, margin_px + W). Escala: W / ancho real (cm) px/cm.
"""
import cv2
import numpy as np


def output_size(cfg, frame_width, median_width_px):
    """Tamaño fijo del cuadro normalizado: (ancho, alto, margen en px)."""
    W = cfg["bed_width_px"]
    margin = int(round(W * cfg["margin_frac"]))
    out_w = int(round(frame_width * W / median_width_px / 2) * 2)  # par, para H.264
    return out_w, W + 2 * margin, margin


def affine_matrix(row, cfg, out_w, out_h):
    """Matriz 2x3 que lleva el cuadro original al normalizado (rotación + escala + traslación)."""
    s = cfg["bed_width_px"] / row["width_px"]
    M = cv2.getRotationMatrix2D((row["center_x"], row["center_y"]), row["angle_deg"], s)
    M[0, 2] += out_w / 2 - row["center_x"]
    M[1, 2] += out_h / 2 - row["center_y"]
    return M


def px_per_cm(cfg):
    """Escala del cuadro normalizado: px/cm = W / ancho real de la cama (cm)."""
    return cfg["bed_width_px"] / cfg["bed_width_cm"]


def bed_band_mask(out_w, out_h, margin):
    """Máscara de la cama en el cuadro normalizado (franja central de alto W)."""
    mask = np.zeros((out_h, out_w), np.uint8)
    mask[margin:out_h - margin] = 255
    return mask
