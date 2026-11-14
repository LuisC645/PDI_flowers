"""Acceso a la salida del Bloque A: el "contrato" que usa el Bloque B.

Por cada video, el Bloque A deja:
  data/normalized/video/<v>.mp4                  video normalizado (cama horizontal, ancho W)
  data/normalized/transforms/<v>_normalized_meta.json  tamaño, margen, px/cm, paralaje
  data/normalized/transforms/<v>_transforms.csv  matriz 2x3 por cuadro + estado + color
  data/normalized/masks/<v>_operator_mask.png    zona del camarógrafo (cuadro original)
La región útil de cada cuadro normalizado (roi) es la franja de la cama menos la zona del
camarógrafo llevada a ese cuadro con su matriz.
"""
import json

import cv2
import numpy as np
import pandas as pd

from src.common.paths import MASKS_DIR, NORMALIZED_VIDEO_DIR, TRANSFORMS_DIR


class NormalizedVideo:
    """Lee la salida del Bloque A de un video y entrega la región útil de cada cuadro."""

    def __init__(self, stem):
        self.stem = stem
        self.video_path = NORMALIZED_VIDEO_DIR / f"{stem}.mp4"
        self.meta = json.loads((TRANSFORMS_DIR / f"{stem}_normalized_meta.json").read_text())
        self.transforms = pd.read_csv(TRANSFORMS_DIR / f"{stem}_transforms.csv")
        self.out_w, self.out_h = self.meta["out_w"], self.meta["out_h"]
        self.margin = self.meta["margin_px"]
        self.px_per_cm = self.meta["px_per_cm"]
        self.band = np.zeros((self.out_h, self.out_w), np.uint8)
        self.band[self.margin:self.out_h - self.margin] = 255
        op_path = MASKS_DIR / f"{stem}_operator_mask.png"
        self.operator = cv2.imread(str(op_path), cv2.IMREAD_GRAYSCALE) if op_path.exists() else None
        self.src_size = tuple(self.meta["src_size"])  # (ancho, alto) del cuadro original

    @property
    def valid_frames(self):
        """Cuadros utilizables (medidos o interpolados)."""
        return self.transforms.frame[self.transforms.status != "invalid"].tolist()

    def matrix(self, frame):
        """Matriz 2x3 cuadro original -> normalizado (NaN si el cuadro es inválido)."""
        row = self.transforms.iloc[frame]
        return np.array([[row.m00, row.m01, row.m02], [row.m10, row.m11, row.m12]])

    def operator_in_frame(self, frame):
        """Zona del camarógrafo en el cuadro normalizado (0/255)."""
        if self.operator is None:
            return np.zeros((self.out_h, self.out_w), np.uint8)
        M = self.matrix(frame)
        if np.isnan(M).any():
            return np.full((self.out_h, self.out_w), 255, np.uint8)
        return cv2.warpAffine(self.operator, M, (self.out_w, self.out_h), flags=cv2.INTER_NEAREST)

    def has_image(self, frame):
        """Píxeles del cuadro normalizado que vienen del original (las cuñas de la rotación no)."""
        M = self.matrix(frame)
        if np.isnan(M).any():
            return np.zeros((self.out_h, self.out_w), np.uint8)
        ones = np.full(self.src_size[::-1], 255, np.uint8)
        valid = cv2.warpAffine(ones, M, (self.out_w, self.out_h), flags=cv2.INTER_NEAREST)
        return cv2.erode(valid, np.ones((5, 5), np.uint8))  # sin el borde interpolado

    def roi(self, frame):
        """Región útil: franja de la cama, con imagen y sin el camarógrafo (0/255)."""
        roi = cv2.bitwise_and(self.band, self.has_image(frame))
        return cv2.bitwise_and(roi, cv2.bitwise_not(self.operator_in_frame(frame)))
