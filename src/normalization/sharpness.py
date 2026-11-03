"""Índice de nitidez de un cuadro."""
import cv2


def laplacian_variance(frame, downscale=0.5):
    """Varianza del Laplaciano en gris. Valores más altos indican un cuadro más nítido."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if downscale != 1:
        gray = cv2.resize(gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
