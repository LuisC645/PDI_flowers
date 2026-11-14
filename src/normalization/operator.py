"""A2 (complemento). Zona del camarógrafo (piernas y pies).  [BORRADOR]

El camarógrafo se mueve con la cámara: sus píxeles cambian poco en el tiempo, mientras la
escena pasa. No depende del color (sirve también con flores blancas):
  1. En ventanas de `window` cuadros se mide la desviación temporal de cada píxel; los
     píxeles "estáticos" (Otsu) se marcan.
  2. Se cuenta en qué fracción de las ventanas cada píxel fue estático.
  3. Zona del camarógrafo = la componente más grande, estática en >= min_freq de las
     ventanas y unida al borde inferior, convertida en una caja conservadora (las franjas
     oscuras entre camas también son algo estáticas, pero menos).
El resultado es una máscara fija por video, en coordenadas del cuadro original.
"""
import cv2
import numpy as np


def static_frequency(small_frames, window, step):
    """Fracción de ventanas en que cada píxel es estático. small_frames: (N, h, w, 3) uint8."""
    half = window // 2
    h, w = small_frames.shape[1:3]
    freq, n = np.zeros((h, w)), 0
    for c in range(half, len(small_frames) - half, step):
        sd = small_frames[c - half:c + half + 1].astype(np.float32).std(axis=0).mean(axis=2)
        sd = cv2.GaussianBlur(sd, (0, 0), 3)
        v = np.clip(sd / (np.percentile(sd, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)
        _, m = cv2.threshold(v, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        freq += m
        n += 1
    return freq / max(n, 1)


def operator_mask(freq, min_freq, extend_up_frac, dilate_px):
    """Zona del camarógrafo (uint8 0/255) como una caja conservadora.

    Núcleo: la componente más grande con frecuencia >= min_freq que toca el borde inferior
    (el pantalón oscuro). Las piernas se mueven al caminar y no son "estáticas", así que la
    zona es la caja del núcleo extendida hacia arriba extend_up_frac del alto y dilatada.
    Excluir de más no quita flores: cada flor del borde pasa por otras x del cuadro.
    """
    h, w = freq.shape
    m = (freq >= min_freq).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    touching = [i for i in range(1, n)
                if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= h - 1]
    out = np.zeros((h, w), np.uint8)
    if not touching:
        return out
    core = max(touching, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, bw = stats[core, cv2.CC_STAT_LEFT], stats[core, cv2.CC_STAT_TOP], stats[core, cv2.CC_STAT_WIDTH]
    top = max(0, int(y - extend_up_frac * h))
    out[top:, x:x + bw] = 255
    if dilate_px > 0:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_px + 1,) * 2))
    return out
