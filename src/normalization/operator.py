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


def operator_mask(freq, min_freq, extend_up_frac, dilate_px, bottom_frac=0.15):
    """Zona del camarógrafo (uint8 0/255) como una caja conservadora.

    Núcleo: la componente más grande con frecuencia >= min_freq cuyo borde de abajo cae en
    el último bottom_frac del alto y que está centrada en horizontal (el pantalón oscuro). Las piernas se mueven al caminar y no son "estáticas", así que la
    zona es la caja del núcleo extendida hacia arriba extend_up_frac del alto y dilatada.
    Excluir de más no quita flores: cada flor del borde pasa por otras x del cuadro.
    """
    h, w = freq.shape
    m = (freq >= min_freq).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    # Cerca del borde inferior (no necesariamente tocándolo: a veces la mano o el zapato,
    # que se mueven, quedan entre el pantalón y el borde) y centrada en horizontal.
    touching = [i for i in range(1, n)
                if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= (1 - bottom_frac) * h
                and w * 0.2 <= stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2 <= w * 0.8]
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


def operator_side(freq, strip_frac=0.12):
    """Borde del cuadro donde está el camarógrafo, por la frecuencia de píxeles estáticos.

    Mira una franja de cada borde (su mitad central) y elige la de mayor frecuencia media.
    Devuelve (borde, giro horario en grados para dejarlo abajo, confianza = mejor / segundo).
    """
    h, w = freq.shape
    sh, sw = max(1, int(strip_frac * h)), max(1, int(strip_frac * w))
    cx, cy = slice(w // 4, 3 * w // 4), slice(h // 4, 3 * h // 4)
    score = {"bottom": freq[h - sh:, cx].mean(), "top": freq[:sh, cx].mean(),
             "right": freq[cy, w - sw:].mean(), "left": freq[cy, :sw].mean()}
    rotation = {"bottom": 0, "left": 270, "top": 180, "right": 90}
    best, second = sorted(score, key=score.get, reverse=True)[:2]
    return best, rotation[best], float(score[best] / max(score[second], 1e-6))


def detect_orientation(video_path, scale=0.25, n_chunks=5, chunk_len=31):
    """Giro horario (0/90/180/270) que deja al camarógrafo abajo, y la confianza de la detección.

    Lee n_chunks tramos de chunk_len cuadros seguidos (OpenCV ya aplica la etiqueta de
    rotación), mide la frecuencia de píxeles estáticos y busca el borde del camarógrafo.
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    freqs = []
    for start in np.linspace(0.15 * n, 0.85 * n - chunk_len, n_chunks).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames = []
        for _ in range(chunk_len):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        if len(frames) >= 21:
            freqs.append(static_frequency(np.stack(frames), 21, 5))
    cap.release()
    side, rotation, confidence = operator_side(np.mean(freqs, axis=0))
    return side, rotation, confidence
