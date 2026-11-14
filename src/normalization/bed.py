"""A2 máscara de la cama y A3 bordes de la cama.  [BORRADOR: decisión 3 sin cerrar]

No depende del color de la flor. En este video no se ve suelo café entre camas (la cama
vecina y el follaje la rodean), así que se combinan dos señales:
  1. Brillo L (Lab) muy suavizado: entre camas hay una franja de sombra oscura.
  2. Paralaje de movimiento: el dosel de la cama está más cerca de la cámara y, al
     caminar, se desplaza más que lo que está más lejos (cama vecina, suelo, follaje bajo).
     Se mide como el flujo óptico que queda tras quitar el movimiento global (afín).
Todo se calcula a escala reducida (bed.scale) y la geometría se devuelve en px originales.
"""
import cv2
import numpy as np


def _norm01(x, valid=None):
    """Lleva x a [0, 1] con los percentiles 2 y 98 (solo de los píxeles válidos, si se dan)."""
    lo, hi = np.percentile(x if valid is None else x[valid > 0], [2, 98])
    out = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    if valid is not None:
        out[valid == 0] = 0
    return out


def parallax_map(gray_a, gray_b, sigma):
    """Paralaje: flujo óptico menos su ajuste afín, proyectado en la dirección de avance.

    Devuelve (mapa de paralaje, desplazamiento global mediano en px).
    """
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    h, w = gray_a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    design = np.c_[xx.ravel(), yy.ravel(), np.ones(h * w)]
    residual = np.empty_like(flow)
    for k in range(2):
        coef, *_ = np.linalg.lstsq(design, flow[..., k].ravel(), rcond=None)
        residual[..., k] = flow[..., k] - (design @ coef).reshape(h, w)
    motion = np.median(flow.reshape(-1, 2), axis=0)
    direction = motion / (np.linalg.norm(motion) + 1e-6)
    par = residual[..., 0] * direction[0] + residual[..., 1] * direction[1]
    return cv2.GaussianBlur(par, (0, 0), sigma), float(np.linalg.norm(motion))


def _central_component(score, cfg, center_guess):
    """Otsu + apertura/cierre + componente que contiene el centro esperado. Devuelve (máscara, solidez)."""
    h, w = score.shape
    _, mask = cv2.threshold((score * 255).astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_kernel"],) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:  # componente que contiene el centro esperado (descarta la cama vecina)
        yc = int(center_guess) if center_guess is not None else h // 2
        lab = labels[np.clip(yc, 0, h - 1), w // 2]
        if lab == 0:
            lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.uint8(labels == lab) * 255
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt:
        return mask, 0.0
    c = max(cnt, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return mask, float(cv2.contourArea(c) / hull) if hull > 0 else 0.0


def bed_mask(small, small_next, cfg, center_guess=None, prev_score=None, valid=None):
    """A2. Máscara binaria (uint8 0/255) de la cama en un cuadro reducido.

    small: cuadro actual reducido (BGR).
    small_next: cuadro siguiente reducido, o lista de los siguientes: si con el primero no
                hay avance suficiente (min_motion_px), el paralaje se mide contra uno más
                lejano (más separación entre las dos vistas).
    center_guess: fila (px reducidos) donde se espera el centro de la cama (cuadro anterior).
    prev_score: puntaje fusionado del cuadro anterior; se promedia (EMA) con el actual
                para reducir el ruido del paralaje (cfg score_ema = peso del cuadro actual).
    valid: píxeles con imagen (en el video normalizado las cuñas negras de la rotación no
           cuentan para normalizar el puntaje, y quedan fuera de la máscara).
    Devuelve (máscara, peso del paralaje usado, puntaje fusionado).
    """
    L = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32)
    score = score_L = _norm01(cv2.GaussianBlur(L, (0, 0), cfg["blur_sigma"]), valid)
    w_par = 0.0
    nexts = small_next if isinstance(small_next, list) else [small_next]
    nexts = [x for x in nexts if x is not None]
    if nexts and cfg["w_parallax"] > 0:
        gray_a = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        for nxt in nexts:
            par, motion = parallax_map(gray_a, cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY),
                                       cfg["blur_sigma"])
            if motion >= cfg["min_motion_px"]:  # sin avance no hay paralaje fiable
                w_par = cfg["w_parallax"]
                score = (1 - w_par) * score + w_par * _norm01(par, valid)
                break

    alpha = cfg.get("score_ema", 1.0)
    if prev_score is not None and alpha < 1:
        score = alpha * score + (1 - alpha) * prev_score
        score_L = alpha * score_L + (1 - alpha) * prev_score
    # Dos candidatas: brillo + paralaje y solo brillo. Si la cámara gira sin avanzar, el
    # paralaje es ruido y rompe la máscara; se queda la más compacta (mayor solidez).
    mask, solidity = _central_component(score, cfg, center_guess)
    if w_par > 0:
        mask_L, solidity_L = _central_component(score_L, cfg, center_guess)
        if solidity_L > solidity + cfg["solidity_margin"]:
            mask, w_par, score = mask_L, 0.0, score_L
    return mask, w_par, score


def _edge_points(mask, center_row, step, search):
    """Por cada columna, el tramo de máscara que contiene la fila central: (x, y_sup, y_inf).

    Los puntos que tocan el borde de la imagen no son borde de cama y se descartan (NaN).
    """
    h, w = mask.shape
    xs, tops, bots = [], [], []
    for x in range(0, w, step):
        col = mask[:, x] > 0
        yc = int(center_row)
        if not col[yc]:  # busca el píxel de máscara más cercano al centro
            on = np.flatnonzero(col[max(0, yc - search):yc + search]) + max(0, yc - search)
            if on.size == 0:
                continue
            yc = on[np.argmin(np.abs(on - yc))]
        off_up = np.flatnonzero(~col[:yc])
        off_dn = np.flatnonzero(~col[yc:])
        top = off_up[-1] + 1 if off_up.size else np.nan
        bot = yc + off_dn[0] - 1 if off_dn.size else np.nan
        xs.append(x)
        tops.append(top if top > 0 else np.nan)
        bots.append(bot if bot < h - 1 else np.nan)
    return np.array(xs, float), np.array(tops, float), np.array(bots, float)


def ransac_line(x, y, iters, tol, rng):
    """Ajusta y = a·x + b con RANSAC y refina por mínimos cuadrados. Devuelve (a, b, % inliers)."""
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan, np.nan, 0.0
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(x), 2, replace=False)
        if x[i] == x[j]:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        inl = np.abs(y - (a * x + y[i] - a * x[i])) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    a, b = np.polyfit(x[best], y[best], 1)
    return a, b, best.sum() / len(x)


def _apply_prior(xs, ys, line, tol):
    """Descarta los puntos de borde a más de tol px de la recta previa (a, b) en px reducidos."""
    if line is None:
        return ys
    a, b = line
    far = np.abs(ys - (a * xs + b)) > tol
    out = ys.copy()
    out[far] = np.nan
    return out if np.sum(~np.isnan(out)) >= 5 else ys


def _drop_at_invalid(xs, ys, valid, offset):
    """Descarta los puntos de borde cuyo vecino de afuera (y + offset) no tiene imagen."""
    out = ys.copy()
    ok = ~np.isnan(ys)
    yy = np.clip(ys[ok] + offset, 0, valid.shape[0] - 1).astype(int)
    bad = valid[yy, xs[ok].astype(int)] == 0
    idx = np.flatnonzero(ok)[bad]
    out[idx] = np.nan
    return out


def bed_edges(mask, cfg, scale, center_guess=None, seed=0, prior=None, valid=None):
    """A3. Geometría de la cama a partir de su máscara (en px del cuadro original).

    La cama cruza el cuadro horizontalmente; los bordes son las rectas superior e inferior.
    angle_deg: inclinación media de los bordes; center_y: centro de la cama en x = mitad;
    width_px: distancia perpendicular entre bordes; valid: pasa las comprobaciones.
    prior: geometría (dict de esta función) del último cuadro válido; si se da, solo se usan
           los puntos de borde a menos de prior_tol (fracción del alto) de sus rectas.
    valid: máscara de píxeles con imagen (en el video normalizado, las cuñas negras que deja
           la rotación no son imagen); los bordes pegados a zonas sin imagen se descartan.
    """
    h, w = mask.shape
    rng = np.random.default_rng(seed)
    yc = center_guess if center_guess is not None else h / 2
    xs, tops, bots = _edge_points(mask, yc, cfg["edge_step"], int(cfg["search_frac"] * h))
    if valid is not None:
        tops = _drop_at_invalid(xs, tops, valid, -2)
        bots = _drop_at_invalid(xs, bots, valid, +2)
    if prior is not None and cfg.get("prior_tol"):
        tol = cfg["prior_tol"] * h
        tops = _apply_prior(xs, tops, (prior["top_a"], prior["top_b"] * scale), tol)
        bots = _apply_prior(xs, bots, (prior["bottom_a"], prior["bottom_b"] * scale), tol)
    a_t, b_t, in_t = ransac_line(xs, tops, cfg["ransac_iters"], cfg["ransac_tol"], rng)
    a_b, b_b, in_b = ransac_line(xs, bots, cfg["ransac_iters"], cfg["ransac_tol"], rng)

    xm = w / 2
    angle = np.degrees(np.arctan((a_t + a_b) / 2))
    center = ((a_t * xm + b_t) + (a_b * xm + b_b)) / 2
    width = ((a_b * xm + b_b) - (a_t * xm + b_t)) * np.cos(np.radians(angle))
    diff = abs(np.degrees(np.arctan(a_t) - np.arctan(a_b)))
    lo, hi = cfg["width_frac_range"]
    checks = {
        "inliers": min(in_t, in_b) >= cfg["min_inlier_frac"],
        "parallel": diff <= cfg["max_edge_angle_diff_deg"],
        "angle": abs(angle) <= cfg["max_abs_angle_deg"],
        "width": lo * h <= width <= hi * h,
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "angle_deg": angle, "center_x": xm / scale, "center_y": center / scale,
        "width_px": width / scale, "edge_angle_diff_deg": diff,
        "inliers_top": in_t, "inliers_bottom": in_b,
        "top_a": a_t, "top_b": b_t / scale, "bottom_a": a_b, "bottom_b": b_b / scale,
        "valid": not failed, "fail_reason": ",".join(failed),
    }


def foliage_lab_stats(small, mask, cfg):
    """A6 (apoyo). Media Lab del follaje verde dentro de la cama; no depende del color de la flor."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sel = cv2.inRange(hsv, np.array(cfg["foliage_hsv_lo"]), np.array(cfg["foliage_hsv_hi"]))
    sel = (sel > 0) & (mask > 0)
    if sel.sum() < 100:
        return np.nan, np.nan, np.nan
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)[sel].astype(float)
    return tuple(lab.mean(axis=0))
