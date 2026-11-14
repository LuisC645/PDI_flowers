"""B3 contar cada flor una vez y B4 cortar / no cortar.  [BORRADOR: decisiones 8 y 9]

B3: en el video normalizado el avance es casi solo en x. El desplazamiento entre cuadros
se mide con correlación de fase; al sumarlo, cada detección recibe una coordenada global
u (cm a lo largo de la cama) y v (cm a lo ancho). Dos métodos para comparar:
  track:   cada detección se enlaza con la del cuadro anterior más cercana en (u, v), a
           menos de eps_cm (asignación húngara). Una flor = un rastro con >= min_obs cuadros.
           (DBSCAN sobre todas las detecciones se probó y encadena flores vecinas: hay
           ~800 flores por cuadro.)
  strip:   cada cuadro solo cuenta las flores de una franja central de ancho igual al
           avance de ese cuadro (línea de conteo); así cada punto pasa una sola vez.
"""
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def frame_shift(gray_a, gray_b, window):
    """Desplazamiento (dx, dy) de b respecto de a, y la respuesta (confianza) de la correlación."""
    (dx, dy), resp = cv2.phaseCorrelate(gray_a.astype(np.float32), gray_b.astype(np.float32), window)
    return dx, dy, resp


def global_offsets(shifts, min_response):
    """Desplazamiento acumulado por cuadro (px). Los pasos poco fiables se interpolan."""
    s = shifts.copy()
    bad = s["response"] < min_response
    s.loc[bad, ["dx", "dy"]] = np.nan
    s[["dx", "dy"]] = s[["dx", "dy"]].interpolate(limit_direction="both")
    s["offset_x"] = np.r_[0, np.cumsum(s["dx"].values[:-1])]
    s["offset_y"] = np.r_[0, np.cumsum(s["dy"].values[:-1])]  # temblor que A5 no quitó
    s["reliable"] = ~bad
    return s


def link_tracks(det, eps_cm, max_missing=2):
    """Enlaza detecciones entre cuadros consecutivos en coordenadas globales (u, v_link).

    v_link = v corregida con el dy acumulado de la correlación de fase (solo para enlazar;
    acumula deriva a lo largo del video, pero en la vida de un rastro es despreciable).
    Un rastro sigue vivo hasta max_missing cuadros sin detección (oclusión breve, blur).
    Devuelve det con la columna flower_id.
    """
    det = det.sort_values("frame").reset_index(drop=True)
    ids = np.full(len(det), -1)
    active = {}  # flower_id -> (u, v, último cuadro)
    next_id = 0
    for frame, grp in det.groupby("frame", sort=True):
        pts = grp[["u_cm", "v_link_cm"]].values
        active = {k: v for k, v in active.items() if frame - v[2] <= max_missing + 1}
        keys = list(active)
        assigned = np.full(len(grp), -1)
        if keys:
            prev = np.array([active[k][:2] for k in keys])
            pairs = cKDTree(pts).query_ball_tree(cKDTree(prev), eps_cm)
            rows = [i for i, p in enumerate(pairs) if p]
            cols = sorted({j for p in pairs for j in p})
            if rows:
                col_pos = {j: c for c, j in enumerate(cols)}
                cost = np.full((len(rows), len(cols)), 1e6)
                for r, i in enumerate(rows):
                    for j in pairs[i]:
                        cost[r, col_pos[j]] = np.hypot(*(pts[i] - prev[j]))
                rr, cc = linear_sum_assignment(cost)
                for r, c in zip(rr, cc):
                    if cost[r, c] < 1e6:
                        assigned[rows[r]] = keys[cols[c]]
        for i in np.flatnonzero(assigned < 0):
            assigned[i] = next_id
            next_id += 1
        for i, fid in enumerate(assigned):
            active[fid] = (pts[i, 0], pts[i, 1], frame)
        ids[grp.index] = assigned
    return det.assign(flower_id=ids)


def aggregate_flowers(det, min_obs):
    """Una fila por flor: mediana de posición, diámetro y color de sus observaciones."""
    agg = det.groupby("flower_id").agg(
        u_cm=("u_cm", "median"), v_cm=("v_cm", "median"), d_cm=("d_cm", "median"),
        saturation=("saturation", "median"), hue=("hue", "median"), value=("value", "median"),
        n_obs=("frame", "nunique"), first_frame=("frame", "min"), last_frame=("frame", "max"))
    return agg[agg.n_obs >= min_obs].reset_index()


def strip_flowers(det, shifts, out_w):
    """Línea de conteo: por cuadro, solo las detecciones en la franja central de ancho |dx|."""
    half = (shifts.set_index("frame")["dx"].abs() / 2).rename("half")
    d = det.join(half, on="frame")
    keep = (d["x"] - out_w / 2).abs() < d["half"]
    return d[keep].drop(columns="half").reset_index(drop=True)


def classify_cut(flowers, profile):
    """B4. Cortar si diámetro >= mínimo (moneda) y el rasgo de color >= umbral."""
    size_ok = flowers["d_cm"] >= profile["min_cut_diameter_cm"]
    color_ok = flowers[profile["cut_feature"]] >= profile["cut_min"]
    return flowers.assign(size_ok=size_ok, color_ok=color_ok, cut=size_ok & color_ok)
