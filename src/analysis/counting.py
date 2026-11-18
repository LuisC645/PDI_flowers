"""B3 contar cada flor una vez y B4 cuándo se corta cada flor.

B3: en el video normalizado el avance es casi solo en x. El desplazamiento entre cuadros
se mide con correlación de fase; al sumarlo, cada detección recibe una coordenada global
u (cm a lo largo de la cama) y v (cm a lo ancho). Dos métodos para comparar:
  track:   cada detección se enlaza con la del cuadro anterior más cercana en (u, v), a
           menos de eps_cm (asignación húngara). Una flor = un rastro con >= min_obs cuadros.
           (DBSCAN sobre todas las detecciones se probó y encadena flores vecinas: hay
           ~800 flores por cuadro.)
  line:    línea de conteo: se cuentan los rastros que cruzan una columna fija del cuadro
           (cada flor la cruza una sola vez); las medidas salen del rastro completo.
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
        d_cm_p75=("d_cm", lambda x: x.quantile(0.75)), circularity=("circularity", "median"),
        saturation=("saturation", "median"), hue=("hue", "median"), value=("value", "median"),
        L=("L", "median"), a=("a", "median"), b=("b", "median"),
        n_obs=("frame", "nunique"), first_frame=("frame", "min"), last_frame=("frame", "max"))
    return agg[agg.n_obs >= min_obs].reset_index()


def strip_flowers(det, shifts, line_x):
    """Línea de conteo: por cuadro, solo las detecciones en una franja vertical de ancho |dx|
    centrada en line_x (una columna donde la región útil está completa, lejos del camarógrafo)."""
    half = (shifts.set_index("frame")["dx"].abs() / 2).rename("half")
    d = det.join(half, on="frame")
    keep = (d["x"] - line_x).abs() < d["half"]
    return d[keep].drop(columns="half").reset_index(drop=True)


def crossing_tracks(det, line_x, min_obs=3):
    """Rastros que cruzan la línea de conteo: la flor está a un lado de line_x en algún cuadro y
    al otro lado en otro. Cada flor cruza una sola vez, así que se cuenta exactamente una vez,
    aunque su rastro esté partido (solo cuenta el pedazo que cruza) y aunque no se detecte
    justo en el cuadro del cruce. Devuelve los flower_id que cruzan."""
    side = np.sign(det["x"] - line_x)
    g = side.groupby(det["flower_id"])
    n = det.groupby("flower_id")["frame"].nunique()
    return g.min().index[(g.min() < 0) & (g.max() > 0) & (n >= min_obs)]


def stitch_tracks(det, max_gap, eps_cm):
    """Une rastros partidos: si uno termina y otro empieza poco después (<= max_gap cuadros)
    en el mismo lugar (<= eps_cm), son la misma flor. Empareja de menor a mayor distancia."""
    g = det.sort_values("frame").groupby("flower_id")
    ends = g.agg(f=("frame", "max"), u=("u_cm", "last"), v=("v_link_cm", "last"))
    starts = g.agg(f=("frame", "min"), u=("u_cm", "first"), v=("v_link_cm", "first"))
    tree = cKDTree(starts[["u", "v"]].values)
    cand = []
    for (fid, e), nb in zip(ends.iterrows(), tree.query_ball_point(ends[["u", "v"]].values, eps_cm)):
        for j in nb:
            sid = starts.index[j]
            gap = starts.f.iloc[j] - e.f
            if sid != fid and 0 < gap <= max_gap:
                cand.append((np.hypot(e.u - starts.u.iloc[j], e.v - starts.v.iloc[j]), fid, sid))
    parent = {}

    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    used_end, used_start = set(), set()
    for _, a, b in sorted(cand):
        if a in used_end or b in used_start or root(a) == root(b):
            continue
        parent[root(b)] = root(a)
        used_end.add(a)
        used_start.add(b)
    return det.assign(flower_id=det.flower_id.map(lambda x: root(x)))


def color_index(df, color):
    """Rasgo de color de flor abierta y madura: saturación (flores de color) o blancura =
    brillo - saturación (flor blanca: mientras más blanca y brillante, más abierta)."""
    if color == "white":
        return df["value"] - df["saturation"]
    return df["saturation"]


def classify_maturity(flowers, color, color_threshold, cfg, min_cut_ratio=1.0):
    """B4. Cuándo se corta cada flor: hoy, mañana u otros días (aproximación).

    - Tamaño: diámetro de corte = moneda de 100 COP x min_cut_ratio (según la flor).
    - Días para alcanzar ese tamaño, suponiendo que el diámetro crece growth_per_day por día:
      n = log(d_corte / d) / log(1 + g)  (0 si ya lo alcanzó).
    - Color: maduro si su índice de color >= color_threshold (calibrado por cama: solo las
      flores claramente más pálidas que su cama quedan como "no maduro").
    Etapas: hoy = tamaño y color listos; mañana = le falta como máximo un día de tamaño, o ya
    tiene el tamaño pero el color aún no está maduro; otros días = el resto.
    """
    d_cut = cfg["coin_diameter_cm"] * min_cut_ratio
    g = cfg["growth_per_day"]
    days = np.ceil(np.log(d_cut / flowers["d_cm"]) / np.log(1 + g)).clip(lower=0)
    days = days.clip(upper=cfg["max_days"])
    color_ok = color_index(flowers, color) >= color_threshold
    stage = np.where((days == 0) & color_ok, "today",
                     np.where((days <= 1) & (color_ok | (days == 0)), "tomorrow", "later"))
    return flowers.assign(d_cut_cm=d_cut, days_to_size=days.astype(int), color_ok=color_ok,
                          stage=stage)
