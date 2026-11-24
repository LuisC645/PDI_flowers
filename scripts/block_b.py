"""Bloque B: análisis de flores (B1-B5) sobre el video normalizado.  [BORRADOR]

Pasos:
  b12 perfil de color + detección y medición  -> results/tables/detections_<v>.csv
                                                 results/figures/b2_detections_<v>.jpg
  b3  desplazamiento entre cuadros y conteo único -> results/tables/shifts_<v>.csv
                                                     results/tables/flowers_<v>.csv
  b4  cortar / no cortar                      -> (columna cut en flowers_<v>.csv)
  b5  resultados                               -> results/metrics/summary_<v>.json
                                                 results/tables/per_meter_<v>.csv
                                                 results/figures/b5_*_<v>.png/jpg
El perfil se elige con la columna flower_color de data/videos.csv.
Regla del proyecto: no se pasa al Bloque B si A7 no abrió la puerta (usar --ignore-gate
solo para explorar el borrador).
Uso:  python scripts/block_b.py --ignore-gate
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.counting import (aggregate_flowers, classify_maturity, crossing_tracks,  # noqa: E402
                                   frame_shift, global_offsets, link_tracks, stitch_tracks)
from src.analysis.detection import detect_flowers  # noqa: E402
from src.common.normalized import NormalizedVideo  # noqa: E402
from src.common.paths import (FIGURES_DIR, METRICS_DIR, NORMALIZED_VIDEO_DIR, TABLES_DIR,  # noqa: E402
                              TRANSFORMS_DIR, VIDEOS_METADATA, excluded_videos, list_videos,
                              load_metadata, load_params)
from src.common.video import iter_frames, read_frames  # noqa: E402

STEPS = ["b12", "b3", "b4", "b5"]


def context(stem, P):
    """Datos comunes: salida del Bloque A, perfil de color y estado de cada cuadro."""
    nv = NormalizedVideo(stem)
    meta = load_metadata()
    if stem not in meta.index or not meta.loc[stem, "flower_color"]:
        raise SystemExit(f"{stem}: needs a row with flower_color in {VIDEOS_METADATA.name}")
    color = meta.loc[stem, "flower_color"]
    return nv.meta, P["profiles"][color], color, nv.transforms[["frame", "status"]], nv


def _detect_job(args):
    i, frame, profile, cfg, px_cm, k, roi = args
    d = detect_flowers(frame, profile, cfg, px_cm, k, roi)
    return d.assign(frame=i)


def step_b12(stem, P):
    """B1 + B2: detecta y mide las flores de cada cuadro válido."""
    meta, profile, color, status, nv = context(stem, P)
    video = nv.video_path
    valid = set(nv.valid_frames)
    # roi: franja de la cama, con imagen y sin la zona del camarógrafo
    jobs = ((i, f, profile, P["detection"], meta["px_per_cm"], meta["parallax_k"], nv.roi(i))
            for i, f in iter_frames(video) if i in valid)
    with Pool() as pool:
        parts = list(tqdm(pool.imap(_detect_job, jobs, chunksize=4), total=len(valid), desc="B2"))
    det = pd.concat(parts, ignore_index=True)
    det["v_cm"] = (det["y"] - meta["margin_px"]) / meta["px_per_cm"]
    det.round(3).to_csv(TABLES_DIR / f"detections_{stem}.csv", index=False)
    per_frame = det.groupby("frame").size()
    print(f"B2 ({color}): {len(det)} detections, {per_frame.median():.0f} per frame (median), "
          f"d_cm median {det.d_cm.median():.2f}")

    pick = sorted(valid)[len(valid) // 4::max(1, len(valid) // 4)][:3]
    tiles = []
    for i, fr in zip(pick, read_frames(video, pick)):
        for r in det[det.frame == i].itertuples():
            cv2.circle(fr, (int(r.x), int(r.y)), int(r.d_px / 2), (255, 0, 255), 2)
        tiles.append(fr)
    cv2.imwrite(str(FIGURES_DIR / f"b2_detections_{stem}.jpg"), np.hstack(tiles),
                [cv2.IMWRITE_JPEG_QUALITY, 80])


def counting_line(nv, n_sample=40):
    """Columna del cuadro normalizado donde la región útil está más completa (fuera de la zona
    del camarógrafo y de las cuñas de la rotación); entre empates, la más cercana al centro."""
    frames = nv.valid_frames[::max(1, len(nv.valid_frames) // n_sample)]
    cov = np.mean([(nv.roi(i)[nv.margin:nv.out_h - nv.margin] > 0).mean(axis=0) for i in frames], axis=0)
    best = np.flatnonzero(cov >= cov.max() - 0.005)
    return int(best[np.argmin(np.abs(best - nv.out_w / 2))])


def step_b3(stem, P):
    """B3: desplazamiento entre cuadros, coordenada global y conteo único."""
    meta, _, _, status, nv = context(stem, P)
    cfg, sc = P["counting"], P["counting"]["phase_scale"]
    m = meta["margin_px"]
    rows, prev = [], None
    window = None
    for i, fr in tqdm(iter_frames(NORMALIZED_VIDEO_DIR / f"{stem}.mp4"), total=len(status), desc="B3"):
        g = cv2.cvtColor(cv2.resize(fr[m:meta["out_h"] - m], None, fx=sc, fy=sc,
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if window is None:
            window = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        usable = ~status.status.isin(["invalid", "off_bed"])
        ok = prev is not None and usable[i] and usable[i - 1]
        dx, dy, resp = frame_shift(prev, g, window) if ok else (np.nan, np.nan, 0.0)
        rows.append({"frame": i - 1, "dx": dx / sc, "dy": dy / sc, "response": resp})
        prev = g
    rows.append({"frame": len(status) - 1, "dx": np.nan, "dy": np.nan, "response": 0.0})
    shifts = global_offsets(pd.DataFrame(rows[1:]), cfg["min_response"])
    det = pd.read_csv(TABLES_DIR / f"detections_{stem}.csv").drop(
        columns=["offset_x", "offset_y", "u_cm", "v_link_cm", "flower_id"], errors="ignore")
    det = det.join(shifts.set_index("frame")[["offset_x", "offset_y"]], on="frame")
    # a: coordenada global alineada con la imagen; u: igual pero creciente en el sentido
    # en que camina el camarógrafo (si la escena se mueve a la derecha, se camina a la izquierda)
    a_px = det["x"] - det["offset_x"]
    walk_left = shifts["dx"].median() > 0
    shifts["a_min_px"], shifts["a_max_px"], shifts["walk_left"] = a_px.min(), a_px.max(), walk_left
    shifts.round(4).to_csv(TABLES_DIR / f"shifts_{stem}.csv", index=False)
    u_px = a_px.max() - a_px if walk_left else a_px - a_px.min()
    det["u_cm"] = u_px / meta["px_per_cm"]
    det["v_link_cm"] = (det["y"] - det["offset_y"]) / meta["px_per_cm"]
    det = link_tracks(det, cfg["eps_cm"], cfg["max_missing"])
    n_raw = det.flower_id.nunique()
    det = stitch_tracks(det, cfg["stitch_gap"], cfg["stitch_eps_cm"])
    flowers = aggregate_flowers(det, cfg["min_obs"])
    line_x = counting_line(nv)
    # Línea de conteo: decide QUÉ flores se cuentan (cada punto de la cama la cruza una vez).
    # Las medidas de cada una salen de su rastro completo (mediana de ~30 observaciones).
    tracks = aggregate_flowers(det, 1).set_index("flower_id")
    strip = tracks.loc[crossing_tracks(det, line_x, cfg["line_min_obs"])].reset_index()
    print(f"B3: tracks {n_raw} -> {det.flower_id.nunique()} after stitching; counting line at x={line_x}")
    det.round(3).to_csv(TABLES_DIR / f"detections_{stem}.csv", index=False)
    # se guardan los dos conteos: seguimiento (flowers) y línea de conteo (flowers_line)
    flowers.round(3).to_csv(TABLES_DIR / f"flowers_{stem}.csv", index=False)
    strip.round(3).to_csv(TABLES_DIR / f"flowers_line_{stem}.csv", index=False)

    length_m = det.u_cm.max() / 100
    print(f"B3: travelled length {length_m:.1f} m (bed = 32 m), "
          f"unreliable shifts {(~shifts.reliable).mean():.1%}, mean |dy| {shifts.dy.abs().mean():.1f} px")
    print(f"B3: unique flowers  track = {len(flowers)}   strip = {len(strip)}   "
          f"(method used: {cfg['method']})")


def step_b4(stem, P):
    """B4: cuándo se corta cada flor (hoy, mañana, otros días), con el umbral de color calibrado."""
    _, profile, color, _, _ = context(stem, P)
    cal_path = METRICS_DIR / "color_calibration.json"
    if not cal_path.exists():
        raise SystemExit("Run scripts/calibrate_colors.py first (after the b3 step of every video)")
    cal = json.loads(cal_path.read_text())
    if stem not in cal:
        raise SystemExit(f"{stem}: no color calibration; run b3 for this video and then "
                         "scripts/calibrate_colors.py before b4")
    thr = cal[stem]["threshold"]  # calibrado por cama
    for name in ("flowers", "flowers_line"):
        path = TABLES_DIR / f"{name}_{stem}.csv"
        f = classify_maturity(pd.read_csv(path), color, thr, P["cut"], profile.get("min_cut_ratio", 1.0))
        f.round(3).to_csv(path, index=False)
        print(f"B4 ({name}): " + ", ".join(f"{k} {v}" for k, v in f.stage.value_counts().items()))


def build_mosaic(stem, meta, shifts, scale):
    """Mosaico de la cama: pega la franja central de cada cuadro en su posición global.

    Se arma en la coordenada de la imagen y, si se caminó hacia la izquierda, se voltea para
    que la columna c corresponda a u = c / (px_per_cm · scale). Los extremos se cubren con el
    primer y el último cuadro válidos completos.
    """
    m, out_w = meta["margin_px"], meta["out_w"]
    H = meta["out_h"] - 2 * m
    sh = shifts.set_index("frame")
    a_min, a_max = sh["a_min_px"].iloc[0], sh["a_max_px"].iloc[0]
    to_global = lambda i, x: (x - sh.offset_x[i] - a_min) * scale  # noqa: E731
    usable = sh.index[sh.dx.notna()]
    length = int((a_max - a_min) * scale) + 1
    mosaic = np.zeros((int(H * scale), length, 3), np.uint8)
    first, last = usable[0], usable[-1]
    for i, fr in iter_frames(NORMALIZED_VIDEO_DIR / f"{stem}.mp4"):
        if i not in usable or not fr.any():
            continue
        if i in (first, last):
            xa, xb = 0, out_w
        else:
            half = abs(sh.dx[i]) + 2  # franjas traslapadas (el avance cambia entre cuadros)
            xa, xb = int(out_w / 2 - half), int(np.ceil(out_w / 2 + half))
        g0, g1 = int(round(to_global(i, xa))), int(round(to_global(i, xb)))
        g0c, g1c = max(g0, 0), min(g1, length)
        if g1c - g0c < 1:
            continue
        strip = cv2.resize(fr[m:m + H, xa:xb], (g1 - g0, mosaic.shape[0]),
                           interpolation=cv2.INTER_AREA)
        mosaic[:, g0c:g1c] = strip[:, g0c - g0:g1c - g0]
    return mosaic[:, ::-1].copy() if sh["walk_left"].iloc[0] else mosaic


STAGE_BGR = {"today": (0, 200, 0), "tomorrow": (0, 215, 255), "later": (255, 120, 0)}
STAGE_ES = {"today": "hoy", "tomorrow": "mañana", "later": "otros días"}


def step_b5(stem, P):
    """B5: conteo total, cuándo se corta, distribución por metro y mosaico marcado."""
    meta, profile, color, _, _ = context(stem, P)
    main = "flowers" if P["counting"]["method"] == "track" else "flowers_line"
    flowers = pd.read_csv(TABLES_DIR / f"{main}_{stem}.csv")
    other = pd.read_csv(TABLES_DIR / f"{'flowers_line' if main == 'flowers' else 'flowers'}_{stem}.csv")
    shifts = pd.read_csv(TABLES_DIR / f"shifts_{stem}.csv")
    bin_m = P["results"]["bin_m"]
    flowers["meter"] = (flowers.u_cm / 100 // bin_m * bin_m).astype(int)
    per_m = flowers.pivot_table(index="meter", columns="stage", values="u_cm", aggfunc="size",
                                fill_value=0).reindex(columns=list(STAGE_BGR), fill_value=0)
    per_m["total"] = per_m.sum(axis=1)
    per_m.reset_index().to_csv(TABLES_DIR / f"per_meter_{stem}.csv", index=False)
    stages = flowers.stage.value_counts()
    days = flowers.loc[flowers.stage == "later", "days_to_size"].value_counts().sort_index()
    summary = {
        "video": stem, "flower_color": color, "method": P["counting"]["method"],
        "total_flowers": int(len(flowers)),
        "total_flowers_other_method": int(len(other)),
        "methods_diff_pct": round(100 * (len(flowers) / max(len(other), 1) - 1), 1),
        **{f"{k}": int(stages.get(k, 0)) for k in STAGE_BGR},
        **{f"{k}_pct": round(100 * stages.get(k, 0) / max(len(flowers), 1), 1) for k in STAGE_BGR},
        "later_by_days": {int(k): int(v) for k, v in days.items()},
        "cut_diameter_cm": round(float(flowers.d_cut_cm.iloc[0]), 2),
        "median_diameter_cm": round(flowers.d_cm.median(), 2),
        "length_m": round(flowers.u_cm.max() / 100, 2),
        "note": "aproximación: crecimiento supuesto de growth_per_day; color calibrado sin etiquetas",
    }
    (METRICS_DIR / f"summary_{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("B5:", json.dumps(summary, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(11, 3.5))
    bottom = np.zeros(len(per_m))
    for k, col in [("today", "#2CA02C"), ("tomorrow", "#F2B701"), ("later", "#4C72B0")]:
        ax.bar(per_m.index + bin_m / 2, per_m[k], bottom=bottom, width=bin_m * 0.9,
               label=STAGE_ES[k], color=col)
        bottom += per_m[k].values
    ax.set_xlabel("posición a lo largo de la cama (m)")
    ax.set_ylabel("flores")
    ax.set_title(f"Flores por metro y cuándo se cortan — {stem} ({color})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"b5_per_meter_{stem}.png", dpi=110)
    plt.close(fig)

    sc = P["results"]["mosaic_scale"]
    mosaic = build_mosaic(stem, meta, shifts, sc)
    for r in flowers.itertuples():
        cv2.circle(mosaic, (int(r.u_cm * meta["px_per_cm"] * sc), int(r.v_cm * meta["px_per_cm"] * sc)),
                   max(2, int(r.d_cm * meta["px_per_cm"] * sc / 2)), STAGE_BGR[r.stage], 1, cv2.LINE_AA)
    cv2.imwrite(str(FIGURES_DIR / f"b5_mosaic_full_{stem}.jpg"), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 85])
    seg = int(4 * 100 * meta["px_per_cm"] * sc)  # vista en filas de 4 m
    rows = [mosaic[:, j:j + seg] for j in range(0, mosaic.shape[1], seg)]
    rows = [np.pad(r, ((0, 6), (0, seg - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(FIGURES_DIR / f"b5_mosaic_{stem}.jpg"), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--video", help="nombre del video sin extensión (por defecto, todos)")
    ap.add_argument("--ignore-gate", action="store_true", help="correr aunque A7 no pase")
    args = ap.parse_args()
    P = load_params()
    for video in list_videos(NORMALIZED_VIDEO_DIR):
        stem = video.stem
        if args.video and stem != args.video:
            continue
        qc = json.loads((METRICS_DIR / f"qc_block_a_{stem}.json").read_text())
        if not qc["gate_open"]:
            msg = f"{stem}: A7 gate is closed ({qc['pass']})"
            if not args.ignore_gate:
                print(msg + " -> skipped (use --ignore-gate to explore)")
                continue
            print("WARNING:", msg)
        print(f"=== {stem}")
        for step in args.steps:
            globals()[f"step_{step}"](stem, P)
    if "b5" in args.steps:
        write_summary_all()


BED_LENGTH_M, BED_WIDTH_M = 32.0, 1.35   # medidas nominales de la cama


def write_summary_all():
    """B5: resumen y métricas del conteo de flores de todas las camas (sin los excluidos).

    Sin conteo manual no hay error absoluto; estas métricas miden la consistencia del conteo:
    acuerdo entre los dos métodos, densidad, homogeneidad a lo largo de la cama (CV por metro,
    sin los 2 m de cada extremo) y error del largo recorrido frente a los 32 m de la cama.
    """
    rows = []
    for path in sorted(METRICS_DIR.glob("summary_*.json")):
        s = json.loads(path.read_text())
        if s["video"] in excluded_videos():
            continue
        pm = pd.read_csv(TABLES_DIR / f"per_meter_{s['video']}.csv").total.iloc[2:-2]
        rows.append({"video": s["video"], "color": s["flower_color"], "total": s["total_flowers"],
                     "tracks>=10": s["total_flowers_other_method"], "diff%": s["methods_diff_pct"],
                     "hoy": s["today"], "hoy%": s["today_pct"], "mañana": s["tomorrow"],
                     "mañana%": s["tomorrow_pct"], "otros": s["later"], "otros%": s["later_pct"],
                     "d_med": s["median_diameter_cm"], "largo_m": s["length_m"],
                     "length_err%": round(100 * (s["length_m"] - BED_LENGTH_M) / BED_LENGTH_M, 1),
                     "flowers_per_m": round(s["total_flowers"] / BED_LENGTH_M),
                     "flowers_per_m2": round(s["total_flowers"] / (BED_LENGTH_M * BED_WIDTH_M)),
                     "per_meter_cv%": round(float(100 * pm.std() / pm.mean()), 1)})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES_DIR / "summary_all_videos.csv", index=False)
    by_color = t.groupby("color").total.agg(["count", "mean", "std"])
    metrics = {
        "beds": len(t), "total_flowers": int(t.total.sum()),
        "flowers_per_bed": {"mean": round(float(t.total.mean())), "min": int(t.total.min()),
                            "max": int(t.total.max())},
        "flowers_per_m2": {"mean": round(float(t.flowers_per_m2.mean())), "min": int(t.flowers_per_m2.min()),
                           "max": int(t.flowers_per_m2.max())},
        "methods_agreement": {"mean_abs_diff_pct": round(float(t["diff%"].abs().mean()), 1),
                              "max_abs_diff_pct": round(float(t["diff%"].abs().max()), 1),
                              "beds_within_10_pct": int((t["diff%"].abs() <= 10).sum())},
        "length": {"mean_abs_err_pct": round(float(t["length_err%"].abs().mean()), 1),
                   "range_m": [float(t.largo_m.min()), float(t.largo_m.max())]},
        "per_meter_cv_pct": {"median": round(float(t["per_meter_cv%"].median()), 1),
                             "max": float(t["per_meter_cv%"].max())},
        "between_beds_same_color_cv_pct": {c: round(float(100 * r["std"] / r["mean"]), 1)
                                           for c, r in by_color.iterrows() if r["count"] > 1},
        "share_pct": {k: round(float(100 * t[k].sum() / t.total.sum()), 1) for k in ["hoy", "mañana", "otros"]},
        "note": "sin conteo manual: miden consistencia, no error absoluto",
    }
    (METRICS_DIR / "flower_count_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"B5: summary of {len(rows)} beds -> summary_all_videos.csv, flower_count_metrics.json")


if __name__ == "__main__":
    main()
