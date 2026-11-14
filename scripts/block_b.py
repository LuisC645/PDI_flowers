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
from src.analysis.counting import (aggregate_flowers, classify_cut, frame_shift,  # noqa: E402
                                   global_offsets, link_tracks, strip_flowers)
from src.analysis.detection import detect_flowers  # noqa: E402
from src.common.normalized import NormalizedVideo  # noqa: E402
from src.common.paths import (FIGURES_DIR, METRICS_DIR, NORMALIZED_VIDEO_DIR, TABLES_DIR,  # noqa: E402
                              TRANSFORMS_DIR, VIDEOS_METADATA, load_params)
from src.common.video import iter_frames, read_frames  # noqa: E402

STEPS = ["b12", "b3", "b4", "b5"]
CUT_BGR, NO_CUT_BGR = (0, 255, 0), (255, 0, 0)  # verde = cortar, azul = no cortar


def context(stem, P):
    """Datos comunes: salida del Bloque A, perfil de color y estado de cada cuadro."""
    nv = NormalizedVideo(stem)
    color = pd.read_csv(VIDEOS_METADATA, dtype=str).set_index("video_id").loc[stem, "flower_color"]
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


def step_b3(stem, P):
    """B3: desplazamiento entre cuadros, coordenada global y conteo único."""
    meta, _, _, status, _ = context(stem, P)
    cfg, sc = P["counting"], P["counting"]["phase_scale"]
    m = meta["margin_px"]
    rows, prev = [], None
    window = None
    for i, fr in tqdm(iter_frames(NORMALIZED_VIDEO_DIR / f"{stem}.mp4"), total=len(status), desc="B3"):
        g = cv2.cvtColor(cv2.resize(fr[m:meta["out_h"] - m], None, fx=sc, fy=sc,
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if window is None:
            window = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        ok = prev is not None and status.status[i] != "invalid" and status.status[i - 1] != "invalid"
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
    flowers = aggregate_flowers(det, cfg["min_obs"])
    strip = strip_flowers(det, shifts, meta["out_w"])
    det.round(3).to_csv(TABLES_DIR / f"detections_{stem}.csv", index=False)
    chosen = flowers if cfg["method"] == "track" else strip
    chosen.round(3).to_csv(TABLES_DIR / f"flowers_{stem}.csv", index=False)

    length_m = det.u_cm.max() / 100
    print(f"B3: travelled length {length_m:.1f} m (bed = 32 m), "
          f"unreliable shifts {(~shifts.reliable).mean():.1%}, mean |dy| {shifts.dy.abs().mean():.1f} px")
    print(f"B3: unique flowers  track = {len(flowers)}   strip = {len(strip)}   "
          f"(method used: {cfg['method']})")


def step_b4(stem, P):
    """B4: etiqueta cortar / no cortar."""
    _, profile, _, _, _ = context(stem, P)
    path = TABLES_DIR / f"flowers_{stem}.csv"
    flowers = classify_cut(pd.read_csv(path), profile)
    flowers.round(3).to_csv(path, index=False)
    print(f"B4: cut {flowers.cut.sum()} / {len(flowers)}  (size ok {flowers.size_ok.mean():.0%}, "
          f"color ok {flowers.color_ok.mean():.0%})")


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


def step_b5(stem, P):
    """B5: conteo total, % a cortar, distribución por metro, histograma y mosaico marcado."""
    meta, profile, color, _, _ = context(stem, P)
    flowers = pd.read_csv(TABLES_DIR / f"flowers_{stem}.csv")
    shifts = pd.read_csv(TABLES_DIR / f"shifts_{stem}.csv")
    bin_m = P["results"]["bin_m"]
    flowers["meter"] = (flowers.u_cm / 100 // bin_m * bin_m).astype(int)
    per_m = flowers.groupby("meter").agg(total=("cut", "size"), cut=("cut", "sum")).reset_index()
    per_m["cut_pct"] = (100 * per_m.cut / per_m.total).round(1)
    per_m.to_csv(TABLES_DIR / f"per_meter_{stem}.csv", index=False)
    summary = {
        "video": stem, "flower_color": color, "method": P["counting"]["method"],
        "total_flowers": int(len(flowers)), "cut": int(flowers.cut.sum()),
        "cut_pct": round(100 * flowers.cut.mean(), 1),
        "length_m": round(flowers.u_cm.max() / 100, 2),
        "median_diameter_cm": round(flowers.d_cm.median(), 2),
        "draft_warning": "BORRADOR: umbrales sin calibrar, moneda y paralaje sin medir (F1.3, F1.4)",
    }
    (METRICS_DIR / f"summary_{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("B5:", json.dumps(summary, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(per_m.meter + bin_m / 2, per_m.total - per_m.cut, width=bin_m * 0.9, label="no cortar",
           color="#4C72B0")
    ax.bar(per_m.meter + bin_m / 2, per_m.cut, bottom=per_m.total - per_m.cut, width=bin_m * 0.9,
           label="cortar", color="#55A868")
    ax.set_xlabel("posición a lo largo de la cama (m)")
    ax.set_ylabel("flores")
    ax.set_title(f"B5 flores por metro — {stem} (borrador)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"b5_per_meter_{stem}.png", dpi=110)
    plt.close(fig)

    sc = P["results"]["mosaic_scale"]
    mosaic = build_mosaic(stem, meta, shifts, sc)
    for r in flowers.itertuples():
        cv2.circle(mosaic, (int(r.u_cm * meta["px_per_cm"] * sc), int(r.v_cm * meta["px_per_cm"] * sc)),
                   max(2, int(r.d_cm * meta["px_per_cm"] * sc / 2)), CUT_BGR if r.cut else NO_CUT_BGR, 1,
                   cv2.LINE_AA)
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
    for video in sorted(NORMALIZED_VIDEO_DIR.glob("*.mp4")):
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


if __name__ == "__main__":
    main()
