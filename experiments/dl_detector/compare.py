"""Compara el detector YOLO con el clásico sobre la franja central de cada cuadro.

- Conteo total con la línea de conteo (misma regla para los dos).
- Acuerdo flor a flor en los cuadros de validación (pares, solo clásico, solo YOLO).
- Figura con las dos detecciones superpuestas para juzgar a ojo quién acierta.
Salida: results/metrics/dl_compare_<v>.json, results/tables/dl_per_meter_<v>.csv,
        results/figures/dl_compare_<v>.jpg
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from common import BOX_FACTOR, IMGSZ, RUNS_DIR, TILE, VAL_RANGE, center_strip, masked, match
from src.common.normalized import NormalizedVideo
from src.common.paths import FIGURES_DIR, METRICS_DIR, TABLES_DIR
from src.common.video import iter_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="63-40A")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights", default=str(RUNS_DIR / "flowers" / "weights" / "best.pt"))
    args = ap.parse_args()
    nv = NormalizedVideo(args.video)
    x0, x1, y0, y1 = center_strip(nv)
    scale = IMGSZ / TILE
    imgsz = (int(np.ceil((y1 - y0) * scale / 32) * 32), IMGSZ)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.weights)
    valid = set(nv.valid_frames)

    rows, keep_frames = [], {}
    show = set(np.linspace(VAL_RANGE[0], VAL_RANGE[1] - 1, 4).astype(int))
    for i, frame in tqdm(iter_frames(nv.video_path), total=len(nv.transforms), desc="YOLO"):
        if i not in valid:
            continue
        crop = masked(frame, nv.roi(i))[y0:y1, x0:x1]
        r = model.predict(crop, imgsz=imgsz, conf=args.conf, max_det=1500, device=device,
                          verbose=False)[0]
        b = r.boxes.xywh.cpu().numpy()
        for (bx, by, bw, bh), c in zip(b, r.boxes.conf.cpu().numpy()):
            rows.append({"frame": i, "x": bx + x0, "y": by + y0,
                         "d_px": (bw + bh) / 2 / BOX_FACTOR, "conf": c})
        if i in show:
            keep_frames[i] = crop
    yolo = pd.DataFrame(rows)
    yolo.round(3).to_csv(TABLES_DIR / f"dl_detections_{args.video}.csv", index=False)

    classic = pd.read_csv(TABLES_DIR / f"detections_{args.video}.csv",
                          usecols=["frame", "x", "y", "d_px", "u_cm"])
    classic = classic[(classic.x >= x0) & (classic.x < x1)]
    shifts = pd.read_csv(TABLES_DIR / f"shifts_{args.video}.csv").set_index("frame")

    def count_line(d):
        """Línea de conteo: detecciones en la franja central de ancho |dx| de su cuadro."""
        half = d.frame.map(shifts.dx.abs() / 2)
        return d[(d.x - nv.out_w / 2).abs() < half]

    yl, cl = count_line(yolo), count_line(classic)
    # posición global para repartir por metro (misma convención que block_b)
    a = yl.x - yl.frame.map(shifts.offset_x)
    u = (shifts.a_max_px.iloc[0] - a) if shifts.walk_left.iloc[0] else (a - shifts.a_min_px.iloc[0])
    yl = yl.assign(u_cm=u / nv.px_per_cm)
    per_m = pd.DataFrame({"classic": cl.groupby((cl.u_cm // 100).astype(int)).size(),
                          "yolo": yl.groupby((yl.u_cm // 100).astype(int)).size()}).fillna(0).astype(int)
    per_m.index.name = "meter"
    per_m.to_csv(TABLES_DIR / f"dl_per_meter_{args.video}.csv")

    pairs = only_c = only_y = 0
    for i in range(*VAL_RANGE):
        if i not in valid:
            continue
        ca = classic.loc[classic.frame == i, ["x", "y"]].values
        ya = yolo.loc[yolo.frame == i, ["x", "y"]].values
        n, oc, oy = match(ca, ya, radius=8)
        pairs, only_c, only_y = pairs + n, only_c + oc, only_y + oy
    full = per_m.iloc[1:-1]  # sin los extremos de la cama
    res = {
        "video": args.video, "conf": args.conf, "labels": "pre-etiquetas del clásico (sin corregir)",
        "line_count_classic": int(len(cl)), "line_count_yolo": int(len(yl)),
        "yolo_vs_classic_pct": round(100 * (len(yl) / len(cl) - 1), 1),
        "per_meter_corr": round(float(np.corrcoef(full.classic, full.yolo)[0, 1]), 3),
        "per_meter_mean_abs_diff_pct": round(float((100 * (full.yolo - full.classic).abs()
                                                    / full.classic).mean()), 1),
        "val_matched": pairs, "val_only_classic": only_c, "val_only_yolo": only_y,
        "val_agreement_f1": round(2 * pairs / max(2 * pairs + only_c + only_y, 1), 3),
        "median_d_cm_classic": round(float(cl.d_px.median() / nv.px_per_cm), 2),
        "median_d_cm_yolo": round(float(yl.d_px.median() / nv.px_per_cm), 2),
    }
    (METRICS_DIR / f"dl_compare_{args.video}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))

    tiles = []
    for i, crop in sorted(keep_frames.items()):  # magenta = clásico, cian = YOLO
        img = cv2.resize(crop[:480], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for r in classic[classic.frame == i].itertuples():
            if r.y - y0 < 480:
                cv2.circle(img, (int((r.x - x0) * 2), int((r.y - y0) * 2)), int(r.d_px), (255, 0, 255), 1,
                           cv2.LINE_AA)
        for r in yolo[yolo.frame == i].itertuples():
            if r.y - y0 < 480:
                h = int(r.d_px * BOX_FACTOR)
                p = (int((r.x - x0) * 2), int((r.y - y0) * 2))
                cv2.rectangle(img, (p[0] - h, p[1] - h), (p[0] + h, p[1] + h), (255, 255, 0), 1)
        cv2.putText(img, f"#{i}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(img)
    cv2.imwrite(str(FIGURES_DIR / f"dl_compare_{args.video}.jpg"), np.hstack(tiles),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
