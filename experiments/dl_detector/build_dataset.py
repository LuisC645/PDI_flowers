"""Arma el conjunto de recortes con pre-etiquetas del detector clásico (formato YOLO).

Entrada: video normalizado + results/tables/detections_<v>.csv (correr block_b antes)
Salida:  data/dl/images/{train,val}/*.jpg, data/dl/labels/{train,val}/*.txt, data/dl/data.yaml
         data/dl/review/  (copia de validación para corregir a mano)
"""
import argparse
import shutil

import cv2
import numpy as np
import pandas as pd

from common import BOX_FACTOR, BUFFER, DL_DIR, ROOT, TILE, VAL_RANGE, masked
from src.common.normalized import NormalizedVideo
from src.common.paths import TABLES_DIR
from src.common.video import iter_frames


def tile_origins(nv):
    xs = np.linspace(0, nv.out_w - TILE, 3).astype(int)
    ys = np.linspace(nv.margin, nv.out_h - nv.margin - TILE, 4).astype(int)
    return [(x, y) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="63-40A")
    ap.add_argument("--n-train", type=int, default=60, help="cuadros de entrenamiento")
    ap.add_argument("--n-val", type=int, default=12, help="cuadros de validación")
    args = ap.parse_args()
    nv = NormalizedVideo(args.video)
    det = pd.read_csv(TABLES_DIR / f"detections_{args.video}.csv", usecols=["frame", "x", "y", "d_px"])
    valid = np.array(nv.valid_frames)
    in_val = (valid >= VAL_RANGE[0]) & (valid < VAL_RANGE[1])
    in_train = (valid < VAL_RANGE[0] - BUFFER) | (valid >= VAL_RANGE[1] + BUFFER)
    pick = {"train": valid[in_train][np.linspace(0, in_train.sum() - 1, args.n_train).astype(int)],
            "val": valid[in_val][np.linspace(0, in_val.sum() - 1, args.n_val).astype(int)]}
    split_of = {int(f): s for s, fs in pick.items() for f in fs}

    for sub in ("images", "labels", "review"):
        shutil.rmtree(DL_DIR / sub, ignore_errors=True)
    for s in pick:
        (DL_DIR / "images" / s).mkdir(parents=True)
        (DL_DIR / "labels" / s).mkdir(parents=True)
    (DL_DIR / "review").mkdir(parents=True)

    by_frame = dict(tuple(det.groupby("frame")))
    counts = {"train": [0, 0], "val": [0, 0]}
    for i, frame in iter_frames(nv.video_path):
        if i not in split_of:
            continue
        s = split_of[i]
        roi = nv.roi(i)
        img = masked(frame, roi)
        d = by_frame.get(i, det.iloc[:0])
        for x0, y0 in tile_origins(nv):
            if (roi[y0:y0 + TILE, x0:x0 + TILE] > 0).mean() < 0.6:
                continue  # recorte casi sin región útil
            t = d[(d.x >= x0) & (d.x < x0 + TILE) & (d.y >= y0) & (d.y < y0 + TILE)]
            side = np.clip(BOX_FACTOR * t.d_px / TILE, 0, 1)
            lab = np.c_[np.zeros(len(t)), (t.x - x0) / TILE, (t.y - y0) / TILE, side, side]
            name = f"{args.video}_{i:04d}_{x0:04d}_{y0:04d}"
            cv2.imwrite(str(DL_DIR / "images" / s / f"{name}.jpg"), img[y0:y0 + TILE, x0:x0 + TILE],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.savetxt(DL_DIR / "labels" / s / f"{name}.txt", lab, fmt="%d %.5f %.5f %.5f %.5f")
            counts[s][0] += 1
            counts[s][1] += len(t)
    (DL_DIR / "data.yaml").write_text(
        f"path: {DL_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: flower\n")
    for kind in ("images", "labels"):  # copia para corregir a mano sin tocar las pre-etiquetas
        shutil.copytree(DL_DIR / kind / "val", DL_DIR / "review" / kind)
    (DL_DIR / "review" / "classes.txt").write_text("flower\n")
    for s, (n_tiles, n_boxes) in counts.items():
        print(f"{s}: {len(pick[s])} frames, {n_tiles} tiles, {n_boxes} boxes "
              f"({n_boxes / max(n_tiles, 1):.0f} per tile)")
    print("dataset:", DL_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
