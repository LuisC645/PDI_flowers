"""Validación del Bloque A contra las máscaras hechas a mano (F1.4).

Compara, en cada cuadro etiquetado, la máscara a mano con:
  mask: la máscara de A2 (por cuadro, sin estabilizar).
  band: la franja que realmente usa el Bloque B (bordes estabilizados de A4/A5, llevada
        al cuadro original con la matriz inversa).
Métricas (fijadas antes de medir): IoU, Dice y contaminación = fracción de la región
automática que cae fuera de la cama a mano (camino, cama vecina, camarógrafo).
Entrada: data/ground_truth/masks/<v>_<cuadro>.png   (blanco = cama, mismo tamaño del cuadro)
Salida:  results/tables/mask_eval_<v>.csv, results/metrics/mask_eval_<v>.json
Uso:     python scripts/evaluate_block_a.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.normalized import NormalizedVideo  # noqa: E402
from src.common.paths import GROUND_TRUTH_DIR, MASKS_DIR, METRICS_DIR, TABLES_DIR  # noqa: E402


def scores(auto, gt):
    """IoU, Dice y contaminación entre dos máscaras booleanas."""
    inter, union = (auto & gt).sum(), (auto | gt).sum()
    return {"iou": inter / max(union, 1), "dice": 2 * inter / max(auto.sum() + gt.sum(), 1),
            "contamination": (auto & ~gt).sum() / max(auto.sum(), 1)}


def main():
    files = sorted((GROUND_TRUTH_DIR / "masks").glob("*.png"))
    if not files:
        print("No hand-drawn masks in data/ground_truth/masks; run scripts/export_gt_frames.py "
              "and label the frames first")
        return
    rows = []
    cache = {}
    for path in files:
        stem, frame = path.stem.rsplit("_", 1)
        frame = int(frame)
        if stem not in cache:
            z = np.load(MASKS_DIR / f"{stem}_bed_masks.npz")
            cache[stem] = (NormalizedVideo(stem), z["masks"], int(z["width"]))
        nv, packed, width = cache[stem]
        gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
        h, w = gt.shape
        mask = np.unpackbits(packed[frame], axis=1)[:, :width]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        row = {"video": stem, "frame": frame, "status": nv.transforms.status[frame],
               **{f"mask_{k}": v for k, v in scores(mask, gt).items()}}
        M = nv.matrix(frame)
        if not np.isnan(M).any():
            band = cv2.warpAffine(nv.band, cv2.invertAffineTransform(M), (w, h),
                                  flags=cv2.INTER_NEAREST) > 0
            row.update({f"band_{k}": v for k, v in scores(band, gt).items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    for stem, t in table.groupby("video"):
        t.round(4).to_csv(TABLES_DIR / f"mask_eval_{stem}.csv", index=False)
        summary = {"n_frames": int(len(t)), **t.drop(columns=["video", "frame", "status"])
                   .mean().round(4).to_dict()}
        (METRICS_DIR / f"mask_eval_{stem}.json").write_text(json.dumps(summary, indent=2))
        print(stem, json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
