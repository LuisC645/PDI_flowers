"""Inventario de videos y muestra visual de cuadros.

Entrada: data/video/*
Salida:  results/tables/inventory.csv
         results/figures/sample_<video>.jpg
Uso:     python scripts/f1_inventory.py   (desde proy/)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import FIGURES_DIR, TABLES_DIR, list_videos, load_params  # noqa: E402
from src.common.video import get_video_info, make_mosaic, read_frames  # noqa: E402

FLOWER_COLOR = {"IMG_1267": "orange"}  # se llena a mano al revisar la muestra


def main():
    params = load_params()
    n_sample = params["sample"]["n_frames"]
    rows = []
    for video_path in list_videos():
        info = get_video_info(video_path)
        info["flower_color"] = FLOWER_COLOR.get(video_path.stem, "")
        rows.append(info)
        print(info)

        sample_idx = np.linspace(0, info["n_frames"] - 1, n_sample).astype(int)
        frames = read_frames(video_path, sample_idx)
        labels = [f"#{i}  {i / info['fps']:.1f}s" for i in sample_idx]
        out_path = FIGURES_DIR / f"sample_{video_path.stem}.jpg"
        cv2.imwrite(str(out_path), make_mosaic(frames, labels), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("Sample saved:", out_path.name)

    inventory = pd.DataFrame(rows)
    inventory.to_csv(TABLES_DIR / "inventory.csv", index=False)
    print(inventory.to_string(index=False))


if __name__ == "__main__":
    main()
