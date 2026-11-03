"""Inventario de videos y muestra visual de cuadros.

Entrada: data/videos_processed/*
         data/videos.csv  (metadatos llenados a mano: fecha, variedad, cama, color)
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
from src.common.paths import (FIGURES_DIR, TABLES_DIR, VIDEOS_METADATA, list_videos,  # noqa: E402
                              load_params)
from src.common.video import get_video_info, make_mosaic, read_frames  # noqa: E402


def main():
    params = load_params()
    n_sample = params["sample"]["n_frames"]
    metadata = pd.read_csv(VIDEOS_METADATA, dtype=str).fillna("").set_index("video_id")
    rows = []
    for video_path in list_videos():
        info = get_video_info(video_path)
        if video_path.stem not in metadata.index:
            print(f"WARNING: {video_path.stem} is not in {VIDEOS_METADATA.name}")
        else:
            info.update(metadata.loc[video_path.stem].to_dict())
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
