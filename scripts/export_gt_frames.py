"""F1.4: exporta los cuadros que se van a etiquetar a mano (máscara de la cama).

Elige n cuadros repartidos por el video y agrega algunos difíciles (inválidos o
interpolados en A4), para que la referencia no salga solo de los cuadros fáciles.
Salida: data/ground_truth/frames/<v>_<cuadro>.png   (cuadro del video estandarizado)
        data/ground_truth/frames_<v>.csv            (lista de cuadros y por qué se eligieron)
Qué hacer después: pintar la cama en blanco (255) y el resto en negro (0), en una imagen
del mismo tamaño, y guardarla en data/ground_truth/masks/<v>_<cuadro>.png (por ejemplo con
LabelMe, CVAT o GIMP). Luego: python scripts/evaluate_block_a.py
Uso:  python scripts/export_gt_frames.py [--n 20] [--video 63_40A]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import GROUND_TRUTH_DIR, TRANSFORMS_DIR, list_videos  # noqa: E402
from src.common.video import get_video_info, read_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="cuadros por video")
    ap.add_argument("--video")
    args = ap.parse_args()
    out_dir = GROUND_TRUTH_DIR / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for video in list_videos():
        if args.video and video.stem != args.video:
            continue
        n_frames = get_video_info(video)["n_frames"]
        n_hard = args.n // 4
        pick = {int(i): "spread" for i in np.linspace(0, n_frames - 1, args.n - n_hard)}
        geom_path = TRANSFORMS_DIR / f"{video.stem}_geometry.csv"
        if geom_path.exists():  # cuadros difíciles, repartidos
            hard = pd.read_csv(geom_path).query("status != 'measured'")
            for r in hard.iloc[np.linspace(0, len(hard) - 1, n_hard).astype(int)].itertuples():
                pick.setdefault(int(r.frame), f"hard_{r.status}")
        idx = sorted(pick)
        for i, frame in zip(idx, read_frames(video, idx)):
            cv2.imwrite(str(out_dir / f"{video.stem}_{i:04d}.png"), frame)
        pd.DataFrame({"video": video.stem, "frame": idx, "reason": [pick[i] for i in idx]}).to_csv(
            GROUND_TRUTH_DIR / f"frames_{video.stem}.csv", index=False)
        print(f"{video.stem}: {len(idx)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
