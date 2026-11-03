"""Extracción de cuadros a menor fps.

Divide el video en ventanas de round(fps / target_fps) cuadros y guarda uno por
ventana: el más nítido si select_sharpest = true, si no el primero.
Se mide la nitidez de TODOS los cuadros; ninguno se descarta del registro.

Entrada: data/video/<video>
Salida:  data/frames/<video>/<video>_<source_frame>.jpg
         data/frames/<video>/index.csv               (cuadros guardados)
         results/tables/sharpness_<video>.csv        (nitidez de todos los cuadros)
Uso:     python scripts/f1_extract_frames.py   (desde proy/)
"""
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.normalization.sharpness import laplacian_variance  # noqa: E402
from src.common.paths import FRAMES_DIR, TABLES_DIR, list_videos, load_params  # noqa: E402
from src.common.video import get_video_info, iter_frames  # noqa: E402


def extract_frames(video_path, target_fps, jpeg_quality, select_sharpest, downscale):
    """Guarda un cuadro por ventana y devuelve (índice de guardados, nitidez de todos)."""
    info = get_video_info(video_path)
    fps = info["fps"]
    step = max(1, round(fps / target_fps))
    out_dir = FRAMES_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    sharpness_rows, index_rows = [], []
    best = None  # (nitidez, índice, cuadro) del mejor cuadro de la ventana actual

    def save(window_id, score, src_idx, frame):
        file_name = f"{video_path.stem}_{src_idx:05d}.jpg"
        cv2.imwrite(str(out_dir / file_name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        index_rows.append({"sample_idx": window_id, "source_frame": src_idx,
                           "time_s": round(src_idx / fps, 3), "sharpness": round(score, 1),
                           "file": file_name})

    for src_idx, frame in tqdm(iter_frames(video_path), total=info["n_frames"], desc=video_path.name):
        score = laplacian_variance(frame, downscale)
        sharpness_rows.append({"source_frame": src_idx, "time_s": round(src_idx / fps, 3),
                               "sharpness": round(score, 1)})
        is_first = src_idx % step == 0
        if is_first or (select_sharpest and score > best[0]):
            best = (score, src_idx, frame)
        if src_idx % step == step - 1:  # cierre de ventana
            save(src_idx // step, *best)
            best = None
    if best is not None:  # última ventana incompleta
        save(len(index_rows), *best)

    pd.DataFrame(index_rows).to_csv(out_dir / "index.csv", index=False)
    pd.DataFrame(sharpness_rows).to_csv(TABLES_DIR / f"sharpness_{video_path.stem}.csv", index=False)
    return pd.DataFrame(index_rows), step


def main():
    params = load_params()
    cfg = params["frames"]
    for video_path in list_videos():
        index, step = extract_frames(video_path, cfg["target_fps"], cfg["jpeg_quality"],
                                     cfg["select_sharpest"], params["sharpness"]["downscale"])
        out_dir = FRAMES_DIR / video_path.stem
        size_mb = sum(p.stat().st_size for p in out_dir.glob("*.jpg")) / 1e6
        print(f"{video_path.name}: 1 of every {step} frames -> {len(index)} saved, {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
