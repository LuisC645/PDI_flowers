"""Estandarización de videos: menos fps, sin audio y comprimidos.

Divide cada video en ventanas de round(fps / target_fps) cuadros y conserva uno por
ventana: el más nítido si select_sharpest = true, si no el primero. Escribe el video
estandarizado (H.264, sin audio) y, si delete_raw = true, borra el original al terminar.
Los videos que ya están en data/videos_processed se omiten.

Entrada: data/videos_raw/<video>
Salida:  data/videos_processed/<video>.mp4
         results/tables/frame_index_<video>.csv     (cuadros conservados y su origen)
         results/tables/sharpness_<video>.csv       (nitidez de todos los cuadros originales)
         results/tables/processing_log.csv          (datos del original y del estandarizado)
Uso:     python scripts/standardize_videos.py   (desde proy/)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import (PROCESSED_VIDEO_DIR, RAW_VIDEO_DIR, TABLES_DIR,  # noqa: E402
                              list_videos, load_params)
from src.common.video import get_video_info, iter_frames  # noqa: E402
from src.normalization.sharpness import laplacian_variance  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
LOG_PATH = TABLES_DIR / "processing_log.csv"


def open_writer(out_path, width, height, fps, crf):
    """Abre ffmpeg para recibir cuadros BGR por stdin y codificarlos en H.264 sin audio."""
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
           "-i", "-", "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def standardize_video(video_path, out_path, target_fps, crf, select_sharpest, downscale):
    """Escribe el video estandarizado y devuelve (índice de conservados, nitidez de todos)."""
    info = get_video_info(video_path)
    fps = info["fps"]
    step = max(1, round(fps / target_fps))
    writer = open_writer(out_path, info["width"], info["height"], target_fps, crf)

    sharpness_rows, index_rows = [], []
    best = None  # (nitidez, índice, cuadro) del mejor cuadro de la ventana actual

    def keep(score, src_idx, frame):
        writer.stdin.write(frame.tobytes())
        index_rows.append({"new_frame": len(index_rows), "source_frame": src_idx,
                           "time_s": round(src_idx / fps, 3), "sharpness": round(score, 1)})

    for src_idx, frame in tqdm(iter_frames(video_path), total=info["n_frames"], desc=video_path.name):
        score = laplacian_variance(frame, downscale)
        sharpness_rows.append({"source_frame": src_idx, "time_s": round(src_idx / fps, 3),
                               "sharpness": round(score, 1)})
        is_first = src_idx % step == 0
        if is_first or (select_sharpest and score > best[0]):
            best = (score, src_idx, frame)
        if src_idx % step == step - 1:  # cierre de ventana
            keep(*best)
            best = None
    if best is not None:  # última ventana incompleta
        keep(*best)

    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {video_path.name}")
    return pd.DataFrame(index_rows), pd.DataFrame(sharpness_rows)


def main():
    params = load_params()
    cfg = params["video"]
    log_rows = []
    raw_videos = list_videos(RAW_VIDEO_DIR)
    if not raw_videos:
        print(f"No videos in {RAW_VIDEO_DIR.relative_to(RAW_VIDEO_DIR.parents[1])}, nothing to do")
    for raw_path in raw_videos:
        out_path = PROCESSED_VIDEO_DIR / f"{raw_path.stem}.mp4"
        if out_path.exists():
            print(f"{raw_path.name}: already processed, skipped")
            continue

        raw_info = get_video_info(raw_path)
        tmp_path = out_path.with_name(f"{raw_path.stem}.tmp.mp4")
        index, sharpness = standardize_video(raw_path, tmp_path, cfg["target_fps"], cfg["crf"],
                                             cfg["select_sharpest"], params["sharpness"]["downscale"])

        new_info = get_video_info(tmp_path)
        if new_info["n_frames"] != len(index):  # verificación antes de borrar nada
            tmp_path.unlink()
            raise RuntimeError(f"{raw_path.name}: wrote {new_info['n_frames']} frames, "
                               f"expected {len(index)}")
        tmp_path.rename(out_path)

        index.to_csv(TABLES_DIR / f"frame_index_{raw_path.stem}.csv", index=False)
        sharpness.to_csv(TABLES_DIR / f"sharpness_{raw_path.stem}.csv", index=False)
        log_rows.append({"video": out_path.name,
                         "raw_fps": raw_info["fps"], "raw_frames": raw_info["n_frames"],
                         "raw_size_mb": raw_info["size_mb"],
                         "fps": new_info["fps"], "frames": new_info["n_frames"],
                         "size_mb": new_info["size_mb"],
                         "width": new_info["width"], "height": new_info["height"],
                         "duration_s": raw_info["duration_s"], "crf": cfg["crf"]})
        if cfg["delete_raw"]:
            raw_path.unlink()
        print(f"{raw_path.name}: {raw_info['n_frames']} -> {new_info['n_frames']} frames, "
              f"{raw_info['size_mb']:.0f} MB -> {new_info['size_mb']:.0f} MB"
              f"{' (raw deleted)' if cfg['delete_raw'] else ''}")

    if log_rows:
        log = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            old = pd.read_csv(LOG_PATH)
            log = pd.concat([old[~old.video.isin(log.video)], log], ignore_index=True)
        log.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    main()
