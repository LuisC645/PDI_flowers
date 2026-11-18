"""Estandarización de videos: orientación, SDR, menos fps, sin audio y comprimidos.

Por cada video de data/videos_raw:
  1. Orientación: el camarógrafo debe quedar abajo (la cama cruza el cuadro en horizontal).
     Se aplica la etiqueta de rotación del archivo y, encima, un giro extra: el de la
     columna `rotate` de data/videos.csv si está, o el detectado automáticamente (borde
     donde los píxeles casi no cambian en el tiempo = camarógrafo).
  2. Color: los videos HDR (HLG/PQ de los iPhone) se pasan a SDR BT.709 con mapeo de tonos.
  3. fps: ventanas de round(fps / target_fps) cuadros; se conserva uno por ventana (el más
     nítido si select_sharpest = true).
Los videos que ya están en data/videos_processed se omiten.

Entrada: data/videos_raw/<video>, data/videos.csv (columna opcional `rotate`)
Salida:  data/videos_processed/<video>.mp4
         results/tables/frame_index_<video>.csv     (cuadros conservados y su origen)
         results/tables/sharpness_<video>.csv       (nitidez de todos los cuadros originales)
         results/tables/processing_log.csv          (original, estandarizado, giro y HDR)
Uso:     python scripts/standardize_videos.py   (desde proy/)
"""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import (PROCESSED_VIDEO_DIR, RAW_VIDEO_DIR, TABLES_DIR,  # noqa: E402
                              VIDEOS_METADATA, list_videos, load_params)
from src.common.video import (close_writer, get_video_info, open_writer, probe_stream,  # noqa: E402
                              read_ffmpeg)
from src.normalization.operator import detect_orientation  # noqa: E402
from src.normalization.sharpness import laplacian_variance  # noqa: E402

LOG_PATH = TABLES_DIR / "processing_log.csv"


def choose_rotation(raw_path, metadata, min_confidence):
    """Giro extra: el de videos.csv si está; si no, el detectado (None si es dudoso)."""
    manual = metadata.get(raw_path.stem, "")
    if str(manual).strip() not in ("", "nan"):
        return int(float(manual)), "manual", None
    side, rotation, conf = detect_orientation(raw_path)
    print(f"{raw_path.name}: operator detected on the {side} -> rotate {rotation} "
          f"(confidence {conf:.2f})")
    if conf < min_confidence:
        return None, "auto", conf
    return rotation, "auto", conf


def standardize_video(video_path, out_path, rotation, cfg, downscale):
    """Escribe el video estandarizado y devuelve (índice de conservados, nitidez de todos)."""
    info = probe_stream(video_path)
    step = max(1, round(info["fps"] / cfg["target_fps"]))
    w, h = (info["height"], info["width"]) if rotation % 180 else (info["width"], info["height"])
    writer = open_writer(out_path, w, h, cfg["target_fps"], cfg["crf"])

    sharpness_rows, index_rows = [], []
    best = None  # (nitidez, índice, tiempo, cuadro) del mejor cuadro de la ventana actual

    def keep(score, src_idx, time_s, frame):
        writer.stdin.write(frame.tobytes())
        index_rows.append({"new_frame": len(index_rows), "source_frame": src_idx,
                           "time_s": round(time_s, 3), "sharpness": round(score, 1)})

    n = get_video_info(video_path)["n_frames"]
    for src_idx, time_s, frame in tqdm(read_ffmpeg(video_path, rotation), total=n, desc=video_path.name):
        score = laplacian_variance(frame, downscale)
        sharpness_rows.append({"source_frame": src_idx, "time_s": round(time_s, 3),
                               "sharpness": round(score, 1)})
        if src_idx % step == 0 or (cfg["select_sharpest"] and score > best[0]):
            best = (score, src_idx, time_s, frame.copy())
        if src_idx % step == step - 1:  # cierre de ventana
            keep(*best)
            best = None
    if best is not None:  # última ventana incompleta
        keep(*best)

    close_writer(writer, video_path.name)
    return pd.DataFrame(index_rows), pd.DataFrame(sharpness_rows), info, step


def main():
    params = load_params()
    cfg = params["video"]
    meta = pd.read_csv(VIDEOS_METADATA, dtype=str)
    rot_meta = dict(zip(meta.video_id, meta["rotate"])) if "rotate" in meta else {}
    log_rows, pending = [], []
    raw_videos = list_videos(RAW_VIDEO_DIR)
    if not raw_videos:
        print(f"No videos in {RAW_VIDEO_DIR.relative_to(RAW_VIDEO_DIR.parents[1])}, nothing to do")
    for raw_path in raw_videos:
        out_path = PROCESSED_VIDEO_DIR / f"{raw_path.stem}.mp4"
        if out_path.exists():
            print(f"{raw_path.name}: already processed, skipped")
            continue
        if raw_path.stem not in set(meta.video_id):
            print(f"WARNING: {raw_path.stem} is not in {VIDEOS_METADATA.name} (needed for block B)")

        rotation, source, conf = choose_rotation(raw_path, rot_meta, cfg["orientation_min_confidence"])
        if rotation is None:
            print(f"{raw_path.name}: orientation is uncertain (confidence {conf:.2f}); set the "
                  f"`rotate` column in {VIDEOS_METADATA.name} (0/90/180/270) -> skipped")
            pending.append(raw_path.stem)
            continue

        raw_info = get_video_info(raw_path)
        tmp_path = out_path.with_name(f"{raw_path.stem}.tmp.mp4")
        index, sharpness, info, step = standardize_video(raw_path, tmp_path, rotation, cfg,
                                                         params["sharpness"]["downscale"])
        new_info = get_video_info(tmp_path)
        if new_info["n_frames"] != len(index):  # verificación antes de borrar nada
            tmp_path.unlink()
            raise RuntimeError(f"{raw_path.name}: wrote {new_info['n_frames']} frames, "
                               f"expected {len(index)}")
        tmp_path.rename(out_path)

        index.to_csv(TABLES_DIR / f"frame_index_{raw_path.stem}.csv", index=False)
        sharpness.to_csv(TABLES_DIR / f"sharpness_{raw_path.stem}.csv", index=False)
        log_rows.append({"video": out_path.name,
                         "raw_fps": round(info["fps"], 3), "raw_frames": len(sharpness),
                         "raw_size_mb": raw_info["size_mb"],
                         "fps": new_info["fps"], "frames": new_info["n_frames"],
                         "size_mb": new_info["size_mb"],
                         "width": new_info["width"], "height": new_info["height"],
                         "duration_s": round(float(sharpness.time_s.iloc[-1]), 2), "crf": cfg["crf"],
                         "window": step, "hdr_tonemapped": info["hdr"],
                         "rotation_tag": info["rotation_tag"], "extra_rotation": rotation,
                         "rotation_source": source,
                         "rotation_confidence": None if conf is None else round(conf, 2)})
        if cfg["delete_raw"]:
            raw_path.unlink()
        print(f"{raw_path.name}: {len(sharpness)} -> {new_info['n_frames']} frames, "
              f"{new_info['width']}x{new_info['height']}, rotate {rotation}, "
              f"{'HDR->SDR' if info['hdr'] else 'SDR'}, "
              f"{raw_info['size_mb']:.0f} MB -> {new_info['size_mb']:.0f} MB")

    if log_rows:
        log = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            old = pd.read_csv(LOG_PATH)
            log = pd.concat([old[~old.video.isin(log.video)], log], ignore_index=True)
        log.to_csv(LOG_PATH, index=False)
    if pending:
        print("Pending (orientation to set by hand):", ", ".join(pending))


if __name__ == "__main__":
    main()
