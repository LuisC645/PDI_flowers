"""Lectura y escritura básica de video (OpenCV para leer, ffmpeg para escribir)."""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")


def get_video_info(video_path):
    """Devuelve las propiedades principales de un video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    info = {
        "video": Path(video_path).name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 3),
        "n_frames": n_frames,
        "duration_s": round(n_frames / fps, 2),
        "size_mb": round(Path(video_path).stat().st_size / 1e6, 1),
    }
    cap.release()
    return info


def read_frames(video_path, frame_indices):
    """Lee los cuadros pedidos y devuelve una lista de imágenes BGR."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            raise IOError(f"Cannot read frame {idx}")
        frames.append(frame)
    cap.release()
    return frames


def iter_frames(video_path, with_time=False):
    """Recorre el video en orden y entrega (índice, cuadro) o (índice, tiempo_s, cuadro).

    El tiempo sale del contenedor (CAP_PROP_POS_MSEC), así que es correcto también en
    videos con fps variable (como los del celular).
    """
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if with_time:
            yield idx, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame
        else:
            yield idx, frame
        idx += 1
    cap.release()


def open_writer(out_path, width, height, fps, crf=23):
    """Abre ffmpeg para recibir cuadros BGR por stdin y codificarlos en H.264 sin audio.

    OpenCV decodifica con la matriz BT.601 y ffmpeg convierte BGR -> YUV con BT.601, así
    que el video se etiqueta como tal (smpte170m) para que los reproductores muestren el
    mismo color que ve OpenCV.
    """
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
           "-i", "-", "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-pix_fmt", "yuv420p", "-colorspace", "smpte170m", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-movflags", "+faststart", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def close_writer(writer, name=""):
    """Cierra el proceso de ffmpeg y falla si terminó con error."""
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError(f"ffmpeg failed {name}")


def make_mosaic(images, labels, n_cols=6, tile_width=270):
    """Une imágenes reducidas en una cuadrícula, con una etiqueta en cada una."""
    tile_height = int(images[0].shape[0] * tile_width / images[0].shape[1])
    tiles = []
    for image, label in zip(images, labels):
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % n_cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + n_cols]) for i in range(0, len(tiles), n_cols)]
    return np.vstack(rows)
