"""Lectura básica de video con OpenCV."""
from pathlib import Path

import cv2
import numpy as np


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


def iter_frames(video_path):
    """Recorre el video en orden y entrega (índice, cuadro BGR)."""
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()


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
